"""The copilot loop: Claude driving the tie engine through its tool surface.

Thin by design. The SDK's tool runner owns the request/execute/loop cycle, the
tools own the engine access, and this module owns only the wiring: model choice,
caching, streaming, and turning the runner's output into something a UI can
render.

**What is and is not verified.** The tool surface, the permit gate, and the
transcript handling are covered by tests that drive the loop with a stub client,
so the parts that touch the engine are exercised without a network. The live API
call itself is not covered -- it needs credentials this environment does not
have. Treat :meth:`Copilot.ask` as unproven against the real service until it has
been run once with a key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..session import TieSession
from .prompts import REPORT_INSTRUCTION, SYSTEM
from .tools import Permit, build_tools

#: Opus 5, with adaptive thinking. Reasoning is summarised rather than omitted
#: because the user is watching a diagnosis unfold and the default -- thinking
#: silently, then answering -- reads as a long stall.
MODEL = "claude-opus-5"
MAX_TOKENS = 16000
THINKING: dict[str, Any] = {"type": "adaptive", "display": "summarized"}


@dataclass
class Turn:
    """One exchange, flattened into something a UI can render."""

    text: str
    tool_calls: list[dict] = field(default_factory=list)
    thinking: str = ""
    stop_reason: str | None = None

    def summary(self) -> dict:
        return {
            "text": self.text,
            "tools_used": [c["name"] for c in self.tool_calls],
            "stop_reason": self.stop_reason,
        }


class Copilot:
    """Claude, wired to one tie session.

    Parameters
    ----------
    session:
        The tie to work on. The copilot mutates it through the same methods the
        UI calls -- there is no second implementation.
    allow_changes:
        ``True`` lets the model run the pipeline; ``False`` (the default) makes
        it a diagnostic assistant that can look but not touch. May also be a
        ``callable(tool_name, arguments) -> bool`` for per-call approval.
    client:
        An ``anthropic.Anthropic`` instance. Injectable so the loop can be tested
        without a network.
    """

    def __init__(
        self,
        session: TieSession,
        allow_changes: bool | Callable[[str, dict], bool] = False,
        client: Any | None = None,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self.session = session
        self.permit = Permit(allow=allow_changes)
        self.tools = build_tools(session, self.permit)
        self.model = model
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self._client = client

    @property
    def client(self):
        """The Anthropic client, created on first use.

        Deferred so that constructing a Copilot -- which the UI does on every
        rerun -- does not require credentials until a question is actually asked.
        """
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    # -- conversation ------------------------------------------------------

    def ask(self, question: str) -> Turn:
        """Put a question to the copilot and run the tool loop to completion."""
        self.messages.append({"role": "user", "content": question})
        return self._run()

    def write_report(self) -> Turn:
        """Ask for a written tie report, built from the journal and the QC."""
        self.messages.append({"role": "user", "content": REPORT_INSTRUCTION})
        return self._run()

    def _run(self) -> Turn:
        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=self.max_tokens,
            # The system prompt and the tool list are byte-stable, so the cache
            # breakpoint goes after them and the volatile conversation follows.
            system=[{
                "type": "text",
                "text": SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            thinking=THINKING,
            tools=self.tools,
            messages=self.messages,
        )

        turn = Turn(text="")
        last = None
        for message in runner:
            last = message
            self.messages.append({"role": "assistant", "content": message.content})
            response = runner.generate_tool_call_response()
            if response is not None:
                self.messages.append(response)
            self._absorb(message, turn)

        turn.stop_reason = getattr(last, "stop_reason", None)
        return turn

    def _absorb(self, message: Any, turn: Turn) -> None:
        """Flatten one assistant message into the turn being built."""
        for block in message.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                turn.text += block.text
            elif kind == "thinking":
                turn.thinking += getattr(block, "thinking", "") or ""
            elif kind == "tool_use":
                turn.tool_calls.append({"name": block.name, "input": dict(block.input)})

    # -- reporting ---------------------------------------------------------

    def transcript(self) -> list[dict]:
        """The conversation so far, flattened for display or export."""
        out: list[dict] = []
        for message in self.messages:
            content = message["content"]
            if isinstance(content, str):
                out.append({"role": message["role"], "text": content})
                continue
            text = "".join(
                getattr(b, "text", "") for b in content
                if getattr(b, "type", None) == "text"
            )
            calls = [
                getattr(b, "name", "") for b in content
                if getattr(b, "type", None) == "tool_use"
            ]
            if text or calls:
                out.append({"role": message["role"], "text": text, "tools": calls})
        return out

    def permit_log(self) -> list[dict]:
        """Every approval decision, including the denials the journal cannot see.

        A denied call changes nothing, so the session never records it -- but
        "the model wanted to rewrite the time-depth and was stopped" is exactly
        the kind of thing an audit trail should contain.
        """
        return list(self.permit.decisions)
