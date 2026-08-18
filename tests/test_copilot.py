"""Tests for the copilot's tool surface and loop.

The live API call is not tested -- it needs credentials this environment does not
have. Everything between the model's decision and the engine *is* tested, by
driving the loop with a stub client that plays a scripted sequence of tool calls
and executes them for real against a real session. That covers the parts where a
mistake would be dangerous: the permit gate, the JSON boundary, error handling,
and transcript bookkeeping.
"""

from __future__ import annotations

import json

import pytest

from swt.copilot.agent import Copilot, Turn
from swt.copilot.tools import MUTATING, Permit, build_tools
from swt.session import TieSession


# ---------------------------------------------------------------------------
# a stub client that plays a script and really executes the tools


class Block:
    def __init__(self, type, **kwargs):
        self.type = type
        for key, value in kwargs.items():
            setattr(self, key, value)


class Message:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class StubRunner:
    """Plays a scripted conversation, executing tool calls for real."""

    def __init__(self, script, tools):
        self.script = script
        self.tools = {t.name: t for t in tools}
        self.pending = None
        self.results = []

    def __iter__(self):
        for turn in self.script:
            blocks = []
            for spec in turn:
                if spec["type"] == "text":
                    blocks.append(Block("text", text=spec["text"]))
                elif spec["type"] == "thinking":
                    blocks.append(Block("thinking", thinking=spec["thinking"]))
                else:
                    blocks.append(Block(
                        "tool_use", name=spec["name"],
                        input=spec.get("input", {}), id=f"tu_{len(blocks)}",
                    ))
            message = Message(blocks, stop_reason=(
                "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
            ))
            self.pending = message
            yield message

    def generate_tool_call_response(self):
        calls = [b for b in self.pending.content if b.type == "tool_use"]
        if not calls:
            return None
        results = []
        for call in calls:
            output = self.tools[call.name].call(call.input)
            self.results.append({"name": call.name, "output": output})
            results.append({"type": "tool_result", "tool_use_id": call.id,
                            "content": output})
        return {"role": "user", "content": results}


class StubMessages:
    def __init__(self, script):
        self.script = script
        self.calls = []

    def tool_runner(self, **kwargs):
        self.calls.append(kwargs)
        return StubRunner(self.script, kwargs["tools"])


class StubClient:
    def __init__(self, script):
        self.beta = type("Beta", (), {"messages": StubMessages(script)})()

    @property
    def runner_kwargs(self):
        return self.beta.messages.calls[-1]


@pytest.fixture
def session():
    session = TieSession.from_synthetic(seed=3, static_s=0.012, n_cycle_skips=2)
    session.condition()
    session.build_time_depth(replacement_velocity=1900.0)
    session.calibrate()
    session.run_tie()
    return session


# ---------------------------------------------------------------------------


class TestToolSurface:
    def test_every_tool_returns_compact_json(self, session):
        tools = build_tools(session, Permit(allow=True))
        for tool in tools:
            if tool.name in MUTATING:
                continue  # exercised separately; several are slow
            output = tool.call({"top_m": 1000.0, "base_m": 1400.0}
                               if tool.name == "describe_interval" else {})
            parsed = json.loads(output)  # raises if not JSON
            assert isinstance(parsed, dict)
            assert len(output) < 6000, f"{tool.name} returned {len(output)} chars"

    def test_no_tool_leaks_an_array(self, session):
        tools = build_tools(session, Permit(allow=True))
        for tool in tools:
            if tool.name in MUTATING:
                continue
            parsed = json.loads(
                tool.call({"top_m": 1000.0, "base_m": 1400.0}
                          if tool.name == "describe_interval" else {})
            )
            for key, value in parsed.items():
                if isinstance(value, list):
                    assert len(value) < 60, f"{tool.name}.{key} looks like a curve"

    def test_describe_interval_localises_a_cycle_skip(self, session):
        """The diagnostic tool must actually find the thing it is for."""
        tools = {t.name: t for t in build_tools(session, Permit(allow=True))}
        assert session.cycle_skips, "this fixture should carry injected skips"
        skip = session.cycle_skips[0]
        parsed = json.loads(tools["describe_interval"].call(
            {"top_m": skip.start_depth - 50.0, "base_m": skip.end_depth + 50.0}
        ))
        assert parsed["cycle_skips_here"], "the skip was not reported in its own interval"

    def test_tie_quality_carries_the_significance_verdict(self, session):
        tools = {t.name: t for t in build_tools(session, Permit(allow=True))}
        parsed = json.loads(tools["get_tie_quality"].call({}))
        assert parsed["tied"] is True
        assert "verdict" in parsed["significance"]

    def test_errors_come_back_as_data_not_exceptions(self, session):
        tools = {t.name: t for t in build_tools(session, Permit(allow=True))}
        parsed = json.loads(tools["describe_interval"].call(
            {"top_m": 2000.0, "base_m": 1000.0}
        ))
        assert "error" in parsed

    def test_tools_report_absent_inputs_rather_than_failing(self):
        empty = TieSession()
        tools = {t.name: t for t in build_tools(empty, Permit(allow=True))}
        assert json.loads(tools["describe_logs"].call({}))["loaded"] is False
        assert json.loads(tools["describe_seismic"].call({}))["loaded"] is False
        assert json.loads(tools["get_tie_quality"].call({}))["tied"] is False


class TestPermitGate:
    def test_mutating_tools_are_denied_by_default(self, session):
        tools = {t.name: t for t in build_tools(session, Permit(allow=False))}
        before = session.result
        parsed = json.loads(tools["run_tie"].call({}))
        assert parsed["denied"] is True
        assert "do not retry" in parsed["message"].lower()
        assert session.result is before, "a denied call changed the session"

    def test_read_only_tools_are_never_gated(self, session):
        tools = {t.name: t for t in build_tools(session, Permit(allow=False))}
        assert "denied" not in json.loads(tools["get_state"].call({}))
        assert "denied" not in json.loads(tools["get_tie_quality"].call({}))

    def test_permission_can_be_granted(self, session):
        tools = {t.name: t for t in build_tools(session, Permit(allow=True))}
        parsed = json.loads(tools["condition_logs"].call({"despike_threshold": 5.0}))
        assert "denied" not in parsed
        assert parsed["step"] == "condition"

    def test_permission_can_be_decided_per_call(self, session):
        permit = Permit(allow=lambda name, args: name == "condition_logs")
        tools = {t.name: t for t in build_tools(session, permit)}
        assert "denied" not in json.loads(tools["condition_logs"].call({}))
        assert json.loads(tools["run_tie"].call({}))["denied"] is True

    def test_every_decision_is_recorded_including_denials(self, session):
        """A denied call changes nothing, so the journal never sees it."""
        permit = Permit(allow=False)
        tools = {t.name: t for t in build_tools(session, permit)}
        tools["run_tie"].call({})
        tools["run_auto_tie"].call({})
        assert [d["tool"] for d in permit.decisions] == ["run_tie", "run_auto_tie"]
        assert all(d["allowed"] is False for d in permit.decisions)


class TestLoop:
    def test_a_read_only_conversation_runs_to_completion(self, session):
        client = StubClient([
            [{"type": "thinking", "thinking": "check the state first"},
             {"type": "tool_use", "name": "get_state"}],
            [{"type": "tool_use", "name": "get_tie_quality"}],
            [{"type": "text", "text": "The tie correlates at 0.93 and is significant."}],
        ])
        copilot = Copilot(session, client=client)
        turn = copilot.ask("How good is this tie?")

        assert "0.93" in turn.text
        assert [c["name"] for c in turn.tool_calls] == ["get_state", "get_tie_quality"]
        assert turn.thinking
        assert turn.stop_reason == "end_turn"

    def test_the_loop_passes_the_cached_system_prompt_and_thinking(self, session):
        client = StubClient([[{"type": "text", "text": "hello"}]])
        Copilot(session, client=client).ask("hi")

        kwargs = client.runner_kwargs
        assert kwargs["model"] == "claude-opus-5"
        assert kwargs["thinking"]["type"] == "adaptive"
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "correlation is not evidence" in kwargs["system"][0]["text"]

    def test_a_denied_mutation_reaches_the_model_as_text(self, session):
        """The model must be told it was stopped, not silently fail."""
        client = StubClient([
            [{"type": "tool_use", "name": "run_auto_tie"}],
            [{"type": "text", "text": "I wanted to warp but need your permission."}],
        ])
        copilot = Copilot(session, allow_changes=False, client=client)
        turn = copilot.ask("Improve the tie")

        runner = client.beta.messages
        assert "permission" in turn.text
        decisions = copilot.permit_log()
        assert decisions and decisions[0]["allowed"] is False

    def test_a_permitted_mutation_actually_changes_the_session(self, session):
        client = StubClient([
            [{"type": "tool_use", "name": "condition_logs",
              "input": {"despike_threshold": 6.0}}],
            [{"type": "text", "text": "Reconditioned at 6 sigma."}],
        ])
        copilot = Copilot(session, allow_changes=True, client=client)
        copilot.ask("Recondition the logs at 6 sigma")

        # Conditioning invalidates the tie -- proof the real method ran.
        assert session.result is None
        assert session.journal[-1]["step"] == "condition"

    def test_conversation_state_carries_across_turns(self, session):
        client = StubClient([[{"type": "text", "text": "first"}]])
        copilot = Copilot(session, client=client)
        copilot.ask("one")
        first_length = len(copilot.messages)

        client.beta.messages.script = [[{"type": "text", "text": "second"}]]
        copilot.ask("two")
        assert len(copilot.messages) > first_length
        assert copilot.messages[0]["content"] == "one"

    def test_transcript_flattens_for_display(self, session):
        client = StubClient([
            [{"type": "tool_use", "name": "get_state"}],
            [{"type": "text", "text": "All loaded."}],
        ])
        copilot = Copilot(session, client=client)
        copilot.ask("status?")

        transcript = copilot.transcript()
        assert transcript[0] == {"role": "user", "text": "status?"}
        assert any(entry.get("tools") == ["get_state"] for entry in transcript)
        assert any("All loaded." in entry.get("text", "") for entry in transcript)

    def test_report_uses_the_report_instruction(self, session):
        client = StubClient([[{"type": "text", "text": "REPORT"}]])
        copilot = Copilot(session, client=client)
        copilot.write_report()
        assert "tie report" in copilot.messages[0]["content"]

    def test_client_is_not_needed_until_a_question_is_asked(self, session):
        """The UI constructs a Copilot on every rerun; that must not need a key."""
        copilot = Copilot(session)
        assert copilot._client is None
        assert len(copilot.tools) == 14
