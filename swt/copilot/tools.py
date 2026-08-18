"""The copilot's tool surface over a tie session.

This is the *only* way the model reaches the engine. There is no path by which it
touches a numpy array, and no second implementation of the pipeline that could
drift out of step with the one the UI drives -- both call the same
:class:`~swt.session.TieSession` methods.

Three rules hold here, and each exists for a reason that showed up in building
the layers underneath:

**Nothing array-shaped crosses the boundary.** Every tool returns compact JSON --
statistics, intervals, verdicts. A 16,000-sample curve is unreadable to the model
and would exhaust the context on numbers it cannot act on. The session's own
tests already assert this for its methods; the tools inherit it.

**Mutating tools are gated.** Read-only tools always run. Anything that changes
the tie goes through a permit, and a denied call returns an explanation rather
than raising, so the model can tell the user what it wanted to do instead of
retrying blindly. This is defence in depth rather than the only defence -- the
engine's own guardrails already make a 30% squeeze structurally impossible -- but
"the model changed my tie without asking" is a failure of a different kind from
"the model produced a bad tie", and it deserves its own stop.

**Errors come back as text, not exceptions.** A tool that raises kills the loop.
A tool that returns ``{"error": ...}`` lets the model read the message, which is
usually specific enough to act on -- the engine's exceptions were written to be
read by a person, and that turns out to serve a model equally well.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from anthropic import beta_tool

from ..session import SessionError, TieSession

#: Tools that change the tie. Everything else is read-only.
MUTATING = {
    "condition_logs",
    "build_time_depth",
    "calibrate_to_checkshots",
    "run_tie",
    "run_auto_tie",
    "run_uncertainty",
}


@dataclass
class Permit:
    """Decides whether a mutating tool may run.

    ``allow`` may be a bool or a ``callable(tool_name, arguments) -> bool``, so a
    UI can prompt, a script can allow everything, and a diagnosis-only session
    can allow nothing.
    """

    allow: bool | Callable[[str, dict], bool] = False
    #: Every decision, for the audit trail the journal cannot capture (a denied
    #: call changes nothing, so the session never hears about it).
    decisions: list[dict] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.decisions is None:
            self.decisions = []

    def granted(self, tool: str, arguments: dict) -> bool:
        allowed = self.allow(tool, arguments) if callable(self.allow) else bool(self.allow)
        self.decisions.append({"tool": tool, "arguments": arguments, "allowed": allowed})
        return allowed


def _ok(payload: dict) -> str:
    return json.dumps(payload, default=str)


def _error(exc: Exception) -> str:
    """Return the failure as data.

    The engine raises with messages written for a person -- naming the offending
    depth interval, or what to change. Handing that text back unmodified is more
    useful than a generic failure code.
    """
    return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _denied(tool: str, arguments: dict) -> str:
    return json.dumps({
        "denied": True,
        "tool": tool,
        "arguments": arguments,
        "message": (
            "This tool changes the tie and the user has not granted permission. "
            "Nothing was changed. Explain what you wanted to do and why, and let "
            "the user decide -- do not retry."
        ),
    })


def build_tools(session: TieSession, permit: Permit | None = None) -> list:
    """Build the tool list bound to one session.

    Returns Anthropic ``BetaFunctionTool`` objects ready for
    ``client.beta.messages.tool_runner``.
    """
    gate = permit if permit is not None else Permit(allow=False)

    def guard(name: str, arguments: dict):
        """``None`` when permitted, otherwise the refusal to return."""
        return None if gate.granted(name, arguments) else _denied(name, arguments)

    # -- read-only ---------------------------------------------------------

    @beta_tool
    def get_state() -> str:
        """Report where the tie has got to: what is loaded, what steps have run,
        and the current correlation. Call this first in any conversation."""
        try:
            return _ok(session.state())
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @beta_tool
    def describe_logs() -> str:
        """Describe the well logs: depth range, sample interval, curve coverage,
        conditioning edits already applied, and any cycle skips detected."""
        try:
            if session.depth_tvdss is None:
                return _ok({"loaded": False})
            payload = {
                "loaded": True,
                "depth_range_m": [round(float(session.depth_tvdss[0]), 1),
                                  round(float(session.depth_tvdss[-1]), 1)],
                "n_samples": int(session.depth_tvdss.size),
                "has_sonic": session.sonic_us_per_m is not None,
                "has_density": session.density_g_cm3 is not None,
                "cycle_skips": [s.summary() for s in session.cycle_skips],
            }
            if session.conditioning is not None:
                payload["conditioning"] = session.conditioning.summary()
            return _ok(payload)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @beta_tool
    def describe_checkshots() -> str:
        """Describe the checkshot survey, including any intervals whose implied
        velocity is implausible -- those usually mean a bad pick."""
        try:
            if session.checkshots is None:
                return _ok({"loaded": False})
            payload = session.checkshots.summary()
            if session.drift is not None:
                payload["drift"] = session.drift.summary()
            return _ok(payload)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @beta_tool
    def describe_seismic() -> str:
        """Describe the extracted seismic trace: time range, sample interval, and
        where it came from."""
        try:
            if session.seismic is None:
                return _ok({"loaded": False})
            trace = session.seismic
            return _ok({
                "loaded": True,
                "time_range_ms": [round(float(trace.twt[0]) * 1e3, 1),
                                  round(float(trace.twt[-1]) * 1e3, 1)],
                "dt_ms": round(trace.dt * 1e3, 3),
                "n_samples": trace.n,
                "rms": round(trace.rms(), 6),
                "meta": trace.meta,
            })
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @beta_tool
    def get_tie_quality() -> str:
        """Report the current tie's quality: correlation, NRMS, PEP, residual
        shift, the wavelet, and -- importantly -- whether the correlation is
        statistically significant for the window and bandwidth it was measured
        over. A correlation without its significance means little."""
        try:
            if session.result is None:
                return _ok({"tied": False, "message": "no tie has been run yet"})
            payload = {
                "tied": True,
                "metrics": session.result.metrics.summary(),
                "significance": session.result.significance.summary(),
                "wavelet": session.result.wavelet.summary(),
                "total_shift_ms": round(session.result.total_shift_s * 1e3, 2),
                "phase_deg": round(session.result.phase_deg, 1),
            }
            if session.auto is not None:
                payload["warp"] = {
                    "accepted": session.auto.warp_accepted,
                    "verdict": session.auto.verdict(),
                }
                if session.auto.guardrail is not None:
                    payload["warp"]["guardrail"] = session.auto.guardrail.summary()
            graded = session.grade()
            if graded:
                payload["ground_truth"] = graded
            return _ok(payload)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @beta_tool
    def describe_interval(top_m: float, base_m: float) -> str:
        """Examine one depth interval in detail -- the tool for diagnosing a bad
        tie. Reports the sonic and density statistics there, any cycle skips or
        conditioning edits, and the local drift against the checkshots.

        Args:
            top_m: Top of the interval, TVDSS in metres.
            base_m: Base of the interval, TVDSS in metres.
        """
        try:
            import numpy as np

            if session.depth_tvdss is None:
                return _ok({"error": "no logs loaded"})
            if base_m <= top_m:
                return _ok({"error": "base_m must be deeper than top_m"})

            depth = session.depth_tvdss
            inside = (depth >= top_m) & (depth <= base_m)
            if not np.any(inside):
                return _ok({"error": f"no log samples between {top_m} and {base_m} m"})

            payload: dict = {
                "interval_m": [top_m, base_m],
                "n_samples": int(np.count_nonzero(inside)),
            }
            if session.sonic_us_per_m is not None:
                sonic = session.sonic_us_per_m[inside]
                payload["slowness_us_per_m"] = {
                    "min": round(float(np.nanmin(sonic)), 1),
                    "median": round(float(np.nanmedian(sonic)), 1),
                    "max": round(float(np.nanmax(sonic)), 1),
                }
                payload["velocity_m_s"] = {
                    "min": round(float(1e6 / np.nanmax(sonic)), 1),
                    "median": round(float(1e6 / np.nanmedian(sonic)), 1),
                    "max": round(float(1e6 / np.nanmin(sonic)), 1),
                }
            if session.density_g_cm3 is not None:
                density = session.density_g_cm3[inside]
                payload["density_g_cm3"] = {
                    "min": round(float(np.nanmin(density)), 3),
                    "median": round(float(np.nanmedian(density)), 3),
                    "max": round(float(np.nanmax(density)), 3),
                }
            payload["cycle_skips_here"] = [
                s.summary() for s in session.cycle_skips
                if s.end_depth >= top_m and s.start_depth <= base_m
            ]
            if session.conditioning is not None:
                payload["edits_here"] = [
                    e.summary() for e in session.conditioning.edits
                    if e.end_depth >= top_m and e.start_depth <= base_m
                ][:10]
            if session.time_depth is not None:
                twt = session.time_depth.time_at(np.array([top_m, base_m]))
                payload["twt_ms"] = [round(float(t) * 1e3, 1) for t in twt]
            observed = session.observed_drift()
            if observed is not None:
                cs_depth, drift = observed
                near = (cs_depth >= top_m) & (cs_depth <= base_m)
                if np.any(near):
                    payload["drift_here_ms"] = {
                        "min": round(float(np.min(drift[near])) * 1e3, 2),
                        "max": round(float(np.max(drift[near])) * 1e3, 2),
                    }
            return _ok(payload)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @beta_tool
    def get_uncertainty() -> str:
        """Report the uncertainty ensemble if one has been run: the time-depth
        corridor, per-horizon tolerance in milliseconds, and whether the ensemble
        splits into two alignments a cycle apart."""
        try:
            if session.ensemble is None:
                return _ok({"run": False, "message": "no ensemble has been run yet"})
            return _ok(session.ensemble.summary())
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @beta_tool
    def get_journal() -> str:
        """Return the ordered record of every step taken on this tie and what it
        produced. This is the material for a tie report."""
        try:
            return _ok({"journal": session.journal})
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    # -- mutating ----------------------------------------------------------

    @beta_tool
    def condition_logs(despike_threshold: float = 4.0, max_gap_m: float = 5.0) -> str:
        """Despike the sonic, fill short gaps, and re-detect cycle skips. This
        discards any existing time-depth model and tie, since both depend on the
        log.

        Args:
            despike_threshold: Rejection level in robust sigmas. 4.0 is
                conservative; lower removes more.
            max_gap_m: Longest gap to interpolate. Longer gaps are left as gaps
                rather than invented.
        """
        arguments = {"despike_threshold": despike_threshold, "max_gap_m": max_gap_m}
        refusal = guard("condition_logs", arguments)
        if refusal:
            return refusal
        try:
            return _ok(session.condition(
                despike_threshold=despike_threshold, max_gap_m=max_gap_m
            ))
        except (SessionError, ValueError) as exc:
            return _error(exc)

    @beta_tool
    def build_time_depth(
        replacement_velocity: float | None = None,
        twt_at_log_top: float | None = None,
    ) -> str:
        """Integrate the sonic into a time-depth curve. Exactly one of the two
        arguments must be given -- the interval between the seismic datum and the
        top of the log has to be bridged by an assumption or a measurement, and
        an error here shifts the whole synthetic rigidly.

        Args:
            replacement_velocity: Constant velocity (m/s) above the log top.
            twt_at_log_top: Two-way time (s) at the log top, from a checkshot.
        """
        arguments = {"replacement_velocity": replacement_velocity,
                     "twt_at_log_top": twt_at_log_top}
        refusal = guard("build_time_depth", arguments)
        if refusal:
            return refusal
        try:
            return _ok(session.build_time_depth(
                replacement_velocity=replacement_velocity,
                twt_at_log_top=twt_at_log_top,
            ))
        except (SessionError, ValueError) as exc:
            return _error(exc)

    @beta_tool
    def calibrate_to_checkshots(knot_spacing_m: float | None = None) -> str:
        """Correct the time-depth model onto the checkshots via a drift curve.

        Args:
            knot_spacing_m: Spacing of the drift curve's knots in metres. Omit to
                honour every checkshot exactly, which also injects any checkshot
                noise into the velocity field. Widening the spacing smooths the
                correction, and is the fix when calibration reports that it would
                invert the time-depth.
        """
        arguments = {"knot_spacing_m": knot_spacing_m}
        refusal = guard("calibrate_to_checkshots", arguments)
        if refusal:
            return refusal
        try:
            return _ok(session.calibrate(knot_spacing_m=knot_spacing_m))
        except (SessionError, ValueError) as exc:
            return _error(exc)

    @beta_tool
    def run_tie(
        wavelet_length_s: float = 0.128,
        max_shift_s: float = 0.060,
        max_iterations: int = 3,
    ) -> str:
        """Run the deterministic tie: bulk shift, constant-phase scan, and
        least-squares wavelet, iterated. No stretching -- use run_auto_tie for
        that.

        Args:
            wavelet_length_s: Wavelet length in seconds.
            max_shift_s: Half-width of the bulk-shift search in seconds. Keep it
                to a physically plausible static; a wide search invites a
                confident answer a cycle away.
            max_iterations: Maximum phase/shift/wavelet iterations.
        """
        arguments = {"wavelet_length_s": wavelet_length_s,
                     "max_shift_s": max_shift_s, "max_iterations": max_iterations}
        refusal = guard("run_tie", arguments)
        if refusal:
            return refusal
        try:
            return _ok(session.run_tie(
                wavelet_length_s=wavelet_length_s,
                max_shift_s=max_shift_s,
                max_iterations=max_iterations,
            ))
        except (SessionError, ValueError) as exc:
            return _error(exc)

    @beta_tool
    def run_auto_tie(
        velocity_limit_percent: float = 15.0,
        max_warp_shift_s: float = 0.060,
    ) -> str:
        """Run the tie and then attempt a stretch-and-squeeze warp. The warp is
        kept only if the velocity change it implies against the sonic is
        plausible, it was not merely clipped to a plausible value, and it earns
        enough correlation to justify a degree of freedom per sample. A rejected
        warp leaves the tie unwarped and says why.

        Args:
            velocity_limit_percent: Largest velocity change any interval of the
                warp may imply against the sonic.
            max_warp_shift_s: Half-width of the warp's shift search in seconds.
        """
        arguments = {"velocity_limit_percent": velocity_limit_percent,
                     "max_warp_shift_s": max_warp_shift_s}
        refusal = guard("run_auto_tie", arguments)
        if refusal:
            return refusal
        try:
            return _ok(session.run_auto_tie(
                velocity_limit_percent=velocity_limit_percent,
                max_warp_shift_s=max_warp_shift_s,
            ))
        except (SessionError, ValueError) as exc:
            return _error(exc)

    @beta_tool
    def run_uncertainty(n_members: int = 16) -> str:
        """Re-run the whole tie over sampled interpreter choices and report a
        time-depth corridor plus a per-horizon tolerance in milliseconds. Slow:
        each member is a full pipeline run.

        Args:
            n_members: Ensemble size. 16 is usually enough for a P10-P90 band.
        """
        arguments = {"n_members": n_members}
        refusal = guard("run_uncertainty", arguments)
        if refusal:
            return refusal
        try:
            return _ok(session.run_uncertainty(n_members=n_members))
        except (SessionError, ValueError) as exc:
            return _error(exc)

    return [
        get_state, describe_logs, describe_checkshots, describe_seismic,
        get_tie_quality, describe_interval, get_uncertainty, get_journal,
        condition_logs, build_time_depth, calibrate_to_checkshots,
        run_tie, run_auto_tie, run_uncertainty,
    ]
