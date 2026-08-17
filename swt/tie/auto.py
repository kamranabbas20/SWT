"""The auto-tie: the M1 loop, then a warp that must earn its place.

The sequence is deliberate. Warping runs **last**, after the bulk shift, the
phase scan and the deterministic wavelet have done their work, for a reason that
is easy to state and easy to get wrong: a warp handed a synthetic that is 40 ms
out will spend its strain budget removing that static, and the stretch it then
reports is an artefact of the static rather than a statement about velocity. The
guardrail would be judging the wrong quantity. Remove what a single number can
remove before reaching for a number per sample.

And the warp is not accepted merely because it improves the correlation. It is
converted back into the velocity change it claims against the sonic and checked
against what rock can do. **A rejected warp is discarded even when it correlates
better than the tie that passes** -- which is the entire thesis of this package
expressed in one branch of one function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..petro.elastic import acoustic_impedance
from ..qc.metrics import evaluate
from ..qc.significance import assess
from ..synth.convolve import synthesise
from ..synth.resample import reflectivity_in_time
from ..timedepth.model import TimeDepth
from ..trace import Trace
from ..units import slowness_to_velocity
from ..wavelet.deterministic import deterministic_wavelet
from .dtw import Warp, apply_warp, auto_warp
from .guardrail import GuardrailReport, check_warp, strain_limit_for
from .iterate import TieResult, tie


@dataclass(frozen=True)
class AutoTieResult:
    """The outcome of an auto-tie, whether or not the warp survived."""

    result: TieResult
    before_warp: TieResult
    warp: Warp | None
    coarse_warp: Warp | None
    guardrail: GuardrailReport | None
    warp_accepted: bool
    rejection_reason: str | None = None

    @property
    def correlation_gain(self) -> float:
        """How much the warp improved the correlation -- zero if it was rejected."""
        if not self.warp_accepted:
            return 0.0
        return self.result.metrics.correlation - self.before_warp.metrics.correlation

    def summary(self) -> dict:
        out = {
            **self.result.summary(),
            "warp_accepted": self.warp_accepted,
            "correlation_before_warp": round(self.before_warp.metrics.correlation, 4),
            "correlation_gain": round(self.correlation_gain, 4),
        }
        if self.warp is not None:
            out["warp"] = self.warp.summary()
        if self.guardrail is not None:
            out["guardrail"] = self.guardrail.summary()
        if self.rejection_reason:
            out["rejection_reason"] = self.rejection_reason
        return out

    def verdict(self) -> str:
        """What happened, in a sentence, for the report and the copilot."""
        if self.warp is None:
            return "No warp was attempted."
        if self.warp_accepted:
            return (
                f"Warp accepted: correlation {self.before_warp.metrics.correlation:.3f} "
                f"-> {self.result.metrics.correlation:.3f} for at most "
                f"{self.guardrail.max_change_percent:.1f}% velocity change against the sonic."
            )
        return f"Warp rejected, tie left unwarped. {self.rejection_reason}"


def auto_tie(
    seismic: Trace,
    time_depth: TimeDepth,
    dt_us_per_m: np.ndarray,
    rho_g_cm3: np.ndarray,
    velocity_limit_percent: float = 15.0,
    tolerated_fraction: float = 0.0,
    max_warp_shift_s: float = 0.060,
    curvature_penalty: float = 0.5,
    anchors: dict | None = None,
    wavelet_length_s: float = 0.128,
    require_improvement: float = 0.05,
    max_saturated_fraction: float = 0.50,
    **tie_kwargs,
) -> AutoTieResult:
    """Tie, then warp -- and keep the warp only if it is geologically admissible.

    Parameters
    ----------
    velocity_limit_percent:
        Largest velocity change against the sonic that any interval of the warp
        may imply. This single number sets both the guardrail's acceptance test
        and the solver's strain limit (via
        :func:`~swt.tie.guardrail.strain_limit_for`), so the solver cannot search
        a region the guardrail would reject.
    tolerated_fraction:
        Fraction of the log allowed past the limit before the warp is rejected.
    require_improvement:
        Minimum correlation gain for the warp to be worth its complexity. A warp
        that adds a per-sample degree of freedom to buy a hundredth of
        correlation has not found anything; it has fitted noise, and the simpler
        tie is the better answer.

        The 0.05 default is measured. Over forward-modelled cases, warps that
        *improved* the time-depth gained between 0.10 and 0.25 correlation, while
        every warp that *degraded* it gained 0.036 or less -- a clean gap with a
        threefold margin. The damaging cases all shared a cause: an injected
        cycle skip put a spurious event in the synthetic, and the warp stretched
        locally to align that artefact with real seismic, buying a little
        correlation and moving the time-depth several milliseconds. The lesson
        generalises past the threshold -- **a warp that gains only a little is
        usually chasing a defect in the log**, and the fix is to repair the log,
        not to let the warp absorb it.
    max_saturated_fraction:
        Largest fraction of the warp allowed to sit at its strain limit. A warp
        pinned at the limit over most of its length has been *clipped* by the
        constraint rather than solved: the correction genuinely needed is larger
        than the permitted velocity change. Such a warp passes the guardrail --
        it was clipped to a passing value, after all -- looks smooth, and
        improves the correlation, while leaving the tie badly wrong. It is
        therefore rejected, and the rejection points at the inputs, which is
        where the problem actually is.

        The 0.5 default is empirical, not a round number. Over 24 forward-modelled
        cases carrying a localised sonic anomaly, every warp that improved the
        time-depth saturated at 0.42 or below, while the one warp that degraded it
        -- raising the correlation from 0.73 to 0.89 while nearly doubling the
        time-depth error -- saturated at 0.59. A threshold of 0.5 accepted 21 of
        24 improvements and rejected that one degradation. The margin between the
        two populations is real but not wide, so treat this as a tuned default
        rather than a law, and re-measure it if the wavelet band or the strain
        limit change materially.
    anchors:
        ``{time_s: shift_s}`` hard constraints, e.g. from formation tops.

    Returns
    -------
    AutoTieResult
        ``result`` is the tie to use. When the warp is rejected it is identical
        to ``before_warp``, and ``rejection_reason`` says why.
    """
    if not 0.0 < velocity_limit_percent < 100.0:
        raise ValueError("velocity_limit_percent must lie strictly between 0 and 100")

    base = tie(
        seismic, time_depth, dt_us_per_m, rho_g_cm3,
        wavelet_length_s=wavelet_length_s, **tie_kwargs,
    )

    strain = strain_limit_for(velocity_limit_percent)
    coarse, fine = auto_warp(
        seismic, base.synthetic,
        strain_limit=strain,
        max_shift_s=max_warp_shift_s,
        curvature_penalty=curvature_penalty,
        anchors=anchors,
    )

    warped_td = apply_warp(base.time_depth, fine)
    guardrail = check_warp(
        base.time_depth, warped_td,
        limit_percent=velocity_limit_percent,
        tolerated_fraction=tolerated_fraction,
    )

    # Saturation is checked before the guardrail, because a clipped warp *passes*
    # the guardrail by construction -- it was clipped to a passing value. Judging
    # it on its velocity claim alone would accept the one warp whose velocity
    # claim carries no information.
    saturated = fine.saturated_fraction()
    if saturated > max_saturated_fraction:
        return AutoTieResult(
            result=base, before_warp=base, warp=fine, coarse_warp=coarse,
            guardrail=guardrail, warp_accepted=False,
            rejection_reason=(
                f"Warp is pinned at its strain limit over {saturated:.0%} of the log "
                f"(limit {max_saturated_fraction:.0%}). The correction actually needed is "
                f"larger than a {velocity_limit_percent:.0f}% velocity change can supply, so "
                "this warp is the answer clipped rather than the answer. Fix the input "
                "instead: check the replacement velocity above the log top, whether the "
                "checkshots were applied, and that the trace is at the right well."
                + (
                    f" The warp also reaches its {max_warp_shift_s * 1e3:.0f} ms shift limit."
                    if fine.at_shift_limit(max_warp_shift_s) else ""
                )
            ),
        )

    if not guardrail.passed:
        return AutoTieResult(
            result=base, before_warp=base, warp=fine, coarse_warp=coarse,
            guardrail=guardrail, warp_accepted=False,
            rejection_reason=guardrail.verdict(),
        )

    warped = _rebuild(
        seismic, warped_td, dt_us_per_m, rho_g_cm3, base, wavelet_length_s, fine
    )

    gain = warped.metrics.correlation - base.metrics.correlation
    if gain < require_improvement:
        return AutoTieResult(
            result=base, before_warp=base, warp=fine, coarse_warp=coarse,
            guardrail=guardrail, warp_accepted=False,
            rejection_reason=(
                f"Warp is admissible but gains only {gain:+.4f} correlation, under the "
                f"{require_improvement:.3f} threshold. A per-sample degree of freedom that "
                "buys nothing is fitting noise; the simpler tie is the better answer."
            ),
        )

    return AutoTieResult(
        result=warped, before_warp=base, warp=fine, coarse_warp=coarse,
        guardrail=guardrail, warp_accepted=True,
    )


def _rebuild(
    seismic: Trace,
    warped_td: TimeDepth,
    dt_us_per_m: np.ndarray,
    rho_g_cm3: np.ndarray,
    base: TieResult,
    wavelet_length_s: float,
    warp: Warp,
) -> TieResult:
    """Re-estimate the wavelet and QC on the warped time-depth.

    The wavelet must be re-estimated after warping -- it was fitted to the
    unwarped reflectivity, and keeping it would flatter the warp by measuring it
    against a wavelet that already absorbed the misalignment the warp just fixed.
    """
    impedance = acoustic_impedance(slowness_to_velocity(dt_us_per_m), rho_g_cm3)
    reflectivity = reflectivity_in_time(
        warped_td.twt, impedance, seismic.dt,
        t_start=float(seismic.twt[0]), t_end=float(seismic.twt[-1]),
    )
    window = reflectivity.valid_window()

    observed = seismic.window(*window)
    aligned = np.interp(observed.twt, reflectivity.twt, reflectivity.rc)
    wavelet = deterministic_wavelet(
        aligned, observed.amplitude, seismic.dt, wavelet_length_s
    )
    wavelet, absorbed = wavelet.recentred()
    if absorbed:
        warped_td = warped_td.shifted(absorbed)
        reflectivity = reflectivity_in_time(
            warped_td.twt, impedance, seismic.dt,
            t_start=float(seismic.twt[0]), t_end=float(seismic.twt[-1]),
        )
        window = reflectivity.valid_window()

    synthetic = synthesise(reflectivity, wavelet)
    metrics = evaluate(seismic, synthetic, window=window)
    low, high = wavelet.bandwidth()
    significance = assess(metrics.correlation, window[1] - window[0], max(high - low, 1e-6))

    history = list(base.history) + [
        {"step": "warp", **warp.summary()},
        {"step": "qc after warp", **metrics.summary()},
    ]

    return TieResult(
        time_depth=warped_td,
        wavelet=wavelet,
        synthetic=synthetic,
        reflectivity=reflectivity,
        metrics=metrics,
        significance=significance,
        window_s=window,
        total_shift_s=base.total_shift_s + absorbed,
        phase_deg=base.phase_deg,
        iterations=base.iterations,
        history=history,
        phase_scan=base.phase_scan,
        shift_search=base.shift_search,
    )
