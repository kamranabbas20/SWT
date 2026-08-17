"""The M1 tie sequence: the deterministic outer loop, no warping yet.

The order of operations here is not arbitrary, and getting it wrong produces a
tie that is confidently misaligned.  The sequence is:

1. **Coarse align on the envelope.**  Phase is unknown at this point, so the
   comparison must be phase-blind.  Correlating traces here locks onto the wrong
   half-cycle whenever the data carries significant residual phase, and converts
   a phase error into a time error that nothing downstream can undo.
2. **Scan constant phase**, now that the events are roughly on top of each other.
3. **Fine align on the trace**, which is the sharper estimator once phase is known.
4. **Re-estimate the wavelet deterministically**, which needs an approximately
   correct time-depth to be meaningful at all.
5. **Repeat 2-4** until the correlation stops improving.

Time shift and constant phase are partly interchangeable for a band-limited
signal -- at 30 Hz, 12 ms of shift looks very like 130 degrees of rotation -- so
they cannot be estimated independently in one pass.  The loop separates them by
alternating, starting from the estimator that is blind to the other.

Warping (stretch and squeeze) is deliberately absent: that is M3, and it belongs
*after* this sequence has removed the static and the phase, so that its stretch
budget is spent on velocity error rather than on a static.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..petro.elastic import acoustic_impedance
from ..qc.metrics import TieMetrics, evaluate
from ..qc.significance import Significance, assess
from ..synth.convolve import synthesise
from ..synth.resample import TimeReflectivity, reflectivity_in_time
from ..timedepth.model import TimeDepth
from ..trace import Trace
from ..units import slowness_to_velocity
from ..wavelet.core import Wavelet
from ..wavelet.deterministic import deterministic_wavelet
from ..wavelet.statistical import statistical_wavelet
from .align import apply_bulk_shift, find_bulk_shift
from .phase import scan_constant_phase


@dataclass(frozen=True)
class TieResult:
    """A finished first-pass tie and everything needed to judge it."""

    time_depth: TimeDepth
    wavelet: Wavelet
    synthetic: Trace
    reflectivity: TimeReflectivity
    metrics: TieMetrics
    significance: Significance
    window_s: tuple[float, float]
    total_shift_s: float
    phase_deg: float
    iterations: int
    history: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        """Compact JSON-safe summary -- the shape the copilot tools return."""
        return {
            "correlation": round(self.metrics.correlation, 4),
            "nrms_percent": round(self.metrics.nrms_percent, 2),
            "pep": round(self.metrics.pep, 4),
            "total_shift_ms": round(self.total_shift_s * 1e3, 2),
            "phase_deg": round(self.phase_deg, 1),
            "window_ms": [round(t * 1e3, 1) for t in self.window_s],
            "iterations": self.iterations,
            "significant": self.significance.is_significant,
            "verdict": self.significance.verdict(),
            "wavelet": self.wavelet.summary(),
        }

    @property
    def residual_shift_ms(self) -> float:
        return self.metrics.best_lag_s * 1e3


def tie(
    seismic: Trace,
    time_depth: TimeDepth,
    dt_us_per_m: np.ndarray,
    rho_g_cm3: np.ndarray,
    wavelet_length_s: float = 0.128,
    max_shift_s: float = 0.060,
    max_iterations: int = 3,
    tolerance: float = 1e-3,
    ridge: float = 1e-2,
    deterministic: bool = True,
) -> TieResult:
    """Run the deterministic first-pass tie.

    Parameters
    ----------
    seismic:
        The extracted trace at the well.
    time_depth:
        Starting time-depth model, normally checkshot-calibrated.
    dt_us_per_m, rho_g_cm3:
        Conditioned logs on the *same* depth axis as ``time_depth``.
    deterministic:
        Whether to re-estimate the wavelet by least squares after aligning.  Turn
        off on noisy or short logs, where the statistical estimate is steadier.

    Returns
    -------
    TieResult
    """
    if dt_us_per_m.shape != rho_g_cm3.shape:
        raise ValueError("slowness and density logs differ in shape")
    if dt_us_per_m.shape != time_depth.depth_tvdss.shape:
        raise ValueError(
            f"logs ({dt_us_per_m.shape}) and time-depth model "
            f"({time_depth.depth_tvdss.shape}) are on different depth axes"
        )
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    impedance = acoustic_impedance(slowness_to_velocity(dt_us_per_m), rho_g_cm3)
    dt = seismic.dt
    history: list[dict] = []

    def build(td: TimeDepth) -> tuple[TimeReflectivity, tuple[float, float]]:
        rc = reflectivity_in_time(
            td.twt, impedance, dt, t_start=float(seismic.twt[0]), t_end=float(seismic.twt[-1])
        )
        return rc, rc.valid_window()

    # -- step 1: statistical wavelet and phase-blind coarse alignment --------
    current_td = time_depth
    rc, window = build(current_td)
    wavelet = statistical_wavelet(
        seismic.window(*window).amplitude, dt, wavelet_length_s
    )

    coarse = find_bulk_shift(
        seismic, synthesise(rc, wavelet), window=window, max_shift_s=max_shift_s, method="envelope"
    )
    current_td = apply_bulk_shift(current_td, coarse)
    total_shift = coarse.shift_s
    history.append({"step": "coarse envelope shift", **coarse.summary()})

    phase_deg = 0.0
    result: TieResult | None = None
    previous_correlation = -np.inf

    # -- steps 2-4, repeated ------------------------------------------------
    for iteration in range(1, max_iterations + 1):
        rc, window = build(current_td)

        scan = scan_constant_phase(seismic, rc, wavelet, window=window)
        wavelet = scan.wavelet
        phase_deg += scan.best_phase_deg
        history.append({"step": f"phase scan {iteration}", **scan.summary()})

        fine = find_bulk_shift(
            seismic, scan.synthetic, window=window, max_shift_s=max_shift_s, method="trace"
        )
        if fine.shift_s:
            current_td = apply_bulk_shift(current_td, fine)
            total_shift += fine.shift_s
            rc, window = build(current_td)
        history.append({"step": f"fine trace shift {iteration}", **fine.summary()})

        if deterministic:
            observed = seismic.window(*window)
            aligned_rc = np.interp(observed.twt, rc.twt, rc.rc)
            wavelet = deterministic_wavelet(
                aligned_rc, observed.amplitude, dt, wavelet_length_s, ridge=ridge
            )

            # A least-squares wavelet will happily absorb a residual static into
            # its own energy offset, which converges the loop on a tie that is
            # still a few milliseconds out -- and hides the error where no later
            # shift search can find it.  Move any such offset back into the
            # time-depth model, where it stays visible and correctable.
            wavelet, absorbed = wavelet.recentred()
            if absorbed:
                current_td = current_td.shifted(absorbed)
                total_shift += absorbed
                rc, window = build(current_td)
            history.append(
                {
                    "step": f"deterministic wavelet {iteration}",
                    "recentred_by_ms": round(absorbed * 1e3, 2),
                    **wavelet.summary(),
                }
            )

        synthetic = synthesise(rc, wavelet)
        metrics = evaluate(seismic, synthetic, window=window, max_lag_s=max_shift_s)
        low, high = wavelet.bandwidth()
        significance = assess(
            metrics.correlation, window[1] - window[0], max(high - low, 1e-6)
        )
        history.append({"step": f"qc {iteration}", **metrics.summary()})

        result = TieResult(
            time_depth=current_td,
            wavelet=wavelet,
            synthetic=synthetic,
            reflectivity=rc,
            metrics=metrics,
            significance=significance,
            window_s=window,
            total_shift_s=total_shift,
            phase_deg=phase_deg,
            iterations=iteration,
            history=history,
        )

        if metrics.correlation - previous_correlation < tolerance:
            break
        previous_correlation = metrics.correlation

    assert result is not None  # loop runs at least once
    return result
