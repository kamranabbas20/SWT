"""Checkshot calibration of an integrated sonic log (drift correction).

A sonic log and a checkshot survey measure the same thing and disagree.  The
sonic is a high-frequency, short-offset, invaded-zone measurement; the checkshot
is a direct seismic-frequency time pick.  Their difference,

    drift(z) = TWT_sonic(z) - TWT_checkshot(z)

accumulates with depth (dispersion, invasion, cycle skips, borehole effects).
Calibration removes it while keeping the sonic's fine detail, which is exactly
what a synthetic needs: checkshot times at the tie points, sonic character in
between.

The drift curve is a first-class diagnostic and should always be plotted:

* a smooth ramp is normal dispersion;
* a **step** is a log problem at that depth -- a cycle skip or a bad hole;
* a change of *slope* localises where the sonic stops agreeing with seismic.

Reading the drift curve is usually faster than reading the logs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import TimeDepth


@dataclass(frozen=True)
class DriftResult:
    """A calibrated time-depth model plus the diagnostics that justify it."""

    time_depth: TimeDepth
    knot_depth: np.ndarray
    knot_drift: np.ndarray
    residual_s: np.ndarray
    checkshot_depth: np.ndarray
    max_abs_residual_ms: float
    max_abs_drift_ms: float
    max_drift_slope_ms_per_100m: float

    def summary(self) -> dict:
        """Compact JSON-safe summary -- the shape the copilot tools return."""
        return {
            "n_checkshots": int(self.checkshot_depth.size),
            "n_knots": int(self.knot_depth.size),
            "max_abs_drift_ms": round(self.max_abs_drift_ms, 2),
            "max_abs_residual_ms": round(self.max_abs_residual_ms, 3),
            "max_drift_slope_ms_per_100m": round(self.max_drift_slope_ms_per_100m, 3),
        }


class CalibrationError(ValueError):
    """Raised when a drift correction would produce a non-physical time-depth."""


def calibrate_to_checkshots(
    time_depth: TimeDepth,
    checkshot_depth_tvdss: np.ndarray,
    checkshot_twt: np.ndarray,
    knot_depth: np.ndarray | None = None,
) -> DriftResult:
    """Correct an integrated-sonic time-depth model onto checkshot times.

    Parameters
    ----------
    time_depth:
        The uncalibrated model, normally straight from :func:`integrate_sonic`.
    checkshot_depth_tvdss, checkshot_twt:
        Checkshot depths (m TVDSS) and two-way times (s).  Sorted internally.
    knot_depth:
        Depths at which the piecewise-linear drift curve bends.  Defaults to the
        checkshot depths themselves, which forces the calibrated model through
        every checkshot exactly.  Pass a subset to smooth the correction and
        leave non-zero residuals -- often the better choice, because a noisy
        checkshot honoured exactly injects its noise into the velocity field.

    Notes
    -----
    Outside the knot range the drift is held **constant**, not extrapolated.  An
    extrapolated drift slope is unsupported by data and readily inverts the
    time-depth curve; holding it constant applies an honest bulk shift instead.

    Raises
    ------
    CalibrationError
        If the correction would make time non-monotonic with depth, naming the
        interval responsible.  That normally means the drift knots are too
        closely spaced around a noisy checkshot, or a checkshot time is wrong.
    """
    cs_depth = np.asarray(checkshot_depth_tvdss, dtype=float)
    cs_twt = np.asarray(checkshot_twt, dtype=float)

    if cs_depth.shape != cs_twt.shape:
        raise ValueError("checkshot depth and time arrays differ in shape")
    if cs_depth.size < 2:
        raise ValueError("need at least two checkshots to calibrate")
    if not (np.all(np.isfinite(cs_depth)) and np.all(np.isfinite(cs_twt))):
        raise ValueError("checkshots contain non-finite values")

    order = np.argsort(cs_depth)
    cs_depth, cs_twt = cs_depth[order], cs_twt[order]
    if np.any(np.diff(cs_depth) <= 0):
        raise ValueError("checkshot depths must be distinct")
    if np.any(np.diff(cs_twt) <= 0):
        raise ValueError(
            "checkshot times must increase with depth; the survey is inconsistent"
        )

    drift_at_cs = time_depth.time_at(cs_depth) - cs_twt

    if knot_depth is None:
        knots = cs_depth
        knot_drift = drift_at_cs
    else:
        knots = np.asarray(knot_depth, dtype=float)
        knots = np.sort(knots)
        if knots.size < 2:
            raise ValueError("need at least two drift knots")
        if np.any(np.diff(knots) <= 0):
            raise ValueError("drift knots must be distinct")
        # Sample the observed drift at the knots, so the fitted curve is a
        # coarser piecewise-linear version of the observed one.
        knot_drift = np.interp(knots, cs_depth, drift_at_cs)

    drift_curve = _piecewise_linear_flat_ends(time_depth.depth_tvdss, knots, knot_drift)
    corrected_twt = time_depth.twt - drift_curve

    _assert_still_monotonic(time_depth.depth_tvdss, corrected_twt)

    calibrated = TimeDepth(
        depth_tvdss=time_depth.depth_tvdss,
        twt=corrected_twt,
        provenance=f"{time_depth.provenance} + checkshot calibration",
        meta={**time_depth.meta, "n_checkshots": int(cs_depth.size)},
    )

    residual = calibrated.time_at(cs_depth) - cs_twt
    knot_slope = np.diff(knot_drift) / np.diff(knots) * 1e3 * 100.0  # ms per 100 m

    return DriftResult(
        time_depth=calibrated,
        knot_depth=knots,
        knot_drift=knot_drift,
        residual_s=residual,
        checkshot_depth=cs_depth,
        max_abs_residual_ms=float(np.max(np.abs(residual)) * 1e3),
        max_abs_drift_ms=float(np.max(np.abs(drift_at_cs)) * 1e3),
        max_drift_slope_ms_per_100m=float(np.max(np.abs(knot_slope))) if knot_slope.size else 0.0,
    )


def _piecewise_linear_flat_ends(
    depth: np.ndarray, knots: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Linear between knots, constant outside -- ``np.interp``'s default."""
    return np.interp(depth, knots, values)


def _assert_still_monotonic(depth: np.ndarray, twt: np.ndarray) -> None:
    dt = np.diff(twt)
    bad = np.flatnonzero(dt <= 0)
    if not bad.size:
        return
    first = int(bad[0])
    raise CalibrationError(
        "checkshot calibration inverted the time-depth curve over "
        f"{depth[first]:.1f}-{depth[first + 1]:.1f} m TVDSS "
        f"({bad.size} inverted interval(s)). The drift correction is steeper than "
        "the sonic's own time gradient there -- widen the knot spacing, or check "
        "that checkshot for a bad pick."
    )


def drift_curve(
    time_depth: TimeDepth,
    checkshot_depth_tvdss: np.ndarray,
    checkshot_twt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed drift at the checkshots, for plotting before any correction."""
    cs_depth = np.asarray(checkshot_depth_tvdss, dtype=float)
    cs_twt = np.asarray(checkshot_twt, dtype=float)
    order = np.argsort(cs_depth)
    cs_depth, cs_twt = cs_depth[order], cs_twt[order]
    return cs_depth, time_depth.time_at(cs_depth) - cs_twt
