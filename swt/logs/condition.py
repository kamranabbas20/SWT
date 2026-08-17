"""Log conditioning: spikes, cycle skips, bad hole, and gaps.

Conditioning is not cosmetic.  An integrated sonic accumulates every error in
the log, so a single 4 m cycle skip -- a stretch where the tool locked onto the
wrong arrival and the slowness roughly doubles -- adds a permanent few
milliseconds to every sample below it and tilts the whole tie.

The design rule here is that **nothing is silently edited**.  Every routine
returns the corrected log *and* a report of what it changed and where, because
"the software fixed it" and "the software hid it" are the same operation
observed from different distances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Edit:
    """One contiguous interval that conditioning changed."""

    kind: str
    start_depth: float
    end_depth: float
    n_samples: int
    detail: str = ""

    def summary(self) -> dict:
        return {
            "kind": self.kind,
            "interval_m": [round(self.start_depth, 2), round(self.end_depth, 2)],
            "n_samples": self.n_samples,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ConditioningReport:
    """What conditioning did to a log, in full."""

    log: np.ndarray
    edits: list[Edit] = field(default_factory=list)
    remaining_nan: int = 0

    def summary(self) -> dict:
        by_kind: dict[str, int] = {}
        for edit in self.edits:
            by_kind[edit.kind] = by_kind.get(edit.kind, 0) + 1
        return {
            "n_edits": len(self.edits),
            "edits_by_kind": by_kind,
            "remaining_nan": self.remaining_nan,
            "worst_intervals": [e.summary() for e in self.edits[:8]],
        }

    @property
    def is_clean(self) -> bool:
        return not self.edits and self.remaining_nan == 0


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median, NaN-aware, with shrinking windows at the edges."""
    n = values.size
    half = max(int(window) // 2, 1)
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        segment = values[lo:hi]
        finite = segment[np.isfinite(segment)]
        out[i] = np.median(finite) if finite.size else np.nan
    return out


def despike(
    log: np.ndarray,
    depth: np.ndarray,
    window: int = 15,
    threshold: float = 4.0,
    max_width: int = 5,
) -> ConditioningReport:
    """Replace narrow outliers with the local median.

    Outliers are found by the median absolute deviation, not the standard
    deviation: a spike inflates the standard deviation enough to hide itself,
    which is why a sigma-clip on raw logs under-detects exactly the samples it
    was deployed to catch.

    Parameters
    ----------
    threshold:
        Rejection level in robust sigmas (``1.4826 * MAD``).  4.0 is
        conservative -- it removes obvious bad readings and leaves real thin
        beds alone.
    max_width:
        Longest run of consecutive outliers still treated as a spike.  This
        guard is what stops despiking from eating the log.  A *spike* is narrow
        by definition -- one or two bad samples.  A sustained departure from the
        local median is either real geology (a bed boundary, which a median
        filter reads as an outlier on both sides) or a cycle skip, and both must
        survive this function: geology because it is the signal, and cycle skips
        because :func:`detect_cycle_skips` reports them for a human to judge,
        which cannot happen if despiking has already quietly smoothed them away.
    """
    values = np.asarray(log, dtype=float).copy()
    depth = np.asarray(depth, dtype=float)
    if values.shape != depth.shape:
        raise ValueError("log and depth differ in shape")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    smooth = rolling_median(values, window)
    residual = values - smooth
    finite = residual[np.isfinite(residual)]
    if not finite.size:
        return ConditioningReport(values, [], int(np.count_nonzero(~np.isfinite(values))))

    scale = _robust_scale(residual, values)
    if scale <= 0:
        return ConditioningReport(values, [], int(np.count_nonzero(~np.isfinite(values))))

    flagged = np.isfinite(residual) & (np.abs(residual) > threshold * scale)

    edits: list[Edit] = []
    narrow = np.zeros_like(flagged)
    for lo, hi in _runs(flagged):
        if hi - lo > max_width:
            continue  # too wide to be a spike; leave it for the human to judge
        narrow[lo:hi] = True
        edits.append(
            Edit(
                "spike",
                float(depth[lo]),
                float(depth[hi - 1]),
                hi - lo,
                f"|residual| up to {np.nanmax(np.abs(residual[lo:hi])) / scale:.1f} sigma",
            )
        )

    values[narrow] = smooth[narrow]

    return ConditioningReport(values, edits, int(np.count_nonzero(~np.isfinite(values))))


def detect_cycle_skips(
    dt_us_per_m: np.ndarray,
    depth: np.ndarray,
    ratio: float = 1.6,
    window: int = 201,
    min_samples: int = 3,
) -> list[Edit]:
    """Find intervals where the sonic appears to have skipped a cycle.

    A cycle skip is a *sustained* step in slowness -- the tool locks onto a
    later arrival and stays there for a few metres -- as distinct from a spike,
    which is one sample.  It is detected as a run of samples whose slowness
    exceeds the local trend by more than ``ratio``.

    Parameters
    ----------
    window:
        Samples in the running median that defines the local trend.  **It must
        be several times wider than the widest skip to be found.**  Once a skip
        fills much of the window the median moves onto the skipped values, the
        trend becomes the anomaly, and the ratio falls back towards 1 -- so a
        skip too wide for the window is invisible, not obvious.  The default of
        201 samples is about 30 m on a 0.15 m log, which comfortably finds the
        few-metre skips that occur in practice.  Widen it to hunt for longer
        ones.
    min_samples:
        Shortest run reported.  Below this a departure is a spike, and
        :func:`despike` handles it.

    Returns
    -------
    Detection only -- nothing is edited.  Whether to repair the interval, cut
    it, or leave it and widen the checkshot knots is a judgement call, and the
    drift curve is usually the better evidence for making it.
    """
    dt = np.asarray(dt_us_per_m, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if dt.shape != depth.shape:
        raise ValueError("log and depth differ in shape")
    if ratio <= 1.0:
        raise ValueError("ratio must exceed 1")

    trend = rolling_median(dt, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.where(np.isfinite(trend) & (trend > 0), dt / trend, np.nan)

    suspect = np.isfinite(relative) & (relative > ratio)
    return [
        Edit(
            "cycle_skip",
            float(depth[lo]),
            float(depth[hi - 1]),
            hi - lo,
            f"slowness up to {np.nanmax(relative[lo:hi]):.2f}x the local trend",
        )
        for lo, hi in _runs(suspect)
        if hi - lo >= min_samples
    ]


def flag_bad_hole(
    caliper: np.ndarray,
    depth: np.ndarray,
    bit_size: float,
    tolerance: float = 1.15,
) -> list[Edit]:
    """Flag washed-out intervals where the density log is not to be trusted.

    Density is a pad measurement: where the borehole is enlarged the pad loses
    contact and reads too low, which turns into a fictitious soft layer and a
    fictitious reflector.  Sonic is far less affected, so a mismatch between the
    two is often a hole problem rather than a rock property.
    """
    cal = np.asarray(caliper, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if cal.shape != depth.shape:
        raise ValueError("caliper and depth differ in shape")
    if bit_size <= 0:
        raise ValueError("bit_size must be positive")

    washed = np.isfinite(cal) & (cal > bit_size * tolerance)
    return [
        Edit(
            "bad_hole",
            float(depth[lo]),
            float(depth[hi - 1]),
            hi - lo,
            f"caliper up to {np.nanmax(cal[lo:hi]):.2f} vs bit {bit_size:.2f}",
        )
        for lo, hi in _runs(washed)
    ]


def fill_gaps(
    log: np.ndarray,
    depth: np.ndarray,
    max_gap_m: float = 5.0,
) -> ConditioningReport:
    """Interpolate short gaps; leave long ones as NaN.

    Short gaps are interpolated because integration cannot proceed through a
    NaN.  Long gaps are *not*, on purpose: inventing hundreds of metres of log
    produces a synthetic with invented reflectors, and the honest response is to
    stop and make the user decide -- fill from an offset well, use a trend, or
    restrict the tie window.
    """
    values = np.asarray(log, dtype=float).copy()
    depth = np.asarray(depth, dtype=float)
    if values.shape != depth.shape:
        raise ValueError("log and depth differ in shape")

    missing = ~np.isfinite(values)
    if not np.any(missing):
        return ConditioningReport(values, [], 0)

    known = np.flatnonzero(~missing)
    if known.size < 2:
        raise ValueError("log has fewer than two finite samples; nothing to interpolate from")

    edits: list[Edit] = []
    for lo, hi in _runs(missing):
        span = float(depth[min(hi, depth.size - 1)] - depth[lo])
        interior = lo > known[0] and hi <= known[-1] + 1
        if interior and span <= max_gap_m:
            values[lo:hi] = np.interp(depth[lo:hi], depth[known], values[known])
            edits.append(Edit("gap_filled", float(depth[lo]), float(depth[hi - 1]), hi - lo,
                              f"{span:.1f} m interpolated"))
        else:
            reason = "beyond the logged interval" if not interior else f"{span:.1f} m exceeds max_gap_m"
            edits.append(Edit("gap_left", float(depth[lo]), float(depth[hi - 1]), hi - lo, reason))

    return ConditioningReport(values, edits, int(np.count_nonzero(~np.isfinite(values))))


def _robust_scale(residual: np.ndarray, values: np.ndarray) -> float:
    """A non-degenerate robust scale for the despike threshold.

    The obvious choice -- ``1.4826 * MAD`` of the residual from the local median
    -- collapses to exactly zero on a smooth log.  Where the log is monotone
    across the median window the median *is* the centre sample, so more than half
    the residuals are identically zero, the MAD is zero, and a threshold built
    from it rejects nothing.  A log that is clean apart from a few spikes is
    precisely the case where that happens, and precisely the case despiking
    exists for, so the degenerate branch cannot simply return "nothing found".

    The fallback is the sample-to-sample variability of the log itself, which is
    non-zero for any log that is not exactly constant.
    """
    finite = residual[np.isfinite(residual)]
    if not finite.size:
        return 0.0

    mad = float(np.median(np.abs(finite - np.median(finite))))
    if mad > 0:
        return 1.4826 * mad

    differences = np.diff(values[np.isfinite(values)])
    if not differences.size:
        return 0.0
    difference_mad = float(np.median(np.abs(differences - np.median(differences))))
    return 1.4826 * difference_mad


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs as half-open ``(start, stop)`` index pairs."""
    if not np.any(mask):
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[::2].tolist(), changes[1::2].tolist()))
