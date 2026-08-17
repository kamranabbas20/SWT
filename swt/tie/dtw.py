"""Constrained dynamic warping: the stretch-and-squeeze solver.

Unconstrained dynamic time warping will align anything with anything. Given two
traces of noise it will return a path, and that path will correlate well, because
a path that correlates well is the only thing it was ever asked for. So the
constraints are not a refinement of the method here -- they *are* the method, and
the search is built around them rather than having them checked afterwards.

Three constraints, in order of importance:

**Strain limit.** The warp's local time gradient may differ from 1 by at most
``r``. This is the constraint that makes the result a geophysical statement:
``r`` converts directly into the velocity change the warp claims against the
sonic, and it is derived from the guardrail's velocity limit via
:func:`~swt.tie.guardrail.strain_limit_for` rather than being set independently.
The two numbers are one number.

It is enforced structurally rather than as a penalty. The shift is defined on
control points spaced ``b = ceil(1/r)`` samples apart and may change by at most
one sample between adjacent control points, so the warp is piecewise linear with
slope in ``{-r, 0, +r}`` by construction. There is no path through the search
space that violates the limit, so none has to be rejected later.

**Monotonicity.** Implied by the strain limit for any ``r < 1``: the warped time
gradient is at least ``1 - r > 0``. Time cannot invert.

**Curvature penalty.** Changing slope costs something, so the solver prefers long
straight runs to a zig-zag that chases noise. Tracked as a third state in the
dynamic program (the direction of the last move) rather than applied afterwards
by smoothing, because smoothing a warp can push it back outside the strain limit.

The alignment itself runs coarse to fine: first on the analytic-signal envelope,
which is phase-blind and so cannot be trapped by a wrong polarity, then on the
traces themselves within a narrow band around the coarse answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import hilbert

from ..timedepth.model import TimeDepth
from ..trace import Trace

#: Cost added when the warp changes slope at a control point, in units of the
#: mean alignment error. Applied per slope change, not per sample.
DEFAULT_CURVATURE_PENALTY = 0.5


@dataclass(frozen=True)
class Warp:
    """A time-varying shift from synthetic time onto seismic time."""

    twt: np.ndarray
    shift_s: np.ndarray
    strain_limit: float
    control_spacing_s: float
    cost: float
    stage: str = "trace"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.twt.shape != self.shift_s.shape:
            raise ValueError("warp time axis and shift differ in shape")
        if self.twt.size < 2:
            raise ValueError("a warp needs at least two samples")

    @property
    def max_shift_ms(self) -> float:
        return float(np.max(np.abs(self.shift_s))) * 1e3

    def strain(self) -> np.ndarray:
        """Local ``d(shift)/d(time)`` -- the fractional stretch at each sample."""
        return np.gradient(self.shift_s, self.twt)

    def max_strain(self) -> float:
        return float(np.max(np.abs(self.strain())))

    def saturated_fraction(self, tolerance: float = 0.95) -> float:
        """Fraction of the warp pinned at its strain limit.

        The single most important diagnostic a warp carries. A warp that spends
        most of its length at the maximum permitted strain is not the answer to
        the problem -- it is the answer *clipped* by the constraint, which means
        the correction actually required is larger than the constraint allows.

        The result still looks respectable: it is smooth, it is monotonic, its
        implied velocity change passes the guardrail exactly because it was
        clipped to pass, and it improves the correlation. Every check says yes
        and the tie is still wrong. So saturation has to be tested separately,
        and a saturated warp treated as a report about the *inputs* -- a bad
        replacement velocity, missing checkshots, the wrong trace -- rather than
        as a tie.
        """
        return float(np.mean(np.abs(self.strain()) >= tolerance * self.strain_limit))

    def at_shift_limit(self, max_shift_s: float, tolerance: float = 0.95) -> bool:
        """Whether the warp reaches the edge of the shift range it was given."""
        return self.max_shift_ms / 1e3 >= tolerance * max_shift_s

    def shift_at(self, twt) -> np.ndarray:
        """The shift at arbitrary times, held constant outside the warp's range."""
        return np.interp(np.asarray(twt, float), self.twt, self.shift_s)

    def summary(self) -> dict:
        return {
            "stage": self.stage,
            "max_shift_ms": round(self.max_shift_ms, 2),
            "mean_shift_ms": round(float(np.mean(self.shift_s)) * 1e3, 2),
            "max_strain_percent": round(self.max_strain() * 100.0, 2),
            "strain_limit_percent": round(self.strain_limit * 100.0, 2),
            "saturated_fraction": round(self.saturated_fraction(), 3),
            "control_spacing_ms": round(self.control_spacing_s * 1e3, 1),
            "cost": round(self.cost, 5),
        }


def dynamic_warp(
    seismic: Trace,
    synthetic: Trace,
    strain_limit: float = 0.13,
    max_shift_s: float = 0.060,
    curvature_penalty: float = DEFAULT_CURVATURE_PENALTY,
    error_smoothing_s: float = 0.020,
    on_envelope: bool = False,
    anchors: dict | None = None,
    shift_centre: np.ndarray | None = None,
    band_s: float | None = None,
) -> Warp:
    """Find the constrained shift field aligning ``synthetic`` onto ``seismic``.

    Parameters
    ----------
    strain_limit:
        Maximum ``|d(shift)/d(time)|``. Derive it from the guardrail's velocity
        limit with :func:`~swt.tie.guardrail.strain_limit_for`; do not set it
        independently, or the solver and the guardrail will disagree about what
        is admissible.
    max_shift_s:
        Half-width of the shift search.
    curvature_penalty:
        Cost of a slope change, in units of the mean alignment error. Zero
        allows the warp to zig-zag freely within the strain limit.
    error_smoothing_s:
        Length of a moving average applied to the alignment errors before the
        search. Smoothing here is what stops the warp locking onto individual
        noisy samples; roughly one wavelet length is a sensible choice.
    on_envelope:
        Match analytic-signal envelopes rather than the traces. Phase-blind, so
        it survives a wrong polarity -- use it for the coarse pass.
    anchors:
        ``{time_s: shift_s}`` hard constraints, e.g. from formation tops.
    shift_centre, band_s:
        Restrict the search to ``band_s`` either side of a shift field from a
        previous pass. This is how the fine pass is kept near the coarse answer.

    Returns
    -------
    Warp
        Guaranteed to satisfy the strain limit by construction.
    """
    if not 0.0 < strain_limit < 1.0:
        raise ValueError("strain_limit must lie strictly between 0 and 1")
    if max_shift_s <= 0:
        raise ValueError("max_shift_s must be positive")
    if curvature_penalty < 0:
        raise ValueError("curvature_penalty must be non-negative")

    observed = seismic
    predicted = synthetic.resample_to(seismic)
    dt = observed.dt

    a = _prepare(predicted.amplitude, on_envelope)
    b = _prepare(observed.amplitude, on_envelope)

    max_lag = int(round(max_shift_s / dt))
    if max_lag < 1:
        raise ValueError("max_shift_s is shorter than one sample")
    lags = np.arange(-max_lag, max_lag + 1)

    errors = _alignment_errors(a, b, lags)
    if error_smoothing_s > 0:
        errors = _smooth_rows(errors, max(int(round(error_smoothing_s / dt)), 1))

    if shift_centre is not None and band_s is not None:
        errors = _restrict_to_band(errors, lags, shift_centre, band_s, dt, observed.twt)

    if anchors:
        errors = _apply_anchors(errors, lags, anchors, observed.twt, dt)

    spacing = int(np.ceil(1.0 / strain_limit))
    shift_samples, cost = _solve(errors, lags, spacing, curvature_penalty)

    return Warp(
        twt=observed.twt,
        shift_s=shift_samples * dt,
        strain_limit=strain_limit,
        control_spacing_s=spacing * dt,
        cost=cost,
        stage="envelope" if on_envelope else "trace",
        meta={"max_lag_samples": int(max_lag), "n_control_points": int(np.ceil(a.size / spacing))},
    )


def auto_warp(
    seismic: Trace,
    synthetic: Trace,
    strain_limit: float = 0.13,
    max_shift_s: float = 0.060,
    curvature_penalty: float = DEFAULT_CURVATURE_PENALTY,
    fine_band_s: float = 0.016,
    anchors: dict | None = None,
) -> tuple[Warp, Warp]:
    """Coarse-to-fine warping: envelope first, then trace.

    The envelope pass is phase-blind, so it cannot be captured by a polarity or
    phase error the way a trace-domain search can -- but its peak is broad, so
    it locates the warp without pinning it. The trace pass then refines within
    ``fine_band_s`` of that answer, where it is sharp and no longer at risk of
    finding the wrong half-cycle.

    Returns both warps: the coarse one is worth keeping, because a large
    disagreement between the two is itself a diagnostic.
    """
    coarse = dynamic_warp(
        seismic, synthetic,
        strain_limit=strain_limit, max_shift_s=max_shift_s,
        curvature_penalty=curvature_penalty, on_envelope=True, anchors=anchors,
    )
    fine = dynamic_warp(
        seismic, synthetic,
        strain_limit=strain_limit, max_shift_s=max_shift_s,
        curvature_penalty=curvature_penalty, on_envelope=False, anchors=anchors,
        shift_centre=coarse.shift_s, band_s=fine_band_s,
    )
    return coarse, fine


def apply_warp(time_depth: TimeDepth, warp: Warp) -> TimeDepth:
    """Fold a warp into the time-depth model.

    Applied to the T-D, never to the displayed trace: a warp is a change to the
    well's calibration, and it must survive into the horizon picks and the depth
    conversion built from it. Warping only the picture produces a tie that looks
    right and exports wrong.
    """
    shifted = time_depth.twt + warp.shift_at(time_depth.twt)
    return TimeDepth(
        depth_tvdss=time_depth.depth_tvdss,
        twt=shifted,
        provenance=f"{time_depth.provenance} + warp (max {warp.max_shift_ms:+.1f} ms)",
        meta={**time_depth.meta, "warp": warp.summary()},
    )


# ---------------------------------------------------------------------------
# internals


def _prepare(x: np.ndarray, on_envelope: bool) -> np.ndarray:
    """Unit-RMS amplitudes, or their envelope, ready for differencing."""
    values = np.asarray(x, dtype=float)
    if on_envelope:
        values = np.abs(hilbert(values))
    values = values - np.mean(values)
    rms = float(np.sqrt(np.mean(values**2)))
    return values / rms if rms > 0 else values


def _alignment_errors(a: np.ndarray, b: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """``errors[i, j]`` is the mismatch putting sample ``i`` of ``a`` at lag ``j``.

    Where the shifted sample falls outside ``b`` there is nothing to compare, and
    the score given to that impossible comparison decides what the warp does at
    the ends of the trace. Scoring it *badly* -- the obvious choice -- is wrong:
    large lags become impossible near the ends, so the solver drags the warp back
    towards zero shift there, inventing a taper that the data never asked for and
    that the strain limit then spreads inward. Scoring it as **neutral** (the mean
    error over comparisons that could be made) instead leaves the ends
    uninformative, and the strain limit and curvature penalty carry the warp
    straight through, which is the honest extrapolation.
    """
    n = a.size
    errors = np.empty((n, lags.size), dtype=float)
    index = np.arange(n)

    for j, lag in enumerate(lags):
        target = index + lag
        inside = (target >= 0) & (target < b.size)
        difference = np.empty(n, dtype=float)
        difference[inside] = a[inside] - b[target[inside]]
        difference[~inside] = np.nan
        errors[:, j] = difference**2

    neutral = float(np.nanmean(errors)) if np.any(np.isfinite(errors)) else 1.0
    return np.where(np.isfinite(errors), errors, neutral)


def _smooth_rows(errors: np.ndarray, window: int) -> np.ndarray:
    """Moving average of the error matrix along time."""
    if window < 2:
        return errors
    kernel = np.ones(window) / window
    smoothed = np.empty_like(errors)
    for j in range(errors.shape[1]):
        smoothed[:, j] = np.convolve(errors[:, j], kernel, mode="same")
    return smoothed


def _restrict_to_band(
    errors: np.ndarray, lags: np.ndarray, centre: np.ndarray,
    band_s: float, dt: float, twt: np.ndarray,
) -> np.ndarray:
    """Penalise lags far from a previous pass's answer."""
    centre_samples = np.interp(twt, twt[: centre.size], centre[: twt.size] / dt) \
        if centre.size != twt.size else centre / dt
    band = max(band_s / dt, 1.0)
    distance = np.abs(lags[None, :] - centre_samples[:, None])
    outside = distance > band
    penalty = float(np.max(errors)) * 4.0
    return np.where(outside, errors + penalty, errors)


def _apply_anchors(
    errors: np.ndarray, lags: np.ndarray, anchors: dict,
    twt: np.ndarray, dt: float,
) -> np.ndarray:
    """Force the warp through known points by penalising every other lag there."""
    out = errors.copy()
    penalty = float(np.max(errors)) * 100.0
    for time_s, shift_s in anchors.items():
        i = int(np.argmin(np.abs(twt - float(time_s))))
        wanted = int(round(float(shift_s) / dt))
        mask = lags != wanted
        out[i, mask] += penalty
    return out


def _solve(
    errors: np.ndarray, lags: np.ndarray, spacing: int, curvature_penalty: float
) -> tuple[np.ndarray, float]:
    """Dynamic program over control points, with slope-change penalised.

    State is ``(lag, last move)``. Carrying the last move is what lets a slope
    *change* be charged for while a continuing slope is not -- the difference
    between a curvature penalty and a slope penalty, and only the former leaves
    a genuine ramp unpunished.
    """
    n, n_lags = errors.shape
    controls = list(range(0, n, spacing))
    if controls[-1] != n - 1:
        # *Move* the last control point to the end rather than appending one.
        # Appending would leave a final segment shorter than `spacing`, and a
        # one-sample lag change across a short segment exceeds the strain limit
        # -- the constraint would hold everywhere except the last few samples,
        # which is exactly where nobody looks. Moving it makes that segment
        # longer than `spacing`, which only tightens the strain.
        controls[-1] = n - 1
    n_controls = len(controls)

    segment_cost = _segment_costs(errors, controls)
    penalty = curvature_penalty * float(np.mean(errors))

    moves = (-1, 0, 1)
    big = np.inf

    # cost[m, u] -- best cost reaching control point k at lag u having last moved m
    cost = np.full((3, n_lags), big)
    cost[1, :] = errors[controls[0], :]  # first point: no previous move
    back = np.zeros((n_controls, 3, n_lags), dtype=np.int8)

    for k in range(1, n_controls):
        new_cost = np.full((3, n_lags), big)
        new_back = np.zeros((3, n_lags), dtype=np.int8)

        for mi, move in enumerate(moves):
            # Arriving at lag u having moved `move` means departing from u - move.
            source = np.full(n_lags, big)
            if move == -1:
                source[:-1] = np.minimum.reduce([
                    cost[pi, 1:] + (0.0 if moves[pi] == move else penalty)
                    for pi in range(3)
                ])
                previous = np.argmin(np.stack([
                    cost[pi, 1:] + (0.0 if moves[pi] == move else penalty)
                    for pi in range(3)
                ]), axis=0)
                new_back[mi, :-1] = previous
            elif move == 1:
                source[1:] = np.minimum.reduce([
                    cost[pi, :-1] + (0.0 if moves[pi] == move else penalty)
                    for pi in range(3)
                ])
                previous = np.argmin(np.stack([
                    cost[pi, :-1] + (0.0 if moves[pi] == move else penalty)
                    for pi in range(3)
                ]), axis=0)
                new_back[mi, 1:] = previous
            else:
                stacked = np.stack([
                    cost[pi, :] + (0.0 if moves[pi] == move else penalty)
                    for pi in range(3)
                ])
                source = np.min(stacked, axis=0)
                new_back[mi, :] = np.argmin(stacked, axis=0)

            new_cost[mi] = source + segment_cost[k][mi]

        cost, back[k] = new_cost, new_back

    # Backtrack from the cheapest end state.
    end_move, end_lag = np.unravel_index(int(np.argmin(cost)), cost.shape)
    total = float(cost[end_move, end_lag])

    control_lags = np.zeros(n_controls, dtype=int)
    move_index, lag_index = int(end_move), int(end_lag)
    for k in range(n_controls - 1, -1, -1):
        control_lags[k] = lags[lag_index]
        if k == 0:
            break
        previous_move = int(back[k, move_index, lag_index])
        lag_index -= moves[move_index]
        move_index = previous_move

    shift = np.interp(np.arange(n), np.array(controls, dtype=float), control_lags.astype(float))
    return shift, total


def _segment_costs(errors: np.ndarray, controls: list[int]) -> list[np.ndarray]:
    """Error accumulated over each segment, per arriving lag and per move.

    Charged along the *interpolated* path rather than at the control point only.
    Sampling the error at control points alone would let the warp pass through
    high-error regions between them without paying, which is how a warp acquires
    a plausible shape and an implausible fit.
    """
    n_lags = errors.shape[1]
    moves = (-1, 0, 1)
    out: list[np.ndarray] = [np.zeros((3, n_lags))]

    for k in range(1, len(controls)):
        start, stop = controls[k - 1], controls[k]
        span = stop - start
        costs = np.zeros((3, n_lags))
        for mi, move in enumerate(moves):
            accumulated = np.zeros(n_lags)
            for step in range(1, span + 1):
                # Lag at this sample, linearly interpolated towards the arrival lag.
                offset = move * (1.0 - step / span)
                index = np.arange(n_lags) - int(round(offset))
                valid = (index >= 0) & (index < n_lags)
                sample = np.full(n_lags, np.max(errors))
                sample[valid] = errors[start + step, index[valid]]
                accumulated += sample
            costs[mi] = accumulated
        out.append(costs)
    return out
