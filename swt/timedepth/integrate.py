"""Sonic integration: turning a slowness log into a time-depth curve.

The integral itself is trivial.  What is not trivial, and what decides whether a
tie works at all, is the interval *above* the log:

    Logs commonly start several hundred metres below the seismic reference
    datum (SRD).  A synthetic built from the log alone therefore starts at an
    unknown time.  Every millisecond of error in that start time shifts the
    entire synthetic rigidly, and no amount of stretch and squeeze below will
    repair it -- it will simply be absorbed as a bulk shift that hides the real
    problem.

So :func:`integrate_sonic` refuses to invent the shallow section.  The caller
must state how the SRD-to-log-top gap is filled, via :class:`ShallowModel`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import TimeDepth


@dataclass(frozen=True)
class ShallowModel:
    """How the section between the seismic datum and the log top is filled.

    Exactly one of the two mechanisms must be given.

    Parameters
    ----------
    replacement_velocity:
        Constant velocity (m/s) for the whole SRD-to-log-top interval.  The
        usual first pass when there is no checkshot.
    twt_at_log_top:
        The two-way time (s) at the top of the log, taken from a checkshot or a
        previous tie.  Preferred when available: it is a measurement rather than
        an assumption.
    """

    replacement_velocity: float | None = None
    twt_at_log_top: float | None = None

    def __post_init__(self) -> None:
        given = [self.replacement_velocity is not None, self.twt_at_log_top is not None]
        if sum(given) != 1:
            raise ValueError(
                "specify exactly one of replacement_velocity or twt_at_log_top -- "
                "SWT will not guess the time at the top of the log"
            )
        if self.replacement_velocity is not None and self.replacement_velocity <= 0:
            raise ValueError("replacement_velocity must be positive")
        if self.twt_at_log_top is not None and self.twt_at_log_top < 0:
            raise ValueError("twt_at_log_top must be non-negative")

    def start_time(self, log_top_tvdss: float) -> float:
        """TWT (s) at the top of the log."""
        if self.twt_at_log_top is not None:
            return float(self.twt_at_log_top)
        if log_top_tvdss < 0:
            raise ValueError(
                f"log top is above the seismic datum (TVDSS {log_top_tvdss:.1f} m); "
                "a replacement velocity cannot fill a negative interval"
            )
        return 2.0 * log_top_tvdss / float(self.replacement_velocity)

    def describe(self, log_top_tvdss: float) -> str:
        if self.twt_at_log_top is not None:
            return f"TWT at log top fixed at {self.twt_at_log_top * 1e3:.1f} ms"
        return (
            f"{log_top_tvdss:.1f} m of replacement velocity "
            f"{self.replacement_velocity:.0f} m/s above the log top"
        )


def integrate_sonic(
    depth_tvdss: np.ndarray,
    dt_us_per_m: np.ndarray,
    shallow: ShallowModel,
) -> TimeDepth:
    """Integrate a slowness log into a two-way time-depth curve.

    ``TWT(z) = TWT(z_top) + 2 * integral of slowness dz``

    With slowness in us/m and depth in m, each interval contributes
    ``2e-6 * DT * dz`` seconds.  Slowness is averaged over each interval
    (trapezoidal), which matters on coarsely sampled logs.

    Parameters
    ----------
    depth_tvdss:
        Strictly increasing TVDSS in metres.
    dt_us_per_m:
        Slowness in us/m, same length, finite everywhere.  Condition and
        gap-fill the log *before* calling this -- integration through a NaN
        poisons every sample below it.
    shallow:
        How to fill the datum-to-log-top interval.

    Returns
    -------
    TimeDepth
        Monotonicity is guaranteed by construction: positive slowness gives
        positive time increments, and a non-positive slowness raises here.
    """
    depth = np.asarray(depth_tvdss, dtype=float)
    dt = np.asarray(dt_us_per_m, dtype=float)

    if depth.shape != dt.shape:
        raise ValueError(f"depth {depth.shape} and slowness {dt.shape} differ in shape")
    if depth.size < 2:
        raise ValueError("need at least two samples to integrate")
    if not np.all(np.isfinite(dt)):
        n_bad = int(np.count_nonzero(~np.isfinite(dt)))
        raise ValueError(
            f"slowness contains {n_bad} non-finite sample(s); condition and gap-fill "
            "the log before integrating (a NaN corrupts every sample below it)"
        )
    if np.any(dt <= 0):
        raise ValueError("slowness must be strictly positive")
    if np.any(np.diff(depth) <= 0):
        raise ValueError("depth must be strictly increasing")

    dz = np.diff(depth)
    dt_mean = 0.5 * (dt[:-1] + dt[1:])
    increments = 2.0e-6 * dt_mean * dz

    twt = np.empty_like(depth)
    twt[0] = shallow.start_time(float(depth[0]))
    twt[1:] = twt[0] + np.cumsum(increments)

    return TimeDepth(
        depth_tvdss=depth,
        twt=twt,
        provenance="sonic integration",
        meta={"shallow_model": shallow.describe(float(depth[0]))},
    )
