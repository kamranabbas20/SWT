"""The time-depth model and its one hard invariant.

A time-depth (T-D) relationship maps true vertical depth sub-sea (TVDSS, m) to
two-way travel time (TWT, s).  It must be **strictly monotonically increasing**:
deeper is always later.  A non-monotonic T-D is not merely inaccurate, it is
meaningless -- it maps two depths to the same time, silently corrupts the
depth-to-time resampling of the reflectivity series, and produces a synthetic
that looks plausible and is wrong.

Every operation that produces or edits a T-D returns a :class:`TimeDepth`, and
the constructor enforces the invariant.  There is no way to build an invalid one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class MonotonicityError(ValueError):
    """Raised when a proposed time-depth relationship inverts time with depth."""


@dataclass(frozen=True)
class TimeDepth:
    """A strictly increasing map from TVDSS (m) to two-way time (s).

    Parameters
    ----------
    depth_tvdss:
        Depths in metres below the seismic reference datum, strictly increasing.
    twt:
        Two-way travel times in seconds, strictly increasing.
    provenance:
        Free-form record of how this curve was built (``"sonic integration"``,
        ``"checkshot-calibrated"``, ``"warped by auto-tie"``).  Carried through
        the whole pipeline so a report can state where the numbers came from.
    """

    depth_tvdss: np.ndarray
    twt: np.ndarray
    provenance: str = "unspecified"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_tvdss, dtype=float)
        twt = np.asarray(self.twt, dtype=float)

        if depth.ndim != 1 or twt.ndim != 1:
            raise ValueError("depth and twt must be 1-D")
        if depth.size != twt.size:
            raise ValueError(f"depth ({depth.size}) and twt ({twt.size}) differ in length")
        if depth.size < 2:
            raise ValueError("a time-depth model needs at least two points")
        if not np.all(np.isfinite(depth)) or not np.all(np.isfinite(twt)):
            raise ValueError("time-depth models may not contain NaN or inf")

        _assert_strictly_increasing(depth, "depth")
        _assert_strictly_increasing(twt, "twt")

        object.__setattr__(self, "depth_tvdss", depth)
        object.__setattr__(self, "twt", twt)

    # -- interpolation ----------------------------------------------------

    def time_at(self, depth_tvdss) -> np.ndarray:
        """TWT (s) at the given TVDSS (m), linearly interpolated.

        Depths outside the model's range are extrapolated at the end interval's
        velocity rather than clamped -- clamping would create a zero-velocity
        layer, which is worse than an honest extrapolation.
        """
        return _interp_extrapolate(np.asarray(depth_tvdss, float), self.depth_tvdss, self.twt)

    def depth_at(self, twt) -> np.ndarray:
        """TVDSS (m) at the given TWT (s).  Well defined because T is monotonic."""
        return _interp_extrapolate(np.asarray(twt, float), self.twt, self.depth_tvdss)

    # -- derived quantities ------------------------------------------------

    def interval_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Interval velocity (m/s) between successive knots.

        Returns the midpoint depth of each interval and its velocity, computed
        as ``2 * dz / dt`` -- the factor of two because TWT is two-way.
        """
        dz = np.diff(self.depth_tvdss)
        dt = np.diff(self.twt)
        midpoints = 0.5 * (self.depth_tvdss[:-1] + self.depth_tvdss[1:])
        return midpoints, 2.0 * dz / dt

    def average_velocity(self) -> np.ndarray:
        """Average velocity (m/s) from the top of the model to each knot."""
        dz = self.depth_tvdss - self.depth_tvdss[0]
        dt = self.twt - self.twt[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            vavg = np.where(dt > 0, 2.0 * dz / np.where(dt > 0, dt, 1.0), np.nan)
        return vavg

    def resampled(self, depth_tvdss, provenance: str | None = None) -> "TimeDepth":
        """Return this model evaluated on a new depth axis."""
        depth = np.asarray(depth_tvdss, float)
        return TimeDepth(
            depth_tvdss=depth,
            twt=self.time_at(depth),
            provenance=provenance or self.provenance,
            meta=dict(self.meta),
        )

    def shifted(self, dt_s: float, provenance: str | None = None) -> "TimeDepth":
        """Return this model with a constant bulk time shift applied."""
        return TimeDepth(
            depth_tvdss=self.depth_tvdss,
            twt=self.twt + float(dt_s),
            provenance=provenance or f"{self.provenance} + bulk shift {dt_s * 1e3:.1f} ms",
            meta=dict(self.meta),
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"TimeDepth({self.depth_tvdss[0]:.1f}-{self.depth_tvdss[-1]:.1f} m, "
            f"{self.twt[0] * 1e3:.1f}-{self.twt[-1] * 1e3:.1f} ms, "
            f"n={self.depth_tvdss.size}, provenance={self.provenance!r})"
        )


def _assert_strictly_increasing(values: np.ndarray, name: str) -> None:
    diffs = np.diff(values)
    bad = np.flatnonzero(diffs <= 0)
    if bad.size:
        first = int(bad[0])
        raise MonotonicityError(
            f"{name} is not strictly increasing: "
            f"{name}[{first}]={values[first]:.6g} >= {name}[{first + 1}]={values[first + 1]:.6g} "
            f"({bad.size} violation(s) in total)"
        )


def _interp_extrapolate(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linear interpolation that extrapolates on the end gradients."""
    out = np.interp(x, xp, fp)

    below = x < xp[0]
    if np.any(below):
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        out[below] = fp[0] + (x[below] - xp[0]) * slope

    above = x > xp[-1]
    if np.any(above):
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        out[above] = fp[-1] + (x[above] - xp[-1]) * slope

    return out


def is_monotonic(twt) -> bool:
    """Cheap predicate for callers that want to test before constructing."""
    twt = np.asarray(twt, float)
    return bool(np.all(np.isfinite(twt))) and bool(np.all(np.diff(twt) > 0))
