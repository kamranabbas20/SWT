"""Checkshot / VSP time-depth survey loading.

Small file, disproportionate importance: the checkshot is the only direct
measurement of time versus depth in the whole problem, and everything else is
calibrated to it.

Two conventions cause most of the trouble, and both are handled explicitly
rather than inferred:

* **one-way versus two-way time** -- a factor of two, which shows up as a tie
  that is out by half, and is embarrassingly easy to miss on a deep well;
* **measured versus true vertical depth** -- identical in a vertical well and
  badly different in a deviated one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import depth_to_m, time_to_s


@dataclass(frozen=True)
class Checkshots:
    """A time-depth survey in canonical units: metres TVDSS and seconds TWT."""

    depth_tvdss_m: np.ndarray
    twt_s: np.ndarray
    name: str = "checkshots"

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_tvdss_m, dtype=float)
        twt = np.asarray(self.twt_s, dtype=float)
        if depth.shape != twt.shape:
            raise ValueError("checkshot depth and time arrays differ in shape")
        if depth.size < 2:
            raise ValueError("need at least two checkshots")
        if not (np.all(np.isfinite(depth)) and np.all(np.isfinite(twt))):
            raise ValueError("checkshots contain non-finite values")
        order = np.argsort(depth)
        object.__setattr__(self, "depth_tvdss_m", depth[order])
        object.__setattr__(self, "twt_s", twt[order])

    def interval_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Interval velocity between successive checkshots, for QC.

        Wildly varying or negative interval velocities here mean a bad pick, and
        it is far cheaper to find that now than after it has been baked into a
        drift curve.
        """
        dz = np.diff(self.depth_tvdss_m)
        dt = np.diff(self.twt_s)
        mid = 0.5 * (self.depth_tvdss_m[:-1] + self.depth_tvdss_m[1:])
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity = np.where(dt > 0, 2.0 * dz / np.where(dt > 0, dt, np.nan), np.nan)
        return mid, velocity

    def implausible_intervals(self, low: float = 1000.0, high: float = 7000.0) -> list[dict]:
        """Checkshot intervals whose implied velocity is outside a sane range."""
        mid, velocity = self.interval_velocity()
        bad = ~np.isfinite(velocity) | (velocity < low) | (velocity > high)
        return [
            {
                "depth_m": round(float(mid[i]), 1),
                "interval_velocity_m_s": (
                    None if not np.isfinite(velocity[i]) else round(float(velocity[i]), 1)
                ),
            }
            for i in np.flatnonzero(bad)
        ]

    def summary(self) -> dict:
        mid, velocity = self.interval_velocity()
        finite = velocity[np.isfinite(velocity)]
        return {
            "name": self.name,
            "n": int(self.depth_tvdss_m.size),
            "depth_range_m": [round(float(self.depth_tvdss_m[0]), 1),
                              round(float(self.depth_tvdss_m[-1]), 1)],
            "twt_range_ms": [round(float(self.twt_s[0]) * 1e3, 1),
                             round(float(self.twt_s[-1]) * 1e3, 1)],
            "interval_velocity_range_m_s": (
                [round(float(finite.min()), 1), round(float(finite.max()), 1)]
                if finite.size else None
            ),
            "implausible_intervals": self.implausible_intervals(),
        }


def from_arrays(
    depth: np.ndarray,
    time: np.ndarray,
    depth_unit: str = "m",
    time_unit: str = "s",
    one_way: bool = False,
    name: str = "checkshots",
) -> Checkshots:
    """Build a survey from arrays, converting units and time convention.

    Parameters
    ----------
    one_way:
        Set when the survey lists one-way time, as VSP first-break picks
        usually do.  Times are doubled.  There is no auto-detection: a wrong
        guess is a factor-of-two error in the tie.
    """
    depth_m = depth_to_m(np.asarray(depth, dtype=float), depth_unit)
    time_s = time_to_s(np.asarray(time, dtype=float), time_unit)
    return Checkshots(depth_m, time_s * 2.0 if one_way else time_s, name)


def load_csv(
    path: str,
    depth_column: int = 0,
    time_column: int = 1,
    depth_unit: str = "m",
    time_unit: str = "ms",
    one_way: bool = False,
    delimiter: str | None = None,
    skip_header: int = 0,
    name: str | None = None,
) -> Checkshots:
    """Read a two-column checkshot table.

    Defaults match the most common export: depth in metres, time in
    milliseconds, two-way.  Check them against the file rather than trusting
    them -- ``Checkshots.summary()`` reports interval velocities, and those make
    a unit or convention error obvious immediately.
    """
    table = np.genfromtxt(
        path, delimiter=delimiter, skip_header=skip_header, dtype=float, comments="#"
    )
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] <= max(depth_column, time_column):
        raise ValueError(
            f"{path} has {table.shape[1]} column(s); need columns "
            f"{depth_column} and {time_column}"
        )

    valid = np.isfinite(table[:, depth_column]) & np.isfinite(table[:, time_column])
    return from_arrays(
        table[valid, depth_column],
        table[valid, time_column],
        depth_unit=depth_unit,
        time_unit=time_unit,
        one_way=one_way,
        name=name or _stem(path),
    )


def _stem(path: str) -> str:
    import os

    return os.path.splitext(os.path.basename(path))[0]
