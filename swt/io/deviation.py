"""Deviation surveys: measured depth to true vertical depth and position.

Everything in a well tie happens in TVDSS.  A log is recorded against measured
depth along the borehole, and in a deviated well those two differ by hundreds of
metres -- so integrating a sonic against MD stretches the time-depth curve by
exactly the amount the well is deviated, and no calibration downstream will
recognise the error for what it is.

Minimum curvature is the industry-standard interpolation between survey
stations: it assumes the borehole follows a circular arc rather than a straight
line, which is both closer to reality and what every other package uses -- and
agreeing with the other packages matters more here than being marginally more
clever.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import depth_to_m


@dataclass(frozen=True)
class Deviation:
    """A well path: measured depth, TVD, and horizontal position."""

    md_m: np.ndarray
    tvd_m: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    kb_elevation_m: float = 0.0
    name: str = "deviation"
    #: Surface location in the seismic survey's coordinate system.  The east and
    #: north arrays are *offsets from the wellhead*; adding these gives the
    #: absolute position needed to find a trace.
    wellhead_x: float = 0.0
    wellhead_y: float = 0.0

    def __post_init__(self) -> None:
        arrays = (self.md_m, self.tvd_m, self.east_m, self.north_m)
        if len({a.size for a in arrays}) != 1:
            raise ValueError("deviation arrays differ in length")
        if self.md_m.size < 2:
            raise ValueError("a deviation survey needs at least two stations")
        if np.any(np.diff(self.md_m) <= 0):
            raise ValueError("measured depth must be strictly increasing")

    def tvdss_at_md(self, md) -> np.ndarray:
        """TVD sub-sea (m) at the given measured depths.

        TVDSS is TVD below the seismic datum: positive downward, with the
        Kelly bushing elevation removed.
        """
        tvd = np.interp(np.asarray(md, dtype=float), self.md_m, self.tvd_m)
        return tvd - self.kb_elevation_m

    def position_at_md(self, md) -> tuple[np.ndarray, np.ndarray]:
        """Easting and northing offsets (m) from the wellhead."""
        md = np.asarray(md, dtype=float)
        return (
            np.interp(md, self.md_m, self.east_m),
            np.interp(md, self.md_m, self.north_m),
        )

    def absolute_position_at_md(self, md) -> tuple[np.ndarray, np.ndarray]:
        """Absolute survey coordinates at the given measured depths."""
        east, north = self.position_at_md(md)
        return east + self.wellhead_x, north + self.wellhead_y

    def md_at_tvdss(self, tvdss) -> np.ndarray:
        """Measured depth at a given TVD sub-sea -- the inverse of the survey.

        Needed to answer the question a deviated tie actually asks: *where is the
        well when it is this deep?* The log is indexed by MD and the seismic by
        TVDSS, so going from one to the other is what puts the borehole in the
        right place laterally.

        The inverse only exists where TVD increases with MD. In a horizontal or
        dropping section it does not -- the well passes through the same TVD
        twice, and no single MD answers the question -- so those sections are
        excluded and the interpolation uses the monotone part. A well that never
        increases in TVD at all raises rather than returning nonsense.
        """
        tvdss = np.asarray(tvdss, dtype=float)
        tvd = self.tvd_m - self.kb_elevation_m

        # Keep only the running-maximum samples, which are exactly the ones
        # where the well is still going down.
        keep = np.concatenate([[True], np.diff(np.maximum.accumulate(tvd)) > 0])
        if np.count_nonzero(keep) < 2:
            raise ValueError(
                f"{self.name}: TVD never increases with measured depth, so depth "
                "cannot be inverted to MD. A tie needs a well that goes down."
            )
        return np.interp(tvdss, tvd[keep], self.md_m[keep])

    def position_at_tvdss(self, tvdss) -> tuple[np.ndarray, np.ndarray]:
        """Absolute survey coordinates at a given TVD sub-sea."""
        return self.absolute_position_at_md(self.md_at_tvdss(tvdss))

    @property
    def is_vertical(self) -> bool:
        """True when MD and TVD never differ by more than a metre."""
        return bool(np.max(np.abs(self.md_m - self.tvd_m)) < 1.0)

    def max_departure_m(self) -> float:
        """Greatest horizontal distance from the wellhead."""
        return float(np.max(np.hypot(self.east_m, self.north_m)))

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_stations": int(self.md_m.size),
            "md_range_m": [round(float(self.md_m[0]), 1), round(float(self.md_m[-1]), 1)],
            "tvd_range_m": [round(float(self.tvd_m[0]), 1), round(float(self.tvd_m[-1]), 1)],
            "kb_elevation_m": self.kb_elevation_m,
            "max_departure_m": round(self.max_departure_m(), 1),
            "is_vertical": self.is_vertical,
            "wellhead": [self.wellhead_x, self.wellhead_y],
        }


def vertical(
    md_m: np.ndarray,
    kb_elevation_m: float = 0.0,
    name: str = "vertical",
    wellhead_x: float = 0.0,
    wellhead_y: float = 0.0,
) -> Deviation:
    """A perfectly vertical well -- TVD equals MD, no departure."""
    md = np.asarray(md_m, dtype=float)
    zeros = np.zeros_like(md)
    return Deviation(md, md.copy(), zeros, zeros.copy(), kb_elevation_m, name,
                     wellhead_x, wellhead_y)


def minimum_curvature(
    md: np.ndarray,
    inclination_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    kb_elevation_m: float = 0.0,
    depth_unit: str = "m",
    name: str = "deviation",
    wellhead_x: float = 0.0,
    wellhead_y: float = 0.0,
) -> Deviation:
    """Compute a well path from an MD / inclination / azimuth survey.

    Between two stations the borehole is taken to follow a circular arc.  The
    ratio factor ``RF = (2 / dogleg) * tan(dogleg / 2)`` weights the two
    stations' direction vectors; it tends to 1 as the dogleg tends to zero, and
    that limit is handled explicitly rather than left to divide by zero on the
    straight sections that make up most of a survey.
    """
    md_m = depth_to_m(np.asarray(md, dtype=float), depth_unit)
    inc = np.deg2rad(np.asarray(inclination_deg, dtype=float))
    azi = np.deg2rad(np.asarray(azimuth_deg, dtype=float))

    if not (md_m.shape == inc.shape == azi.shape):
        raise ValueError("md, inclination and azimuth must have the same shape")
    if md_m.size < 2:
        raise ValueError("need at least two survey stations")
    if np.any(np.diff(md_m) <= 0):
        raise ValueError("measured depth must be strictly increasing")

    d_md = np.diff(md_m)
    inc1, inc2 = inc[:-1], inc[1:]
    azi1, azi2 = azi[:-1], azi[1:]

    cos_dogleg = np.clip(
        np.cos(inc2 - inc1) - np.sin(inc1) * np.sin(inc2) * (1.0 - np.cos(azi2 - azi1)),
        -1.0,
        1.0,
    )
    dogleg = np.arccos(cos_dogleg)

    # RF -> 1 as the dogleg -> 0; compute it only where the angle is resolvable.
    ratio = np.ones_like(dogleg)
    bending = dogleg > 1e-8
    ratio[bending] = (2.0 / dogleg[bending]) * np.tan(dogleg[bending] / 2.0)

    half = 0.5 * d_md * ratio
    d_tvd = half * (np.cos(inc1) + np.cos(inc2))
    d_east = half * (np.sin(inc1) * np.sin(azi1) + np.sin(inc2) * np.sin(azi2))
    d_north = half * (np.sin(inc1) * np.cos(azi1) + np.sin(inc2) * np.cos(azi2))

    def accumulate(start: float, increments: np.ndarray) -> np.ndarray:
        return np.concatenate([[start], start + np.cumsum(increments)])

    return Deviation(
        md_m=md_m,
        tvd_m=accumulate(float(md_m[0]), d_tvd),
        east_m=accumulate(0.0, d_east),
        north_m=accumulate(0.0, d_north),
        kb_elevation_m=float(kb_elevation_m),
        name=name,
        wellhead_x=float(wellhead_x),
        wellhead_y=float(wellhead_y),
    )


def load_csv(
    path: str,
    md_column: int = 0,
    inclination_column: int = 1,
    azimuth_column: int = 2,
    depth_unit: str = "m",
    kb_elevation_m: float = 0.0,
    wellhead_x: float = 0.0,
    wellhead_y: float = 0.0,
    delimiter: str | None = None,
    skip_header: int = 0,
    name: str | None = None,
) -> Deviation:
    """Read an MD / inclination / azimuth survey and compute the well path.

    The near-universal export format. Inclination and azimuth are in degrees;
    the depth unit is stated rather than guessed, like everywhere else in SWT.
    Check the result with :meth:`Deviation.summary` before using it -- a survey
    in feet read as metres puts the well in the wrong place by a factor of three.
    """
    table = np.genfromtxt(
        path, delimiter=delimiter, skip_header=skip_header, dtype=float, comments="#"
    )
    if table.ndim == 1:
        table = table.reshape(1, -1)
    needed = max(md_column, inclination_column, azimuth_column)
    if table.shape[1] <= needed:
        raise ValueError(
            f"{path} has {table.shape[1]} column(s); need at least {needed + 1}"
        )

    valid = np.all(np.isfinite(table[:, [md_column, inclination_column, azimuth_column]]),
                   axis=1)
    import os

    return minimum_curvature(
        table[valid, md_column],
        table[valid, inclination_column],
        table[valid, azimuth_column],
        kb_elevation_m=kb_elevation_m,
        depth_unit=depth_unit,
        name=name or os.path.splitext(os.path.basename(path))[0],
        wellhead_x=wellhead_x,
        wellhead_y=wellhead_y,
    )
