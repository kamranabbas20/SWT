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
        }


def vertical(md_m: np.ndarray, kb_elevation_m: float = 0.0, name: str = "vertical") -> Deviation:
    """A perfectly vertical well -- TVD equals MD, no departure."""
    md = np.asarray(md_m, dtype=float)
    zeros = np.zeros_like(md)
    return Deviation(md, md.copy(), zeros, zeros.copy(), kb_elevation_m, name)


def minimum_curvature(
    md: np.ndarray,
    inclination_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    kb_elevation_m: float = 0.0,
    depth_unit: str = "m",
    name: str = "deviation",
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
    )
