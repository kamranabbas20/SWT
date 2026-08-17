"""LAS loading with unit detection and curve aliasing.

Two things this layer refuses to do:

* **Guess units.**  If a curve has no unit, or one that is not recognised, the
  load fails.  A sonic silently treated as us/m when it is us/ft produces a
  time-depth curve wrong by a factor of 3.28 and a synthetic that still looks
  like a synthetic.
* **Guess which curve is which** beyond a documented alias table.  The aliases
  below cover the common mnemonics; anything else must be named explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..units import (
    UnitError,
    density_to_g_cm3,
    depth_to_m,
    looks_like_us_per_ft,
    slowness_to_us_per_m,
)

#: Common mnemonics, in preference order.  Extend rather than guess at runtime.
CURVE_ALIASES: dict[str, tuple[str, ...]] = {
    "depth": ("DEPT", "DEPTH", "MD", "TDEP"),
    "sonic": ("DT", "DTCO", "DTC", "AC", "DT24", "DTP", "SONIC"),
    "density": ("RHOB", "RHOZ", "DEN", "DENS", "RHO", "ZDEN"),
    "caliper": ("CALI", "CAL", "HCAL", "CALS"),
    "gamma": ("GR", "GRD", "SGR", "GRGC"),
    "shear": ("DTS", "DTSM", "DTSH"),
}


@dataclass
class WellLogs:
    """Logs on a single measured-depth axis, in canonical units."""

    depth_md_m: np.ndarray
    sonic_us_per_m: np.ndarray | None = None
    density_g_cm3: np.ndarray | None = None
    caliper: np.ndarray | None = None
    gamma: np.ndarray | None = None
    name: str = "well"
    source_units: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.depth_md_m.size
        for field_name in ("sonic_us_per_m", "density_g_cm3", "caliper", "gamma"):
            curve = getattr(self, field_name)
            if curve is not None and curve.size != n:
                raise ValueError(f"{field_name} has {curve.size} samples, depth has {n}")

    @property
    def n(self) -> int:
        return int(self.depth_md_m.size)

    def coverage(self) -> dict:
        """Which curves are present, and how complete each one is."""
        out = {}
        for field_name in ("sonic_us_per_m", "density_g_cm3", "caliper", "gamma"):
            curve = getattr(self, field_name)
            if curve is None:
                out[field_name] = None
            else:
                finite = int(np.count_nonzero(np.isfinite(curve)))
                out[field_name] = round(100.0 * finite / self.n, 1)
        return out

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_samples": self.n,
            "depth_range_m": [round(float(self.depth_md_m[0]), 2),
                              round(float(self.depth_md_m[-1]), 2)],
            "sample_interval_m": round(float(np.median(np.diff(self.depth_md_m))), 4),
            "curve_coverage_percent": self.coverage(),
            "source_units": self.source_units,
        }


def load_las(
    path: str,
    curves: dict[str, str] | None = None,
    name: str | None = None,
) -> WellLogs:
    """Read a LAS file into canonical units.

    Parameters
    ----------
    path:
        Path to the ``.las`` file.
    curves:
        Explicit mnemonic overrides, e.g. ``{"sonic": "DT4P", "density": "RHOZ"}``.
        Anything not overridden is resolved through :data:`CURVE_ALIASES`.
    name:
        Well name.  Defaults to the LAS well header, then the filename.

    Raises
    ------
    UnitError
        If any required curve carries a missing or unrecognised unit.
    """
    import lasio

    las = lasio.read(path)
    overrides = dict(curves or {})
    available = {curve.mnemonic.upper(): curve for curve in las.curves}

    def find(role: str):
        mnemonic = overrides.get(role)
        if mnemonic is not None:
            curve = available.get(mnemonic.upper())
            if curve is None:
                raise KeyError(
                    f"curve {mnemonic!r} requested for {role!r} but not in {path}; "
                    f"available: {sorted(available)}"
                )
            return curve
        for candidate in CURVE_ALIASES[role]:
            if candidate in available:
                return available[candidate]
        return None

    depth_curve = find("depth")
    if depth_curve is None:
        raise KeyError(f"no depth curve found in {path}; tried {CURVE_ALIASES['depth']}")
    depth = depth_to_m(np.asarray(depth_curve.data, dtype=float), depth_curve.unit)

    order = np.argsort(depth)
    depth = depth[order]

    source_units = {"depth": depth_curve.unit}

    def convert(role: str, converter):
        curve = find(role)
        if curve is None:
            return None
        values = np.asarray(curve.data, dtype=float)[order]
        source_units[role] = curve.unit
        return converter(values, curve.unit) if converter else values

    sonic_curve = find("sonic")
    sonic = None
    if sonic_curve is not None:
        raw = np.asarray(sonic_curve.data, dtype=float)[order]
        try:
            sonic = slowness_to_us_per_m(raw, sonic_curve.unit)
        except UnitError as exc:
            hint = (
                " The values look like us/ft."
                if looks_like_us_per_ft(raw)
                else " The values look like us/m."
            )
            raise UnitError(
                f"{exc} Curve {sonic_curve.mnemonic!r} in {path}.{hint} "
                "Pass the unit explicitly rather than letting SWT assume one."
            ) from exc
        source_units["sonic"] = sonic_curve.unit

    density = convert("density", density_to_g_cm3)
    caliper = convert("caliper", None)
    gamma = convert("gamma", None)

    well_name = name or _header_value(las, "WELL") or _stem(path)

    return WellLogs(
        depth_md_m=depth,
        sonic_us_per_m=sonic,
        density_g_cm3=density,
        caliper=caliper,
        gamma=gamma,
        name=well_name,
        source_units=source_units,
        meta={"path": path, "n_curves": len(las.curves)},
    )


def _header_value(las, key: str) -> str | None:
    try:
        value = las.well[key].value
    except (KeyError, AttributeError):
        return None
    text = str(value).strip()
    return text or None


def _stem(path: str) -> str:
    import os

    return os.path.splitext(os.path.basename(path))[0]
