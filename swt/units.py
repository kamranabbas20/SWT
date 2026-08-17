"""Unit detection and conversion.

Unit chaos is the single most common source of silently wrong well ties: a sonic
log in us/ft integrated as if it were us/m gives a time-depth curve wrong by a
factor of 3.28. The rule throughout SWT is *detect and assert, never assume*.

Canonical internal units, used by every module downstream of :mod:`swt.io`:

======================  =========
quantity                unit
======================  =========
depth                   m
slowness (DT)           us/m
velocity                m/s
density (RHOB)          g/cm3
time                    s
======================  =========
"""

from __future__ import annotations

import math

M_PER_FT = 0.3048

#: Recognised spellings for each canonical unit, lowercased and stripped of
#: punctuation.  LAS files in the wild are inconsistent, so the mapping is
#: deliberately generous -- but anything not listed raises rather than guesses.
_DEPTH_UNITS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "ft": M_PER_FT,
    "f": M_PER_FT,
    "feet": M_PER_FT,
    "foot": M_PER_FT,
}

_SLOWNESS_UNITS = {
    # value -> multiplier taking the log to us/m
    "us/m": 1.0,
    "usec/m": 1.0,
    "us/me": 1.0,
    "usm": 1.0,
    "microsecondspermeter": 1.0,
    "us/ft": 1.0 / M_PER_FT,
    "usec/ft": 1.0 / M_PER_FT,
    "us/f": 1.0 / M_PER_FT,
    "usft": 1.0 / M_PER_FT,
    "microsecondsperfoot": 1.0 / M_PER_FT,
}

_DENSITY_UNITS = {
    "g/cm3": 1.0,
    "g/cc": 1.0,
    "gm/cc": 1.0,
    "g/c3": 1.0,
    "gcc": 1.0,
    "gram/cc": 1.0,
    "kg/m3": 1e-3,
    "kgm3": 1e-3,
    "kg/m^3": 1e-3,
}

_VELOCITY_UNITS = {
    "m/s": 1.0,
    "ms": 1.0,
    "mps": 1.0,
    "m/sec": 1.0,
    "ft/s": M_PER_FT,
    "fts": M_PER_FT,
    "ft/sec": M_PER_FT,
    "f/s": M_PER_FT,
}

_TIME_UNITS = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "msec": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
}


class UnitError(ValueError):
    """Raised when a unit string cannot be recognised.

    Deliberately fatal: a wrong guess here corrupts every number downstream and
    does so *plausibly*, which is worse than failing.
    """


def _normalise(unit: str) -> str:
    """Lowercase and strip the punctuation and whitespace LAS files vary on."""
    if unit is None:
        raise UnitError("unit is None; SWT will not guess a unit")
    cleaned = str(unit).strip().lower()
    for ch in " .()[]-_":
        cleaned = cleaned.replace(ch, "")
    return cleaned


def _lookup(unit: str, table: dict[str, float], quantity: str) -> float:
    cleaned = _normalise(unit)
    if cleaned not in table:
        raise UnitError(
            f"unrecognised {quantity} unit {unit!r} (normalised to {cleaned!r}); "
            f"known: {sorted(set(table))}"
        )
    return table[cleaned]


def depth_to_m(value, unit: str):
    """Convert a depth (or depth array) to metres."""
    return value * _lookup(unit, _DEPTH_UNITS, "depth")


def slowness_to_us_per_m(value, unit: str):
    """Convert a sonic slowness log to us/m."""
    return value * _lookup(unit, _SLOWNESS_UNITS, "slowness")


def density_to_g_cm3(value, unit: str):
    """Convert a density log to g/cm3."""
    return value * _lookup(unit, _DENSITY_UNITS, "density")


def velocity_to_m_s(value, unit: str):
    """Convert a velocity to m/s."""
    return value * _lookup(unit, _VELOCITY_UNITS, "velocity")


def time_to_s(value, unit: str):
    """Convert a time to seconds."""
    return value * _lookup(unit, _TIME_UNITS, "time")


def slowness_to_velocity(dt_us_per_m):
    """Convert slowness in us/m to velocity in m/s."""
    return 1.0e6 / dt_us_per_m


def velocity_to_slowness(v_m_per_s):
    """Convert velocity in m/s to slowness in us/m."""
    return 1.0e6 / v_m_per_s


def looks_like_us_per_ft(dt_values) -> bool:
    """Heuristic sanity check on a sonic log whose unit string is missing.

    Sedimentary rock slowness is roughly 40-200 us/ft, i.e. 130-650 us/m, so the
    two conventions barely overlap.  This is a *warning* aid only -- callers must
    still supply a unit; SWT never converts on the strength of this alone.
    """
    finite = [v for v in dt_values if v is not None and math.isfinite(v)]
    if not finite:
        return False
    median = sorted(finite)[len(finite) // 2]
    return median < 130.0
