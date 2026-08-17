"""Analytic wavelets: Ricker and Ormsby.

These are for the first look and for sanity checks.  A Ricker is never the real
wavelet of a processed seismic volume, but a synthetic built with one tells you
immediately whether the polarity, the rough frequency content and the time-depth
are in the right ballpark -- before any estimation machinery is involved.
"""

from __future__ import annotations

import numpy as np

from .core import Wavelet, odd_length


def ricker(frequency: float, dt: float, length_s: float = 0.128) -> Wavelet:
    """Zero-phase Ricker wavelet of the given peak frequency (Hz).

    ``w(t) = (1 - 2 pi^2 f^2 t^2) exp(-pi^2 f^2 t^2)``
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if length_s <= 0:
        raise ValueError("length_s must be positive")

    n = odd_length(int(round(length_s / dt)))
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * frequency * t) ** 2
    samples = (1.0 - 2.0 * a) * np.exp(-a)
    return Wavelet(
        samples=samples,
        dt=dt,
        provenance=f"Ricker {frequency:.0f} Hz",
        meta={"phase_deg": 0.0, "peak_frequency_hz": float(frequency)},
    )


def ormsby(f1: float, f2: float, f3: float, f4: float, dt: float, length_s: float = 0.128) -> Wavelet:
    """Zero-phase Ormsby (trapezoidal bandpass) wavelet.

    ``f1`` low cut, ``f2`` low pass, ``f3`` high pass, ``f4`` high cut, in Hz.
    Closer than a Ricker to the band of real processed data, because real data
    has a flat-topped spectrum with roll-offs, not a Gaussian one.
    """
    for name, value in (("f1", f1), ("f2", f2), ("f3", f3), ("f4", f4)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not f1 < f2 < f3 < f4:
        raise ValueError(f"need f1 < f2 < f3 < f4, got {f1}, {f2}, {f3}, {f4}")
    if f4 >= 0.5 / dt:
        raise ValueError(f"f4 ({f4} Hz) is at or above Nyquist ({0.5 / dt:.1f} Hz)")

    n = odd_length(int(round(length_s / dt)))
    t = (np.arange(n) - n // 2) * dt

    def term(f: float) -> np.ndarray:
        return (np.pi * f**2) * np.sinc(f * t) ** 2

    samples = (
        (term(f4) - term(f3)) / (f4 - f3) - (term(f2) - term(f1)) / (f2 - f1)
    )
    samples = samples / np.max(np.abs(samples))
    return Wavelet(
        samples=samples,
        dt=dt,
        provenance=f"Ormsby {f1:.0f}-{f2:.0f}-{f3:.0f}-{f4:.0f} Hz",
        meta={"phase_deg": 0.0, "corner_frequencies_hz": [f1, f2, f3, f4]},
    )
