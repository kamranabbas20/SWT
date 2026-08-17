"""Statistical wavelet estimation from the seismic alone.

The amplitude spectrum of a wavelet can be recovered from the seismic without
any well, under one assumption: that the earth's reflectivity is white, so the
seismic autocorrelation is the wavelet autocorrelation.  That assumption is
never exactly true, but it is good enough to get a first wavelet, and it has a
decisive practical advantage -- **it does not need a time-depth model**, so it
works before the tie exists.

What it cannot recover is phase.  The autocorrelation discards it completely.
So this estimator returns a zero-phase wavelet and the phase is found separately
by scanning (:mod:`swt.tie.phase`), or replaced later by a deterministic
estimate once the tie is good enough to support one.
"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import tukey

from .core import Wavelet, odd_length


def statistical_wavelet(
    trace: np.ndarray,
    dt: float,
    length_s: float = 0.128,
    phase_deg: float = 0.0,
    taper: float = 0.25,
) -> Wavelet:
    """Estimate a wavelet from the autocorrelation of a seismic trace.

    Parameters
    ----------
    trace:
        The seismic in the design window -- ideally a few hundred ms around the
        zone of interest.  Longer is statistically better but risks averaging
        over a real change in the wavelet with depth.
    length_s:
        Wavelet length.  Must be shorter than the design window; a wavelet
        approaching the window length is fitting the window, not the data.
    phase_deg:
        Constant phase applied to the zero-phase result.
    taper:
        Tukey taper fraction applied to the autocorrelation before transforming.
        Suppresses the ringing that a hard-truncated autocorrelation puts into
        the spectrum.

    Returns
    -------
    Wavelet
        Unit-energy, so that amplitude scaling stays a separate decision from
        wavelet shape.
    """
    x = np.asarray(trace, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("need a 1-D trace with at least 8 samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("trace contains non-finite samples")
    if dt <= 0:
        raise ValueError("dt must be positive")

    nw = odd_length(int(round(length_s / dt)))
    if nw >= x.size:
        raise ValueError(
            f"wavelet length ({nw} samples) must be shorter than the design window "
            f"({x.size} samples); use a longer window or a shorter wavelet"
        )

    x = x - np.mean(x)
    if not np.any(x):
        raise ValueError("trace is constant; there is no wavelet to estimate")

    autocorr = np.correlate(x, x, mode="full")
    centre = x.size - 1
    half = nw // 2
    segment = autocorr[centre - half : centre + half + 1]
    segment = segment * tukey(nw, alpha=float(taper))

    # Autocorrelation transforms to the power spectrum; the wavelet's amplitude
    # spectrum is its square root.  Clip the tiny negative values that leakage
    # can produce rather than letting a NaN through.
    power = np.fft.rfft(segment).real
    amplitude = np.sqrt(np.clip(power, 0.0, None))

    zero_phase = np.fft.fftshift(np.fft.irfft(amplitude, n=nw))

    wavelet = Wavelet(
        samples=zero_phase,
        dt=dt,
        provenance=f"statistical ({x.size} samples, {nw * dt * 1e3:.0f} ms)",
        meta={"phase_deg": 0.0, "design_window_samples": int(x.size)},
    ).normalised()

    return wavelet.rotate(phase_deg) if phase_deg else wavelet
