"""Deterministic (least-squares) wavelet estimation from the well.

This is the good estimator: it solves directly for the filter that best maps the
well's reflectivity onto the seismic trace,

    minimise  || R w - s ||^2 + lambda || w ||^2

where ``R`` is the convolution matrix of the reflectivity series.  Unlike the
statistical estimator it recovers **amplitude and phase together**, which is
what makes a residual-phase QC meaningful.

The catch, and the reason this is not simply used first: it presupposes that the
reflectivity is already at approximately the right time.  Feed it a poor
time-depth model and it will faithfully return the filter that best maps a
misaligned reflectivity onto the seismic -- a wavelet that is wrong in a way
that hides the misalignment.  So the sequence is always: statistical wavelet ->
rough tie -> deterministic wavelet -> better tie.

The ridge term is not optional.  ``R^T R`` is badly conditioned whenever the
reflectivity is band-limited (always), and the unregularised solution puts large
oscillating values in the wavelet's tails to fit noise.
"""

from __future__ import annotations

import numpy as np

from .core import Wavelet, odd_length


def deterministic_wavelet(
    rc: np.ndarray,
    trace: np.ndarray,
    dt: float,
    length_s: float = 0.128,
    ridge: float = 1e-2,
) -> Wavelet:
    """Least-squares wavelet matching a reflectivity series to a seismic trace.

    Parameters
    ----------
    rc, trace:
        Reflectivity and seismic on the *same* uniform time axis, already
        restricted to the design window.  Same length.
    length_s:
        Wavelet length.  A longer wavelet fits better and generalises worse; a
        quarter of the design window is a sane ceiling.
    ridge:
        Regularisation weight, relative to the mean diagonal of ``R^T R``
        (so it is scale-free -- ``ridge=1e-2`` means the same thing whatever the
        amplitude units of the seismic).  Raise it when the wavelet's tails ring.

    Returns
    -------
    Wavelet
        Not normalised: the least-squares scale is meaningful here, because it
        is the scalar that actually matches synthetic amplitude to seismic
        amplitude.
    """
    r = np.asarray(rc, dtype=float)
    s = np.asarray(trace, dtype=float)

    if r.shape != s.shape:
        raise ValueError(f"reflectivity {r.shape} and trace {s.shape} differ in shape")
    if r.ndim != 1:
        raise ValueError("reflectivity and trace must be 1-D")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(s))):
        raise ValueError("reflectivity or trace contains non-finite samples")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    nw = odd_length(int(round(length_s / dt)))
    if nw >= r.size:
        raise ValueError(
            f"wavelet length ({nw} samples) must be shorter than the design window "
            f"({r.size} samples)"
        )
    if not np.any(r):
        raise ValueError("reflectivity is all zero in this window; nothing to match")

    design = _convolution_matrix(r, nw)

    gram = design.T @ design
    rhs = design.T @ s
    scale = float(np.mean(np.diag(gram)))
    if scale <= 0:
        raise ValueError("degenerate design matrix")
    regularised = gram + ridge * scale * np.eye(nw)

    samples = np.linalg.solve(regularised, rhs)

    residual = s - design @ samples
    energy = float(s @ s)
    pep = 1.0 - float(residual @ residual) / energy if energy > 0 else 0.0

    return Wavelet(
        samples=samples,
        dt=dt,
        provenance=f"deterministic least-squares ({nw * dt * 1e3:.0f} ms, ridge={ridge:g})",
        meta={"design_window_samples": int(r.size), "design_pep": round(pep, 4)},
    )


def _convolution_matrix(rc: np.ndarray, nw: int) -> np.ndarray:
    """Columns are the reflectivity shifted by each wavelet lag.

    Column ``j`` carries lag ``j - nw // 2``, so a symmetric wavelet comes back
    centred and the estimate has no built-in bulk shift.
    """
    n = rc.size
    half = nw // 2
    matrix = np.zeros((n, nw), dtype=float)
    for j in range(nw):
        lag = j - half
        if lag >= 0:
            matrix[lag:, j] = rc[: n - lag]
        else:
            matrix[:lag, j] = rc[-lag:]
    return matrix
