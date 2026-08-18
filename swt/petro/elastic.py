"""Acoustic impedance, normal-incidence reflectivity, and Backus upscaling."""

from __future__ import annotations

import numpy as np


def acoustic_impedance(velocity_m_s, density_g_cm3):
    """P-impedance from velocity (m/s) and density (g/cm3).

    Returned in (m/s)*(g/cm3).  The absolute scale is irrelevant to a well tie --
    reflectivity is a ratio, so any consistent unit cancels -- but keeping it in
    log units means the number plotted next to the logs is the number a user
    recognises.
    """
    v = np.asarray(velocity_m_s, dtype=float)
    rho = np.asarray(density_g_cm3, dtype=float)
    if v.shape != rho.shape:
        raise ValueError(f"velocity {v.shape} and density {rho.shape} differ in shape")
    return v * rho


def reflectivity(impedance) -> np.ndarray:
    """Normal-incidence reflection coefficients from an impedance series.

    ``r_i = (AI_{i+1} - AI_i) / (AI_{i+1} + AI_i)``

    The result has one fewer sample than the input: a reflection coefficient
    belongs to an *interface*, not a layer.  Callers that need a same-length
    series should decide explicitly where to pad; SWT does not pick for them.
    """
    ai = np.asarray(impedance, dtype=float)
    if ai.ndim != 1 or ai.size < 2:
        raise ValueError("need a 1-D impedance series with at least two samples")
    if np.any(ai <= 0):
        raise ValueError("impedance must be strictly positive")
    top, base = ai[:-1], ai[1:]
    return (base - top) / (base + top)


def backus_average(
    depth_m: np.ndarray,
    velocity_m_s: np.ndarray,
    density_g_cm3: np.ndarray,
    length_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Upscale a log to seismic scale by acoustic Backus averaging.

    A sonic log samples every 0.15 m; seismic resolves tens of metres.  Backus
    averaging is the physically correct way to bridge that: it averages elastic
    *compliance*, not velocity, because a stack of thin layers behaves like a
    single layer whose modulus is the harmonic -- not arithmetic -- mean.

    For normal incidence only P-wave modulus is needed, so no shear log is
    required::

        M_eff   = 1 / <1 / (rho * Vp^2)>      harmonic mean of the modulus
        rho_eff = <rho>                       arithmetic mean of density
        Vp_eff  = sqrt(M_eff / rho_eff)

    Parameters
    ----------
    length_m:
        Averaging window.  The usual rule of thumb is a third of the dominant
        wavelength: at 30 Hz and 3000 m/s that is a 100 m wavelength, so ~33 m.
        Too long and real reflectors are smeared away; too short and nothing
        changes.  This is a modelling choice, and the UI exposes it as one.

    Returns
    -------
    Upscaled velocity and density on the *input* depth axis.
    """
    depth = np.asarray(depth_m, dtype=float)
    v = np.asarray(velocity_m_s, dtype=float)
    rho = np.asarray(density_g_cm3, dtype=float)

    if not (depth.shape == v.shape == rho.shape):
        raise ValueError("depth, velocity and density must have the same shape")
    if depth.size < 2:
        raise ValueError("need at least two samples to upscale")
    if length_m <= 0:
        raise ValueError("length_m must be positive")

    spacing = np.diff(depth)
    if np.any(spacing <= 0):
        raise ValueError("depth must be strictly increasing")
    median_spacing = float(np.median(spacing))
    if not np.allclose(spacing, median_spacing, rtol=1e-3):
        raise ValueError(
            "Backus averaging expects a regularly sampled depth axis; "
            "resample the log to a constant increment first"
        )

    window = int(round(length_m / median_spacing))
    if window < 2:
        # Averaging length is below the log sample rate: nothing to upscale.
        return v.copy(), rho.copy()

    modulus = rho * v**2
    compliance_avg = _running_mean(1.0 / modulus, window)
    rho_avg = _running_mean(rho, window)

    modulus_eff = 1.0 / compliance_avg
    return np.sqrt(modulus_eff / rho_avg), rho_avg


def _running_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred running mean with edge samples averaged over what exists.

    Reflecting or zero-padding the edges would invent impedance contrasts at the
    top and bottom of the log, which then appear as spurious reflectors in the
    synthetic.  Shrinking the window at the edges does not.
    """
    n = values.size
    kernel = np.ones(window)
    padded = np.convolve(values, kernel, mode="same")
    counts = np.convolve(np.ones(n), kernel, mode="same")
    return padded / counts
