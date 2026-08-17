"""The convolutional model: synthetic = wavelet * reflectivity."""

from __future__ import annotations

import numpy as np

from ..trace import Trace
from ..wavelet.core import Wavelet
from .resample import TimeReflectivity


def synthesise(reflectivity: TimeReflectivity, wavelet: Wavelet, label: str = "synthetic") -> Trace:
    """Convolve a time-domain reflectivity series with a wavelet.

    The wavelet's sample rate must match the reflectivity's -- an interpolated
    wavelet is a different wavelet, so SWT refuses rather than resampling it
    silently.

    ``mode="same"`` keeps the output on the reflectivity's own time axis, and
    because :class:`~swt.wavelet.core.Wavelet` is forced to odd length that
    centring is exact.
    """
    dt_rc = reflectivity.dt
    if not np.isclose(dt_rc, wavelet.dt, rtol=1e-9, atol=1e-12):
        raise ValueError(
            f"sample rate mismatch: reflectivity at {dt_rc * 1e3:.3f} ms, "
            f"wavelet at {wavelet.dt * 1e3:.3f} ms"
        )

    amplitude = np.convolve(reflectivity.rc, wavelet.samples, mode="same")

    return Trace(
        twt=reflectivity.twt,
        amplitude=amplitude,
        label=label,
        meta={
            "wavelet": wavelet.provenance,
            "valid_window_s": reflectivity.valid_window(),
        },
    )


def synthetic_from_logs(
    twt_log: np.ndarray,
    impedance: np.ndarray,
    wavelet: Wavelet,
    dt: float | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
) -> tuple[Trace, TimeReflectivity]:
    """Convenience path: impedance log plus wavelet straight to a synthetic.

    Returns the synthetic and the intermediate reflectivity, because the
    reflectivity is worth plotting in its own right -- it is where thin-bed
    detail visibly disappears at seismic scale.
    """
    from .resample import reflectivity_in_time

    rc = reflectivity_in_time(twt_log, impedance, dt or wavelet.dt, t_start, t_end)
    return synthesise(rc, wavelet), rc
