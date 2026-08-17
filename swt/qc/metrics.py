"""Tie quality metrics.

Four numbers, each answering a different question:

``correlation``
    How well do the two traces match in shape?
``nrms``
    How well do they match in shape *and* amplitude?  Two traces can correlate
    at 0.9 and still have an NRMS of 60% if the scaling is wrong.
``pep``
    What fraction of the seismic energy does the synthetic actually explain?
    Computed after optimal scaling, because the absolute amplitude of a
    synthetic is arbitrary -- it depends on the wavelet's normalisation, not on
    the earth.
``best_lag``
    Is there a residual bulk shift?  A tie reported at zero lag whose peak
    correlation is at 12 ms is not a finished tie.

None of these tells you whether the number is *significant*; that is
:mod:`swt.qc.significance`, and it should be read alongside every one of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..trace import Trace, common_window


@dataclass(frozen=True)
class TieMetrics:
    """Quality metrics for one synthetic-versus-seismic comparison."""

    correlation: float
    nrms_percent: float
    pep: float
    best_lag_s: float
    best_lag_correlation: float
    optimal_scalar: float
    window_s: tuple[float, float]
    n_samples: int

    def summary(self) -> dict:
        """Compact JSON-safe summary -- the shape the copilot tools return."""
        return {
            "correlation": round(self.correlation, 4),
            "nrms_percent": round(self.nrms_percent, 2),
            "pep": round(self.pep, 4),
            "best_lag_ms": round(self.best_lag_s * 1e3, 2),
            "best_lag_correlation": round(self.best_lag_correlation, 4),
            "window_ms": [round(t * 1e3, 1) for t in self.window_s],
            "n_samples": self.n_samples,
        }

    @property
    def residual_shift_is_significant(self) -> bool:
        """True when moving to the best lag would meaningfully improve the tie."""
        return (
            abs(self.best_lag_s) > 0.5 * 1e-3
            and self.best_lag_correlation > self.correlation + 0.02
        )


def correlation_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-lag Pearson correlation, with the means removed."""
    x = np.asarray(a, float) - np.mean(a)
    y = np.asarray(b, float) - np.mean(b)
    denominator = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(x * y) / denominator)


def nrms(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised RMS difference, in percent.

    ``200 * RMS(a - b) / (RMS(a) + RMS(b))``.  Zero for identical traces, 200%
    for traces of equal amplitude and opposite polarity.  Unlike correlation it
    is sensitive to amplitude and to static shift, which is exactly why it is
    worth reporting next to correlation rather than instead of it.
    """
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    rms_x = float(np.sqrt(np.mean(x**2)))
    rms_y = float(np.sqrt(np.mean(y**2)))
    if rms_x + rms_y == 0.0:
        return 0.0
    return float(200.0 * np.sqrt(np.mean((x - y) ** 2)) / (rms_x + rms_y))


def proportion_of_energy_predicted(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Fraction of observed energy explained by the prediction, after scaling.

    Returns ``(pep, scalar)``.  The scalar is the least-squares amplitude match;
    reporting it matters because a scalar far from 1 means the wavelet's
    amplitude calibration is off even when the shape is right.
    """
    s = np.asarray(observed, float)
    p = np.asarray(predicted, float)
    denominator = float(p @ p)
    scalar = float(s @ p) / denominator if denominator > 0 else 0.0
    residual = s - scalar * p
    energy = float(s @ s)
    if energy == 0.0:
        return 0.0, scalar
    return 1.0 - float(residual @ residual) / energy, scalar


def cross_correlation(
    a: np.ndarray, b: np.ndarray, dt: float, max_lag_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation of ``a`` and ``b`` over a lag range.

    A positive lag means ``b`` must be moved *later* to match ``a``.
    """
    x = np.asarray(a, float) - np.mean(a)
    y = np.asarray(b, float) - np.mean(b)
    max_lag = int(round(max_lag_s / dt))
    max_lag = min(max_lag, x.size - 2)
    if max_lag < 1:
        raise ValueError("lag range is shorter than one sample")

    norm = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    raw = np.correlate(x, y, mode="full")
    centre = x.size - 1
    values = raw[centre - max_lag : centre + max_lag + 1]
    lags = np.arange(-max_lag, max_lag + 1) * dt
    return lags, (values / norm if norm > 0 else np.zeros_like(values))


def evaluate(
    seismic: Trace,
    synthetic: Trace,
    window: tuple[float, float] | None = None,
    max_lag_s: float = 0.040,
) -> TieMetrics:
    """Compare a synthetic against the seismic over a time window.

    The window defaults to the overlap of the two traces.  In practice it should
    be set to the logged interval -- comparing over a range where the synthetic
    is zero-padded flatters nothing and measures nothing.
    """
    t0, t1 = window if window is not None else common_window(seismic, synthetic)
    obs = seismic.window(t0, t1)
    pred = synthetic.window(t0, t1) if synthetic.dt == seismic.dt else synthetic.resample_to(obs)
    if pred.n != obs.n:
        pred = pred.resample_to(obs)

    corr = correlation_coefficient(obs.amplitude, pred.amplitude)
    pep, scalar = proportion_of_energy_predicted(obs.amplitude, pred.amplitude)
    error = nrms(obs.amplitude, scalar * pred.amplitude)

    lags, values = cross_correlation(obs.amplitude, pred.amplitude, obs.dt, max_lag_s)
    best = int(np.argmax(values))

    return TieMetrics(
        correlation=corr,
        nrms_percent=error,
        pep=pep,
        best_lag_s=float(lags[best]),
        best_lag_correlation=float(values[best]),
        optimal_scalar=scalar,
        window_s=(t0, t1),
        n_samples=obs.n,
    )
