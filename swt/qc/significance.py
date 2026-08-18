"""Is that correlation coefficient actually meaningful?

This is the module most well-tie software does not have, and its absence is why
ties get accepted that should not be.

A seismic trace is band-limited.  Over a 200 ms window with a 10-50 Hz band
there are not 100 independent numbers being compared -- there are roughly
``2 * B * T = 2 * 40 * 0.2 = 16``.  Correlating two random band-limited series
of that length gives a coefficient around 0.5 *by chance*, routinely.  So a tie
reported at 0.6 over a short window in a narrow band may be indistinguishable
from noise, while 0.6 over a long window in a broad band is a solid result.  The
raw coefficient alone cannot tell those apart, and a coloured "good/bad"
threshold on it is actively misleading.

The treatment here follows the standard idea -- due in this context to White
(1980) and set out for practitioners in White & Simm's well-tie tutorials --
that the number of independent samples is the time-bandwidth product.  The
statistics applied to that count are the ordinary ones: a Student-t test for
whether the correlation differs from zero, and a Fisher z-transform for its
confidence interval.  This is not a verbatim implementation of any single
published formula, and the docstrings say so rather than implying a false
precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Significance:
    """Whether a measured correlation can be distinguished from chance."""

    correlation: float
    effective_dof: float
    critical_correlation: float
    p_value: float
    confidence_interval: tuple[float, float]
    window_length_s: float
    bandwidth_hz: float
    confidence_level: float

    @property
    def is_significant(self) -> bool:
        return self.p_value < (1.0 - self.confidence_level)

    def verdict(self) -> str:
        """One sentence a human can act on, for the report and the copilot."""
        pct = int(round(self.confidence_level * 100))
        if self.effective_dof < 4:
            return (
                f"The window holds only ~{self.effective_dof:.0f} independent samples "
                f"({self.window_length_s * 1e3:.0f} ms at {self.bandwidth_hz:.0f} Hz bandwidth). "
                "No correlation measured over it can be trusted -- widen the window."
            )
        if self.is_significant:
            lo, hi = self.confidence_interval
            return (
                f"Correlation {self.correlation:.2f} is significant at {pct}% "
                f"(~{self.effective_dof:.0f} independent samples; chance level "
                f"{self.critical_correlation:.2f}). {pct}% interval {lo:.2f} to {hi:.2f}."
            )
        return (
            f"Correlation {self.correlation:.2f} is NOT significant at {pct}%: with only "
            f"~{self.effective_dof:.0f} independent samples in this window, "
            f"{self.critical_correlation:.2f} would arise by chance. "
            "Widen the tie window or broaden the band before trusting this tie."
        )

    def summary(self) -> dict:
        return {
            "correlation": round(self.correlation, 4),
            "effective_dof": round(self.effective_dof, 1),
            "critical_correlation": round(self.critical_correlation, 4),
            "p_value": float(f"{self.p_value:.3g}"),
            "confidence_interval": [round(v, 4) for v in self.confidence_interval],
            "is_significant": self.is_significant,
            "verdict": self.verdict(),
        }


def effective_dof(window_length_s: float, bandwidth_hz: float) -> float:
    """Independent sample count in a band-limited window: ``2 * B * T + 1``.

    The sample *rate* deliberately does not appear.  Resampling a 200 ms window
    from 4 ms to 1 ms quadruples the sample count and adds no information; a
    significance test driven by the raw count would reward that, which is
    precisely the failure mode this module exists to prevent.
    """
    if window_length_s <= 0:
        raise ValueError("window_length_s must be positive")
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    return 2.0 * bandwidth_hz * window_length_s + 1.0


def assess(
    correlation: float,
    window_length_s: float,
    bandwidth_hz: float,
    confidence_level: float = 0.95,
) -> Significance:
    """Test a correlation coefficient against the chance level for its window.

    Parameters
    ----------
    correlation:
        The measured coefficient, from :func:`swt.qc.metrics.evaluate`.
    window_length_s:
        Length of the tie window in seconds.
    bandwidth_hz:
        Usable bandwidth of the seismic, e.g. the -6 dB width of the estimated
        wavelet (:meth:`swt.wavelet.core.Wavelet.bandwidth`).
    """
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"correlation must lie in [-1, 1], got {correlation}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")

    n = effective_dof(window_length_s, bandwidth_hz)
    alpha = 1.0 - confidence_level

    # Student-t test of rho = 0, with n - 2 degrees of freedom.
    df = max(n - 2.0, 1.0)
    r = float(np.clip(abs(correlation), 0.0, 1.0 - 1e-12))
    t_stat = r * np.sqrt(df / max(1.0 - r**2, 1e-12))
    p_value = float(2.0 * stats.t.sf(t_stat, df))

    t_crit = float(stats.t.isf(alpha / 2.0, df))
    critical = float(t_crit / np.sqrt(df + t_crit**2))

    # Fisher z confidence interval.  Needs n > 3 to have a defined variance.
    if n > 3.0:
        z = np.arctanh(np.clip(correlation, -1.0 + 1e-12, 1.0 - 1e-12))
        se = 1.0 / np.sqrt(n - 3.0)
        z_crit = float(stats.norm.isf(alpha / 2.0))
        interval = (float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se)))
    else:
        interval = (-1.0, 1.0)

    return Significance(
        correlation=float(correlation),
        effective_dof=float(n),
        critical_correlation=critical,
        p_value=p_value,
        confidence_interval=interval,
        window_length_s=float(window_length_s),
        bandwidth_hz=float(bandwidth_hz),
        confidence_level=float(confidence_level),
    )
