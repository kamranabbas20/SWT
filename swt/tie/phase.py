"""Constant-phase scan.

Processed seismic rarely arrives at the zero phase its processors intended.  A
residual constant rotation of 20-40 degrees is ordinary; 180 degrees means the
polarity convention is the opposite of what was assumed (SEG normal versus
reverse), which is a labelling problem, not a geophysics problem, and no amount
of stretching will fix it.

Scanning rotations is cheap and it separates two failures that look identical on
screen: a synthetic that is misaligned in *time*, and one that is misaligned in
*phase*.  Both make troughs land on peaks.  Only one is fixed by moving the well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..qc.metrics import correlation_coefficient
from ..synth.convolve import synthesise
from ..synth.resample import TimeReflectivity
from ..trace import Trace, common_window
from ..wavelet.core import Wavelet


@dataclass(frozen=True)
class PhaseScan:
    """Result of rotating a wavelet through every constant phase."""

    best_phase_deg: float
    best_correlation: float
    phases_deg: np.ndarray
    correlations: np.ndarray
    wavelet: Wavelet
    synthetic: Trace

    @property
    def suggests_polarity_flip(self) -> bool:
        """True when the best rotation is closer to 180 than to 0 degrees."""
        return abs(abs(self.best_phase_deg) - 180.0) < 45.0

    def verdict(self) -> str:
        if self.suggests_polarity_flip:
            return (
                f"Best constant phase is {self.best_phase_deg:+.0f} deg, near 180 -- this "
                "is a polarity convention mismatch (SEG normal vs reverse), not a "
                "time-depth problem. Flip the seismic polarity or the impedance "
                "convention rather than shifting the well."
            )
        if abs(self.best_phase_deg) < 15.0:
            return (
                f"Best constant phase is {self.best_phase_deg:+.0f} deg: the data is "
                "effectively zero phase, as processing intended."
            )
        return (
            f"Best constant phase is {self.best_phase_deg:+.0f} deg. A residual rotation "
            "of this size is ordinary in processed data, but carry it explicitly in the "
            "wavelet rather than absorbing it as a time shift."
        )

    def summary(self) -> dict:
        return {
            "best_phase_deg": round(self.best_phase_deg, 1),
            "best_correlation": round(self.best_correlation, 4),
            "suggests_polarity_flip": self.suggests_polarity_flip,
            "verdict": self.verdict(),
        }


def scan_constant_phase(
    seismic: Trace,
    reflectivity: TimeReflectivity,
    wavelet: Wavelet,
    window: tuple[float, float] | None = None,
    step_deg: float = 5.0,
) -> PhaseScan:
    """Rotate the wavelet through all constant phases and keep the best.

    Parameters
    ----------
    step_deg:
        Scan increment.  5 degrees is finer than the estimate is meaningful --
        the point is to distinguish "roughly zero", "roughly 90" and "roughly
        180", not to report a phase to the degree.

    Returns
    -------
    PhaseScan
        Carries the rotated wavelet and its synthetic, so the caller does not
        rebuild them.
    """
    if step_deg <= 0:
        raise ValueError("step_deg must be positive")

    phases = np.arange(-180.0, 180.0, float(step_deg))
    correlations = np.empty_like(phases)

    t0, t1 = window if window is not None else common_window(
        seismic, synthesise(reflectivity, wavelet)
    )
    observed = seismic.window(t0, t1)

    # `best_*` must be assigned from inside the loop, never seeded with the
    # unrotated wavelet: phases[0] is -180 degrees, not 0, so seeding would pair
    # a reported phase of -180 with a zero-phase synthetic whenever no later
    # rotation beat it.  Downstream that mismatch is vicious -- the next
    # cross-correlation compares the trace against its own inverse and picks a
    # side lobe half a period away, and a deterministic wavelet then absorbs the
    # error, leaving a high correlation on a time-depth that is out by half a
    # period.
    best_index = -1
    best_wavelet: Wavelet | None = None
    best_synthetic: Trace | None = None

    for i, phase in enumerate(phases):
        rotated = wavelet.rotate(float(phase)) if phase != 0.0 else wavelet
        synthetic = synthesise(reflectivity, rotated)
        predicted = synthetic.resample_to(observed)
        correlations[i] = correlation_coefficient(observed.amplitude, predicted.amplitude)
        if best_index < 0 or correlations[i] > correlations[best_index]:
            best_index, best_wavelet, best_synthetic = i, rotated, synthetic

    assert best_wavelet is not None and best_synthetic is not None

    return PhaseScan(
        best_phase_deg=float(phases[best_index]),
        best_correlation=float(correlations[best_index]),
        phases_deg=phases,
        correlations=correlations,
        wavelet=best_wavelet,
        synthetic=best_synthetic,
    )
