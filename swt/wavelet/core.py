"""The wavelet object: samples, a centred time axis, and spectral diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import hilbert


@dataclass(frozen=True)
class Wavelet:
    """A seismic wavelet on a time axis centred on zero.

    The length is forced odd so that ``time()[n // 2] == 0`` exactly.  A wavelet
    with an ambiguous centre produces a synthetic with a half-sample bulk shift,
    which then shows up as a spurious phase residual and gets "fixed" by
    shifting the well -- an error that is easy to make and hard to see.
    """

    samples: np.ndarray
    dt: float
    provenance: str = "unspecified"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=float)
        if samples.ndim != 1:
            raise ValueError("a wavelet must be 1-D")
        if samples.size < 3:
            raise ValueError("a wavelet needs at least three samples")
        if samples.size % 2 == 0:
            raise ValueError(
                f"wavelet length must be odd so the centre is unambiguous (got {samples.size})"
            )
        if not np.all(np.isfinite(samples)):
            raise ValueError("wavelet contains non-finite samples")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "dt", float(self.dt))

    # -- axes --------------------------------------------------------------

    @property
    def n(self) -> int:
        return int(self.samples.size)

    @property
    def centre(self) -> int:
        return self.n // 2

    def time(self) -> np.ndarray:
        """Time axis in seconds, symmetric about zero."""
        return (np.arange(self.n) - self.centre) * self.dt

    def length_s(self) -> float:
        return self.n * self.dt

    # -- transformations ---------------------------------------------------

    def rotate(self, degrees: float) -> "Wavelet":
        """Apply a constant phase rotation.

        ``w_theta = w cos(theta) - H(w) sin(theta)``, with ``H`` the Hilbert
        transform.  A constant rotation is the right model for the residual
        phase of a processed seismic volume; anything more elaborate is usually
        fitting noise.
        """
        theta = np.deg2rad(float(degrees))
        analytic = hilbert(self.samples)
        rotated = self.samples * np.cos(theta) - np.imag(analytic) * np.sin(theta)
        return Wavelet(
            samples=rotated,
            dt=self.dt,
            provenance=f"{self.provenance} rotated {degrees:+.1f} deg",
            meta={**self.meta, "phase_deg": self.meta.get("phase_deg", 0.0) + float(degrees)},
        )

    def scaled(self, factor: float) -> "Wavelet":
        return Wavelet(self.samples * float(factor), self.dt, self.provenance, dict(self.meta))

    def normalised(self) -> "Wavelet":
        """Unit-energy wavelet.  Amplitude scaling is a separate concern to shape."""
        energy = float(np.sqrt(np.sum(self.samples**2)))
        if energy == 0.0:
            raise ValueError("cannot normalise a zero wavelet")
        return self.scaled(1.0 / energy)

    def tapered(self, fraction: float = 0.2) -> "Wavelet":
        """Taper the ends with a Tukey window to suppress truncation ringing."""
        from scipy.signal.windows import tukey

        return Wavelet(
            samples=self.samples * tukey(self.n, alpha=float(fraction)),
            dt=self.dt,
            provenance=f"{self.provenance} tapered",
            meta=dict(self.meta),
        )

    # -- spectral diagnostics ---------------------------------------------

    def amplitude_spectrum(self, pad: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """Single-sided amplitude spectrum, zero-padded for a smooth plot."""
        nfft = int(2 ** np.ceil(np.log2(self.n * max(pad, 1))))
        spectrum = np.abs(np.fft.rfft(self.samples, n=nfft))
        freq = np.fft.rfftfreq(nfft, d=self.dt)
        return freq, spectrum

    def dominant_frequency(self) -> float:
        """Peak of the amplitude spectrum, in Hz."""
        freq, spectrum = self.amplitude_spectrum()
        return float(freq[int(np.argmax(spectrum))])

    def bandwidth(self, level_db: float = -6.0) -> tuple[float, float]:
        """Frequency band (Hz) where the spectrum exceeds ``level_db`` of its peak.

        The bandwidth is not decoration: it sets the number of independent
        samples in the tie window, and therefore whether a given correlation
        coefficient means anything at all (see :mod:`swt.qc.significance`).
        """
        freq, spectrum = self.amplitude_spectrum()
        peak = float(np.max(spectrum))
        if peak <= 0:
            raise ValueError("wavelet has no energy")
        threshold = peak * 10.0 ** (level_db / 20.0)
        above = np.flatnonzero(spectrum >= threshold)
        if not above.size:
            raise ValueError("no frequencies above threshold")
        return float(freq[above[0]]), float(freq[above[-1]])

    def energy_centre_s(self) -> float:
        """Time (s) of the wavelet's centre of energy, relative to its own zero.

        Computed as the centroid of the squared envelope, which is insensitive
        to phase -- a 90-degree wavelet has no single obvious "peak", but its
        energy still has a well defined centre.
        """
        envelope = np.abs(hilbert(self.samples)) ** 2
        total = float(np.sum(envelope))
        if total <= 0:
            return 0.0
        return float(np.sum(envelope * self.time()) / total)

    def recentred(self) -> tuple["Wavelet", float]:
        """Move the wavelet's energy back to zero time, reporting the shift.

        A least-squares wavelet is free to place its energy anywhere inside its
        own length, and it will: handed a synthetic that is a few milliseconds
        out, the estimator returns a wavelet whose energy is offset by exactly
        that much, because doing so fits the data.  The tie then looks converged
        while carrying a residual static, and the *next* bulk-shift search finds
        nothing to correct -- the error has been hidden inside the wavelet
        rather than removed.

        Recentring separates the two again.  The returned shift belongs in the
        time-depth model, where it is visible, exportable, and correctable::

            wavelet, shift = wavelet.recentred()
            time_depth = time_depth.shifted(shift)

        The shift is quantised to whole samples so the wavelet can be rolled
        without interpolating it -- an interpolated wavelet is a different
        wavelet, with a different spectrum.
        """
        offset = self.energy_centre_s()
        samples = int(round(offset / self.dt))
        if samples == 0:
            return self, 0.0

        rolled = np.roll(self.samples, -samples)
        # Rolling wraps; zero the wrapped tail rather than letting energy
        # reappear at the other end of the wavelet.
        if samples > 0:
            rolled[-samples:] = 0.0
        else:
            rolled[:-samples] = 0.0

        return (
            Wavelet(
                samples=rolled,
                dt=self.dt,
                provenance=f"{self.provenance} recentred by {samples * self.dt * 1e3:+.1f} ms",
                meta=dict(self.meta),
            ),
            samples * self.dt,
        )

    def estimated_phase(self) -> float:
        """Constant phase implied by the wavelet, in degrees.

        Estimated from the analytic signal at the wavelet's centre, which is
        exact for a genuinely constant-phase wavelet and a reasonable summary
        otherwise.
        """
        analytic = hilbert(self.samples)
        peak = int(np.argmax(np.abs(analytic)))
        return float(np.rad2deg(np.angle(analytic[peak])))

    def summary(self) -> dict:
        """Compact JSON-safe description -- the shape the copilot tools return."""
        low, high = self.bandwidth()
        return {
            "provenance": self.provenance,
            "length_ms": round(self.length_s() * 1e3, 1),
            "dt_ms": round(self.dt * 1e3, 3),
            "dominant_frequency_hz": round(self.dominant_frequency(), 2),
            "bandwidth_hz": [round(low, 2), round(high, 2)],
            "phase_deg": round(self.estimated_phase(), 1),
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Wavelet(n={self.n}, dt={self.dt * 1e3:.1f} ms, "
            f"fdom={self.dominant_frequency():.1f} Hz, provenance={self.provenance!r})"
        )


def odd_length(n: int) -> int:
    """Round a sample count up to the next odd number."""
    n = int(n)
    return n if n % 2 else n + 1
