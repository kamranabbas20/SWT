"""A seismic trace on a uniform two-way time axis.

Used for both the recorded seismic and the synthetic, deliberately: every
comparison between them is then a comparison of two objects of the same type on
the same axis, and the "are these on the same sample rate?" class of bug becomes
impossible to write.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Trace:
    """Amplitudes sampled uniformly in two-way time."""

    twt: np.ndarray
    amplitude: np.ndarray
    label: str = "trace"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        twt = np.asarray(self.twt, dtype=float)
        amp = np.asarray(self.amplitude, dtype=float)
        if twt.ndim != 1 or amp.ndim != 1:
            raise ValueError("trace arrays must be 1-D")
        if twt.size != amp.size:
            raise ValueError(f"time axis ({twt.size}) and amplitude ({amp.size}) differ in length")
        if twt.size < 2:
            raise ValueError("a trace needs at least two samples")
        spacing = np.diff(twt)
        if not np.allclose(spacing, spacing[0], rtol=1e-6, atol=1e-12):
            raise ValueError("trace time axis must be uniformly sampled")
        if spacing[0] <= 0:
            raise ValueError("trace time axis must increase")
        object.__setattr__(self, "twt", twt)
        object.__setattr__(self, "amplitude", amp)

    @property
    def dt(self) -> float:
        return float(self.twt[1] - self.twt[0])

    @property
    def n(self) -> int:
        return int(self.twt.size)

    def window(self, t0: float, t1: float) -> "Trace":
        """The part of this trace inside ``[t0, t1]``, inclusive."""
        if t1 <= t0:
            raise ValueError("t1 must exceed t0")
        mask = (self.twt >= t0) & (self.twt <= t1)
        if np.count_nonzero(mask) < 2:
            raise ValueError(
                f"window {t0:.3f}-{t1:.3f} s selects fewer than two samples of "
                f"{self.label} ({self.twt[0]:.3f}-{self.twt[-1]:.3f} s)"
            )
        return Trace(self.twt[mask], self.amplitude[mask], self.label, dict(self.meta))

    def resample_to(self, other: "Trace") -> "Trace":
        """This trace interpolated onto another's time axis.

        Zero outside its own range -- a trace has no amplitude where it was not
        recorded, and inventing one there would flatter every QC metric.
        """
        amp = np.interp(other.twt, self.twt, self.amplitude, left=0.0, right=0.0)
        return Trace(other.twt, amp, self.label, dict(self.meta))

    def shifted(self, dt_s: float) -> "Trace":
        """Move this trace later by ``dt_s`` seconds, keeping the same axis."""
        amp = np.interp(self.twt - float(dt_s), self.twt, self.amplitude, left=0.0, right=0.0)
        return Trace(self.twt, amp, self.label, {**self.meta, "applied_shift_s": float(dt_s)})

    def normalised(self) -> "Trace":
        """Unit-RMS amplitude, for display alongside a trace of another scale."""
        rms = float(np.sqrt(np.mean(self.amplitude**2)))
        if rms == 0.0:
            raise ValueError(f"{self.label} has zero amplitude")
        return Trace(self.twt, self.amplitude / rms, self.label, dict(self.meta))

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.amplitude**2)))

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Trace({self.label!r}, {self.twt[0] * 1e3:.0f}-{self.twt[-1] * 1e3:.0f} ms, "
            f"dt={self.dt * 1e3:.1f} ms, n={self.n})"
        )


def common_window(a: Trace, b: Trace) -> tuple[float, float]:
    """The time range both traces cover."""
    t0 = max(float(a.twt[0]), float(b.twt[0]))
    t1 = min(float(a.twt[-1]), float(b.twt[-1]))
    if t1 <= t0:
        raise ValueError(
            f"{a.label} ({a.twt[0]:.3f}-{a.twt[-1]:.3f} s) and "
            f"{b.label} ({b.twt[0]:.3f}-{b.twt[-1]:.3f} s) do not overlap in time"
        )
    return t0, t1
