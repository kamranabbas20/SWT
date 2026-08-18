"""Bulk time shift: the cheap, robust first move of any tie.

Before any stretching, find the single constant shift that best aligns the
synthetic with the seismic.  It costs one cross-correlation and it usually does
most of the work, because the dominant error in a first-pass tie is a static
one -- a wrong start time above the log top (see
:class:`~swt.timedepth.integrate.ShallowModel`), not a wrong velocity trend.

Doing this *first* also protects the steps that follow.  A warping algorithm
handed a synthetic that is 40 ms out will spend its stretch budget removing a
static, and the stretch it reports will be an artefact of that static rather
than a statement about velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

from ..qc.metrics import cross_correlation
from ..timedepth.model import TimeDepth
from ..trace import Trace, common_window


@dataclass(frozen=True)
class BulkShift:
    """The best constant shift between a synthetic and the seismic."""

    shift_s: float
    correlation: float
    correlation_at_zero: float
    lags_s: np.ndarray
    correlations: np.ndarray
    window_s: tuple[float, float]
    runner_up_shift_s: float | None
    runner_up_correlation: float | None
    method: str = "trace"

    @property
    def is_ambiguous(self) -> bool:
        """True when a second peak nearly matches the best one.

        Cycle-skipping -- picking the tie a whole loop early or late -- is the
        classic well-tie blunder, and it looks like a perfectly good tie.  The
        cross-correlation function is where it is visible: two peaks of similar
        height about one dominant period apart.  When that happens the honest
        output is "ambiguous", not a number.
        """
        if self.runner_up_correlation is None:
            return False
        return self.runner_up_correlation > 0.9 * self.correlation

    def summary(self) -> dict:
        out = {
            "method": self.method,
            "shift_ms": round(self.shift_s * 1e3, 2),
            "correlation": round(self.correlation, 4),
            "correlation_at_zero_shift": round(self.correlation_at_zero, 4),
            "window_ms": [round(t * 1e3, 1) for t in self.window_s],
            "ambiguous": self.is_ambiguous,
        }
        if self.runner_up_correlation is not None:
            out["runner_up_shift_ms"] = round(self.runner_up_shift_s * 1e3, 2)
            out["runner_up_correlation"] = round(self.runner_up_correlation, 4)
        return out


def find_bulk_shift(
    seismic: Trace,
    synthetic: Trace,
    window: tuple[float, float] | None = None,
    max_shift_s: float = 0.060,
    method: str = "trace",
) -> BulkShift:
    """Cross-correlate to find the constant shift that best aligns the synthetic.

    A **positive** shift means the synthetic must move later -- equivalently,
    that the well's time-depth model is too fast and every event should be
    pushed down.

    Parameters
    ----------
    max_shift_s:
        Half-width of the lag search.  Keep it to a physically plausible static
        (tens of ms).  A wide search invites the algorithm to find a spurious
        peak a cycle or two away and report it with confidence.
    method:
        ``"envelope"`` correlates the analytic-signal envelopes; ``"trace"``
        (default) correlates the traces themselves.

        Use ``"envelope"`` for the **first** alignment, before the wavelet's
        phase is known.  The envelope is phase-blind, so a synthetic built with
        a zero-phase wavelet still aligns correctly against data carrying 90 or
        180 degrees of residual rotation.  A trace-domain correlation in that
        situation locks onto the wrong half-cycle -- it reports a confident
        shift that is out by half a period, and no later phase scan recovers
        from it, because the phase error has already been converted into a time
        error.

        Then use ``"trace"`` for the fine shift once the phase is known: it is
        the sharper estimator, since the envelope's peak is broad.
    """
    if method not in ("trace", "envelope"):
        raise ValueError(f"method must be 'trace' or 'envelope', got {method!r}")

    t0, t1 = window if window is not None else common_window(seismic, synthetic)
    obs = seismic.window(t0, t1)
    pred = synthetic.resample_to(obs)

    if method == "envelope":
        observed_signal = _envelope(obs.amplitude)
        predicted_signal = _envelope(pred.amplitude)
    else:
        observed_signal, predicted_signal = obs.amplitude, pred.amplitude

    lags, values = cross_correlation(observed_signal, predicted_signal, obs.dt, max_shift_s)

    best = int(np.argmax(values))
    zero = int(np.argmin(np.abs(lags)))

    runner_shift, runner_corr = _runner_up(lags, values, best)

    return BulkShift(
        shift_s=float(lags[best]),
        correlation=float(values[best]),
        correlation_at_zero=float(values[zero]),
        lags_s=lags,
        correlations=values,
        window_s=(t0, t1),
        runner_up_shift_s=runner_shift,
        runner_up_correlation=runner_corr,
        method=method,
    )


def _envelope(x: np.ndarray) -> np.ndarray:
    """Analytic-signal envelope, mean removed.

    The envelope of a wavelet peaks at the wavelet's centre whatever its phase,
    which is what makes it the right thing to correlate before the phase is
    known.
    """
    envelope = np.abs(hilbert(np.asarray(x, dtype=float)))
    return envelope - np.mean(envelope)


def _runner_up(
    lags: np.ndarray, values: np.ndarray, best: int
) -> tuple[float | None, float | None]:
    """Highest local maximum that is not the chosen peak, for ambiguity checks."""
    interior = np.arange(1, values.size - 1)
    if not interior.size:
        return None, None
    is_peak = (values[interior] > values[interior - 1]) & (values[interior] > values[interior + 1])
    peaks = interior[is_peak]
    peaks = peaks[peaks != best]
    if not peaks.size:
        return None, None
    top = int(peaks[np.argmax(values[peaks])])
    return float(lags[top]), float(values[top])


def apply_bulk_shift(time_depth: TimeDepth, shift: BulkShift | float) -> TimeDepth:
    """Fold a bulk shift into the time-depth model.

    Applied to the T-D rather than to the synthetic trace, so that the shift
    becomes part of the well's calibration and survives into everything built
    from it -- horizon picks, depth conversion, the next iteration.  Shifting
    only the displayed trace produces a tie that looks right and exports wrong.
    """
    dt_s = shift.shift_s if isinstance(shift, BulkShift) else float(shift)
    return time_depth.shifted(dt_s)
