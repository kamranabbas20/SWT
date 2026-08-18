"""Depth-to-time resampling of the impedance log -- done the correct way.

This module is short and it is the most important twenty lines in the package.

A sonic log sampled every 0.15 m produces time samples spaced roughly 0.1 ms
apart near the top of a well and less than that at depth.  The seismic is
sampled at 2 or 4 ms.  Getting from one to the other by ``np.interp`` -- reading
the impedance at each 2 ms tick -- is *decimation without an anti-alias filter*.
It aliases every thin bed in the log into the seismic band, silently changes the
amplitude spectrum of the reflectivity, and the result still looks entirely
reasonable on screen.  It is a favourite way to produce a confident wrong tie.

The correct operation is to **average impedance over each output time bin**
before differencing.  Bin-averaging is an ideal boxcar low-pass filter followed
by decimation, which is exactly what is wanted, and it has an exact closed form:
integrate impedance over time once, then difference the integral at the bin
edges.  No explicit filter design, no window choice, no leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..petro.elastic import reflectivity


@dataclass(frozen=True)
class TimeReflectivity:
    """A reflectivity series on a uniform two-way time axis.

    Attributes
    ----------
    twt:
        Uniform time axis (s) -- the sample times of ``rc``.
    rc:
        Reflection coefficients, zero outside the interval the log covers.
    valid:
        Boolean mask, ``True`` where the log actually supports the value.  The
        zeros outside are padding, not measurements, and every QC window must be
        restricted to ``valid`` or the metrics are computed against silence.
    """

    twt: np.ndarray
    rc: np.ndarray
    valid: np.ndarray

    @property
    def dt(self) -> float:
        return float(self.twt[1] - self.twt[0])

    def valid_window(self) -> tuple[float, float]:
        """First and last time (s) covered by the log."""
        idx = np.flatnonzero(self.valid)
        if not idx.size:
            raise ValueError("no valid samples: the log does not overlap the time axis")
        return float(self.twt[idx[0]]), float(self.twt[idx[-1]])


def reflectivity_in_time(
    twt_log: np.ndarray,
    impedance: np.ndarray,
    dt: float,
    t_start: float | None = None,
    t_end: float | None = None,
) -> TimeReflectivity:
    """Resample an impedance log to uniform time and difference it.

    Parameters
    ----------
    twt_log:
        Two-way time (s) of each log sample -- ``TimeDepth.twt``.  Strictly
        increasing, which the time-depth model already guarantees.
    impedance:
        Acoustic impedance at those samples, strictly positive.
    dt:
        Output sample interval in seconds, e.g. ``0.002``.
    t_start, t_end:
        Output time range.  Defaults to the log's own range, snapped outward to
        whole samples.  Pass the seismic trace's range to get a reflectivity
        series that lines up sample-for-sample with the data.

    Returns
    -------
    TimeReflectivity
        ``rc[k]`` is the coefficient of the interface between blocks ``k`` and
        ``k+1``, carried at time ``twt[k] + dt/2`` in principle; SWT indexes it
        at ``twt[k]``, matching the convention that a synthetic sample at time
        *t* is the response of the interface at *t*.
    """
    t_log = np.asarray(twt_log, dtype=float)
    ai = np.asarray(impedance, dtype=float)

    if t_log.shape != ai.shape:
        raise ValueError(f"time {t_log.shape} and impedance {ai.shape} differ in shape")
    if t_log.size < 2:
        raise ValueError("need at least two log samples")
    if np.any(np.diff(t_log) <= 0):
        raise ValueError("log times must be strictly increasing (check the time-depth model)")
    if np.any(ai <= 0) or not np.all(np.isfinite(ai)):
        raise ValueError("impedance must be finite and strictly positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    log_t0, log_t1 = float(t_log[0]), float(t_log[-1])
    t0 = log_t0 if t_start is None else float(t_start)
    t1 = log_t1 if t_end is None else float(t_end)
    if t1 <= t0:
        raise ValueError("t_end must exceed t_start")

    n = int(np.floor((t1 - t0) / dt)) + 1
    axis = t0 + dt * np.arange(n)

    # Bin edges: sample k averages impedance over [t_k, t_k + dt).
    edges = np.concatenate([axis, [axis[-1] + dt]])

    # Exact bin means via the cumulative integral of impedance over time.
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (ai[:-1] + ai[1:]) * np.diff(t_log))])
    clipped = np.clip(edges, log_t0, log_t1)
    integral = np.interp(clipped, t_log, cumulative)
    span = np.diff(clipped)

    with np.errstate(invalid="ignore", divide="ignore"):
        block_ai = np.where(span > 0, np.diff(integral) / np.where(span > 0, span, 1.0), np.nan)

    # A bin is usable only if it lies wholly inside the logged interval;
    # partially covered end bins are averages over a shorter span and would
    # carry a spurious impedance contrast.
    fully_covered = (edges[:-1] >= log_t0) & (edges[1:] <= log_t1)
    block_valid = fully_covered & np.isfinite(block_ai)

    rc = np.zeros(n, dtype=float)
    valid = np.zeros(n, dtype=bool)

    pair_valid = block_valid[:-1] & block_valid[1:]
    if np.any(pair_valid):
        top = np.where(pair_valid, block_ai[:-1], 1.0)
        base = np.where(pair_valid, block_ai[1:], 1.0)
        rc[:-1] = np.where(pair_valid, (base - top) / (base + top), 0.0)
        valid[:-1] = pair_valid

    return TimeReflectivity(twt=axis, rc=rc, valid=valid)


def impedance_in_time(
    twt_log: np.ndarray, impedance: np.ndarray, dt: float, t_start: float, t_end: float
) -> tuple[np.ndarray, np.ndarray]:
    """Block-averaged impedance on a uniform time axis, for display.

    Same averaging as :func:`reflectivity_in_time`, stopping before the
    difference -- useful for plotting the upscaled impedance the synthetic
    actually saw, which is rarely the impedance the user thinks they supplied.
    """
    tr = reflectivity_in_time(twt_log, impedance, dt, t_start, t_end)
    t_log = np.asarray(twt_log, float)
    ai = np.asarray(impedance, float)
    edges = np.concatenate([tr.twt, [tr.twt[-1] + dt]])
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (ai[:-1] + ai[1:]) * np.diff(t_log))])
    clipped = np.clip(edges, t_log[0], t_log[-1])
    integral = np.interp(clipped, t_log, cumulative)
    span = np.diff(clipped)
    with np.errstate(invalid="ignore", divide="ignore"):
        block = np.where(span > 0, np.diff(integral) / np.where(span > 0, span, 1.0), np.nan)
    return tr.twt, block


__all__ = ["TimeReflectivity", "reflectivity_in_time", "impedance_in_time", "reflectivity"]
