"""Forward model: a synthetic earth with a known answer.

This is the test oracle, and it is the reason the rest of the package can be
trusted.  On real data there is no ground truth -- a tie is judged by a
correlation coefficient, which is exactly the number a subtly broken pipeline
will happily maximise.  Here the true time-depth curve, the true wavelet and the
true reflectivity are all known, so every stage can be asked the only question
that matters: *did it recover the right answer?*

The model deliberately includes the things that break real ties:

* a **log that starts below the seismic datum**, so the shallow interval must be
  filled by assumption or by checkshot;
* a **sonic that disagrees with the true velocity** by a smooth, depth-varying
  amount, which is the drift a checkshot calibration has to remove;
* optional **cycle skips**, **spikes** and **washout**, so log conditioning has
  something real to find;
* optional **band-limited noise** at a specified signal-to-noise ratio;
* optional **residual constant phase** on the seismic.

Nothing here is random unless a seed is given, and the seed is recorded, so
every failure is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .petro.elastic import acoustic_impedance
from .synth.convolve import synthesise
from .synth.resample import reflectivity_in_time
from .timedepth.integrate import ShallowModel, integrate_sonic
from .timedepth.model import TimeDepth
from .trace import Trace
from .units import velocity_to_slowness
from .wavelet.core import Wavelet
from .wavelet.parametric import ricker

FT_PER_M = 1.0 / 0.3048


@dataclass(frozen=True)
class SyntheticCase:
    """A complete synthetic well-tie problem, with its own answer key."""

    # what the interpreter is given
    log_depth_tvdss: np.ndarray
    log_dt_us_per_m: np.ndarray
    log_rho_g_cm3: np.ndarray
    checkshot_depth_tvdss: np.ndarray
    checkshot_twt: np.ndarray
    seismic: Trace

    # the answer key
    true_time_depth: TimeDepth
    true_wavelet: Wavelet
    true_vp_m_s: np.ndarray
    true_rho_g_cm3: np.ndarray
    true_shallow: ShallowModel
    applied_static_s: float
    seed: int
    params: dict = field(default_factory=dict)

    #: Present only for deviated cases.  The log is recorded against measured
    #: depth; ``log_depth_tvdss`` is the same samples converted through the
    #: survey, and the two differ by hundreds of metres in a deviated well.
    log_depth_md_m: np.ndarray | None = None
    deviation: object | None = None

    @property
    def log_top_tvdss(self) -> float:
        return float(self.log_depth_tvdss[0])

    def true_twt_at_log(self) -> np.ndarray:
        """True two-way time at every log sample -- the target of a T-D test."""
        return self.true_time_depth.time_at(self.log_depth_tvdss)

    def logged_time_window(self, margin_s: float = 0.0) -> tuple[float, float]:
        """The time range the log actually supports, for QC windows."""
        twt = self.true_twt_at_log()
        return float(twt[0] + margin_s), float(twt[-1] - margin_s)

    def true_shallow_for_log(self) -> ShallowModel:
        """The shallow model that reproduces the true time at the log top."""
        return ShallowModel(twt_at_log_top=float(self.true_time_depth.time_at(self.log_top_tvdss)))


def make_case(
    seed: int = 0,
    *,
    z_max: float = 3200.0,
    dz: float = 0.15,
    log_top: float = 700.0,
    log_base: float | None = None,
    replacement_velocity: float = 1900.0,
    mean_layer_thickness: float = 18.0,
    velocity_gradient: float = 0.55,
    velocity_surface: float = 1900.0,
    layer_contrast: float = 0.10,
    dt_seismic: float = 0.002,
    wavelet_frequency: float = 28.0,
    wavelet_phase_deg: float = 0.0,
    signal_to_noise: float | None = 8.0,
    drift_amplitude: float = 0.03,
    static_s: float = 0.0,
    checkshot_spacing: float = 150.0,
    checkshot_noise_s: float = 0.0,
    n_cycle_skips: int = 0,
    n_spikes: int = 0,
    sonic_noise_fraction: float = 0.005,
    anomaly_zone: tuple[float, float, float] | None = None,
    deviation=None,
) -> SyntheticCase:
    """Build a synthetic tie problem.

    Parameters
    ----------
    log_top:
        TVDSS at which the sonic starts.  The interval above it is real earth in
        the seismic but absent from the log -- the gap that a shallow model or a
        checkshot has to bridge.
    drift_amplitude:
        Peak fractional disagreement between the sonic and the true velocity,
        applied as a smooth function of depth.  ``0.03`` means the sonic is up
        to 3% off, which integrates into tens of milliseconds by 3 km -- a
        realistic drift.  Set to zero for a log that agrees with the earth.
    static_s:
        A constant time shift applied to the seismic, simulating a datum or
        processing static.  A bulk-shift search should recover ``-static_s``
        (the synthetic must move by the opposite of what was done to the data).
    signal_to_noise:
        RMS signal-to-noise ratio of band-limited additive noise.  ``None`` for
        noise-free data.
    n_cycle_skips, n_spikes:
        Number of injected sonic cycle skips (a doubling of slowness over a few
        metres) and isolated spikes.  Material for conditioning tests.
    anomaly_zone:
        ``(top_m, base_m, fraction)`` -- an interval over which the sonic reads
        wrong by ``fraction`` while the earth does not, e.g.
        ``(1800, 2100, 0.10)`` for a 300 m zone where the log is 10% slow.

        This is the failure a **warp** exists to correct, and it is a different
        shape of problem from ``drift_amplitude``. Smooth drift is removed
        completely by a checkshot drift curve, however sparse the checkshots,
        because a smooth function is what a piecewise-linear fit through a few
        points reconstructs well. A localised anomaly is not: the checkshots
        either side of it are honoured exactly, the correction is smeared across
        the whole interval between them, and a localised time error survives
        into the tie that no bulk shift and no smooth drift curve can reach.
    deviation:
        A :class:`~swt.io.deviation.Deviation` survey.  When given, the log is
        sampled along *measured depth* -- as a real log is -- and
        ``log_depth_tvdss`` holds the same samples converted through the survey.

        This is what makes the deviated-well failure reproducible. In a well with
        800 m of departure, MD and TVDSS differ by hundreds of metres, and a
        sonic integrated against MD produces a time-depth curve stretched by
        exactly that difference. Nothing downstream recognises the error for what
        it is: the drift curve absorbs part of it, the bulk shift absorbs more,
        and the tie ends up plausible and wrong.
    sonic_noise_fraction:
        Fractional random measurement noise on the sonic.  A small non-zero
        default is deliberate: a *noise-free* log is not a gentler test but a
        harder and less realistic one, because robust statistics computed on it
        degenerate -- the median absolute deviation of a perfectly clean log is
        zero, and every threshold derived from it collapses.  Real logs are
        never that clean.

    Returns
    -------
    SyntheticCase
        Everything the pipeline is given, plus everything needed to grade it.
    """
    rng = np.random.default_rng(seed)

    depth = np.arange(0.0, z_max + dz, dz)
    vp_true = _blocky_velocity(
        rng, depth, velocity_surface, velocity_gradient, mean_layer_thickness, layer_contrast
    )
    rho_true = _gardner(vp_true) * (1.0 + 0.01 * rng.standard_normal(depth.size))

    # True time-depth over the whole section, including the unlogged shallow part.
    true_td = integrate_sonic(
        depth[1:],  # skip z = 0 so the shallow model has a positive interval
        velocity_to_slowness(vp_true[1:]),
        ShallowModel(replacement_velocity=float(vp_true[1])),
    )

    seismic = _make_seismic(
        rng,
        true_td,
        vp_true[1:],
        rho_true[1:],
        dt_seismic,
        wavelet_frequency,
        wavelet_phase_deg,
        signal_to_noise,
        static_s,
    )
    wavelet = ricker(wavelet_frequency, dt_seismic)
    if wavelet_phase_deg:
        wavelet = wavelet.rotate(wavelet_phase_deg)

    # The logged interval, and the sonic the interpreter actually receives.
    base = z_max if log_base is None else min(log_base, z_max)

    log_md = None
    if deviation is None:
        logged = (depth >= log_top) & (depth <= base)
        log_depth = depth[logged]
        log_vp_true = vp_true[logged]
        log_rho = rho_true[logged]
    else:
        # A real log is sampled evenly along the borehole, not along TVD. Build
        # the MD axis first, then convert through the survey to find what depth
        # -- and therefore what rock -- each sample actually sat in.
        md_top = float(deviation.md_at_tvdss(np.array([log_top]))[0])
        md_base = float(deviation.md_at_tvdss(np.array([base]))[0])
        log_md = np.arange(md_top, md_base, dz)
        log_depth = deviation.tvdss_at_md(log_md)
        inside = (log_depth >= depth[0]) & (log_depth <= depth[-1])
        log_md, log_depth = log_md[inside], log_depth[inside]
        if log_depth.size < 2 or np.any(np.diff(log_depth) <= 0):
            raise ValueError(
                "the deviation survey does not give a strictly increasing TVDSS "
                "over the logged interval; a tie needs a well that goes down"
            )
        log_vp_true = np.interp(log_depth, depth, vp_true)
        log_rho = np.interp(log_depth, depth, rho_true)

    dt_log = velocity_to_slowness(log_vp_true) * (
        1.0 + _drift_profile(log_depth, drift_amplitude)
    )
    if anomaly_zone is not None:
        top, bottom, fraction = anomaly_zone
        if bottom <= top:
            raise ValueError("anomaly_zone base must be below its top")
        inside = (log_depth >= top) & (log_depth <= bottom)
        if not np.any(inside):
            raise ValueError(
                f"anomaly_zone {top}-{bottom} m lies outside the logged interval "
                f"{log_depth[0]:.0f}-{log_depth[-1]:.0f} m"
            )
        dt_log = dt_log * np.where(inside, 1.0 + fraction, 1.0)
    if sonic_noise_fraction:
        dt_log = dt_log * (
            1.0 + sonic_noise_fraction * rng.standard_normal(dt_log.size)
        )
    dt_log = _inject_cycle_skips(rng, log_depth, dt_log, n_cycle_skips)
    dt_log = _inject_spikes(rng, dt_log, n_spikes)

    cs_depth = np.arange(log_top, base, checkshot_spacing)
    if cs_depth.size < 2:
        raise ValueError("checkshot_spacing is too coarse for the logged interval")
    cs_twt = true_td.time_at(cs_depth)
    if checkshot_noise_s:
        cs_twt = cs_twt + checkshot_noise_s * rng.standard_normal(cs_depth.size)
        cs_twt = np.maximum.accumulate(cs_twt)  # keep the survey self-consistent

    return SyntheticCase(
        log_depth_tvdss=log_depth,
        log_dt_us_per_m=dt_log,
        log_rho_g_cm3=log_rho,
        checkshot_depth_tvdss=cs_depth,
        checkshot_twt=cs_twt,
        seismic=seismic,
        true_time_depth=true_td,
        true_wavelet=wavelet,
        true_vp_m_s=log_vp_true,
        true_rho_g_cm3=log_rho,
        true_shallow=ShallowModel(replacement_velocity=replacement_velocity),
        applied_static_s=float(static_s),
        seed=int(seed),
        params={
            "dt_seismic": dt_seismic,
            "wavelet_frequency": wavelet_frequency,
            "wavelet_phase_deg": wavelet_phase_deg,
            "signal_to_noise": signal_to_noise,
            "drift_amplitude": drift_amplitude,
            "log_top": log_top,
            "n_cycle_skips": n_cycle_skips,
            "n_spikes": n_spikes,
            "sonic_noise_fraction": sonic_noise_fraction,
            "anomaly_zone": anomaly_zone,
            "deviated": deviation is not None,
        },
        log_depth_md_m=log_md,
        deviation=deviation,
    )


# ---------------------------------------------------------------------------
# building blocks


def _blocky_velocity(
    rng: np.random.Generator,
    depth: np.ndarray,
    surface: float,
    gradient: float,
    mean_thickness: float,
    contrast: float,
) -> np.ndarray:
    """A compaction trend broken into layers of random thickness and contrast.

    Blocky rather than smooth on purpose: a smoothly varying velocity has almost
    no reflectivity, and a tie against it would be measuring nothing.
    """
    span = float(depth[-1] - depth[0])
    n_layers = max(int(span / mean_thickness), 4)
    edges = np.sort(rng.uniform(depth[0], depth[-1], n_layers - 1))
    layer_index = np.searchsorted(edges, depth)

    trend = surface + gradient * depth
    perturbation = rng.normal(0.0, contrast, n_layers + 1)[layer_index]
    velocity = trend * (1.0 + perturbation)
    return np.clip(velocity, 1500.0, 6500.0)


def _gardner(vp_m_s: np.ndarray) -> np.ndarray:
    """Gardner's relation, density in g/cm3 from velocity in m/s."""
    return 0.23 * (vp_m_s * FT_PER_M) ** 0.25


def _drift_profile(depth: np.ndarray, amplitude: float) -> np.ndarray:
    """A smooth, monotone-ish fractional error in the sonic.

    Shaped so the disagreement grows with depth and then flattens, which is what
    dispersion and invasion actually look like -- not white noise, and not a
    constant offset.
    """
    if amplitude == 0.0:
        return np.zeros_like(depth)
    normalised = (depth - depth[0]) / max(depth[-1] - depth[0], 1e-9)
    return amplitude * np.sin(0.5 * np.pi * normalised) ** 1.5


def _inject_cycle_skips(
    rng: np.random.Generator, depth: np.ndarray, dt: np.ndarray, count: int
) -> np.ndarray:
    """Double the slowness over a few metres -- the classic sonic failure."""
    if count <= 0:
        return dt
    out = dt.copy()
    spacing = float(np.median(np.diff(depth)))
    width = max(int(round(4.0 / spacing)), 3)
    for start in rng.integers(width, depth.size - width, count):
        out[start : start + width] *= 2.0
    return out


def _inject_spikes(rng: np.random.Generator, dt: np.ndarray, count: int) -> np.ndarray:
    """Isolated single-sample outliers, as from a bad reading."""
    if count <= 0:
        return dt
    out = dt.copy()
    for index in rng.integers(0, dt.size, count):
        out[index] *= rng.choice([0.4, 2.5])
    return out


def _make_seismic(
    rng: np.random.Generator,
    true_td: TimeDepth,
    vp: np.ndarray,
    rho: np.ndarray,
    dt: float,
    frequency: float,
    phase_deg: float,
    signal_to_noise: float | None,
    static_s: float,
) -> Trace:
    """Convolve the true reflectivity with a known wavelet and add noise."""
    wavelet = ricker(frequency, dt)
    if phase_deg:
        wavelet = wavelet.rotate(phase_deg)

    impedance = acoustic_impedance(vp, rho)
    rc = reflectivity_in_time(true_td.twt, impedance, dt, t_start=0.0)
    trace = synthesise(rc, wavelet, label="seismic")

    amplitude = trace.amplitude
    if signal_to_noise is not None and signal_to_noise > 0:
        # Band-limited noise: white noise shaped by the same wavelet, so the
        # noise occupies the signal's band.  White noise would be trivially
        # removable and would make every metric look better than it should.
        white = rng.standard_normal(amplitude.size)
        coloured = np.convolve(white, wavelet.samples, mode="same")
        signal_rms = float(np.sqrt(np.mean(amplitude**2)))
        noise_rms = float(np.sqrt(np.mean(coloured**2)))
        if noise_rms > 0:
            amplitude = amplitude + coloured * (signal_rms / signal_to_noise / noise_rms)

    trace = Trace(trace.twt, amplitude, "seismic", {"true_wavelet": wavelet.provenance})
    return trace.shifted(static_s) if static_s else trace
