"""Matplotlib panels for well-tie displays.

Deliberately UI-agnostic: these build figures and return them, so the Streamlit
app, a notebook, and a PDF report all draw the same picture.  Nothing here
touches Streamlit.

The displays follow well-tie convention rather than general plotting taste --
depth and time increase *downward*, seismic is drawn as variable-area wiggle
with filled positive lobes, and the synthetic sits immediately beside the
seismic rather than overlaid on it, because judging a tie is a matter of reading
whether events line up, not whether curves overlap.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..trace import Trace
from ..wavelet.core import Wavelet

#: Colours chosen to stay legible on both light and dark backgrounds.
SEISMIC = "#1f77b4"
SYNTHETIC = "#d62728"
NEUTRAL = "#7f7f7f"
ACCENT = "#2ca02c"
WARNING = "#ff7f0e"


def _style(dark: bool) -> dict:
    if dark:
        return {"fg": "#e6e6e6", "bg": "#0e1117", "grid": "#333844"}
    return {"fg": "#222222", "bg": "#ffffff", "grid": "#dddddd"}


def _prepare(fig, axes, dark: bool):
    colours = _style(dark)
    fig.patch.set_facecolor(colours["bg"])
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(colours["bg"])
        ax.tick_params(colors=colours["fg"], labelsize=8)
        ax.grid(True, color=colours["grid"], linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(colours["grid"])
        ax.xaxis.label.set_color(colours["fg"])
        ax.yaxis.label.set_color(colours["fg"])
        ax.title.set_color(colours["fg"])
    return fig


def wiggle(ax, trace: Trace, x_centre: float = 0.0, scale: float = 1.0,
           colour: str = SEISMIC, repeats: int = 1, spacing: float = 1.0,
           norm: float | None = None) -> None:
    """Variable-area wiggle: the line, with positive lobes filled.

    ``repeats`` draws the same trace several times side by side.  That is not
    decoration -- a single trace is hard to read for continuity, and the
    repeated display is how an interpreter actually judges whether a synthetic
    event lines up with a seismic event.

    ``norm`` sets the amplitude each trace is divided by.  Pass a **shared**
    value when several traces are displayed together: a self-normalised residual
    panel fills the track whether the residual is 5% of the data or 90% of it,
    which makes the most important panel on the display the least informative.
    """
    amplitude = trace.amplitude
    peak = float(norm) if norm else float(np.max(np.abs(amplitude)))
    if peak == 0:
        peak = 1.0
    normalised = amplitude / peak * scale

    for k in range(repeats):
        offset = x_centre + k * spacing
        ax.plot(normalised + offset, trace.twt, color=colour, linewidth=0.7)
        ax.fill_betweenx(
            trace.twt, offset, normalised + offset,
            where=(normalised > 0), color=colour, alpha=0.55, linewidth=0,
        )


def log_tracks(session, dark: bool = False, backus_length_m: float = 0.0):
    """Sonic, density and impedance against depth, with conditioning marked."""
    depth = session.depth_tvdss
    fig, axes = plt.subplots(1, 4, figsize=(11, 8), sharey=True)

    velocity = session.velocity()
    axes[0].plot(session.sonic_us_per_m, depth, color=SEISMIC, linewidth=0.6)
    axes[0].set_xlabel("slowness (us/m)")
    axes[0].set_ylabel("TVDSS (m)")

    axes[1].plot(velocity, depth, color=ACCENT, linewidth=0.6)
    axes[1].set_xlabel("velocity (m/s)")

    axes[2].plot(session.density_g_cm3, depth, color=WARNING, linewidth=0.6)
    axes[2].set_xlabel("density (g/cm3)")

    impedance = session.impedance()
    axes[3].plot(impedance, depth, color=NEUTRAL, linewidth=0.6, label="log scale")
    if backus_length_m > 0:
        axes[3].plot(
            session.impedance(backus_length_m), depth,
            color=SYNTHETIC, linewidth=1.0, label=f"Backus {backus_length_m:.0f} m",
        )
        axes[3].legend(fontsize=7, loc="lower right")
    axes[3].set_xlabel("impedance")

    # Conditioning edits and cycle skips, shaded across every track.
    for skip in session.cycle_skips:
        for ax in axes:
            ax.axhspan(skip.start_depth, skip.end_depth, color=SYNTHETIC, alpha=0.25)
    if session.conditioning:
        for edit in session.conditioning.edits:
            if edit.kind in ("gap_left", "gap_filled"):
                for ax in axes:
                    ax.axhspan(edit.start_depth, edit.end_depth, color=WARNING, alpha=0.2)

    axes[0].invert_yaxis()
    fig.suptitle(
        f"{session.name} -- logs"
        + (f"  ({len(session.cycle_skips)} cycle skip(s) shaded red)"
           if session.cycle_skips else ""),
        color=_style(dark)["fg"], fontsize=10,
    )
    fig.tight_layout()
    return _prepare(fig, axes, dark)


def time_depth_panel(session, dark: bool = False):
    """The time-depth curve, the drift at the checkshots, and interval velocity.

    The drift plot is the diagnostic worth reading first: a smooth ramp is
    ordinary dispersion, a *step* is a log problem at that depth, and a change
    of slope localises where the sonic stops agreeing with the seismic.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 6), sharey=True)
    td = session.time_depth

    axes[0].plot(td.twt * 1e3, td.depth_tvdss, color=SEISMIC, linewidth=1.0)
    if session.checkshots is not None:
        axes[0].plot(
            session.checkshots.twt_s * 1e3, session.checkshots.depth_tvdss_m,
            "o", color=SYNTHETIC, markersize=4, label="checkshots",
        )
        axes[0].legend(fontsize=7)
    axes[0].set_xlabel("TWT (ms)")
    axes[0].set_ylabel("TVDSS (m)")
    axes[0].set_title("time-depth", fontsize=9)

    observed = session.observed_drift()
    if observed is not None:
        cs_depth, drift = observed
        axes[1].plot(drift * 1e3, cs_depth, "o-", color=WARNING, markersize=4)
        axes[1].axvline(0.0, color=NEUTRAL, linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("drift (ms)")
    axes[1].set_title("drift at checkshots", fontsize=9)

    midpoints, interval = td.interval_velocity()
    step = max(len(midpoints) // 2000, 1)
    axes[2].plot(interval[::step], midpoints[::step], color=ACCENT, linewidth=0.5)
    if session.checkshots is not None:
        cs_mid, cs_velocity = session.checkshots.interval_velocity()
        axes[2].plot(cs_velocity, cs_mid, "o-", color=SYNTHETIC, markersize=3,
                     linewidth=1.0, label="checkshot")
        axes[2].legend(fontsize=7)
    axes[2].set_xlabel("interval velocity (m/s)")
    axes[2].set_title("interval velocity", fontsize=9)

    axes[0].invert_yaxis()
    fig.tight_layout()
    return _prepare(fig, axes, dark)


def wavelet_panel(wavelet: Wavelet, dark: bool = False):
    """The wavelet and its amplitude spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    time_ms = wavelet.time() * 1e3
    axes[0].plot(time_ms, wavelet.samples, color=SYNTHETIC, linewidth=1.2)
    axes[0].fill_between(time_ms, 0, wavelet.samples,
                         where=(wavelet.samples > 0), color=SYNTHETIC, alpha=0.4)
    axes[0].axvline(0.0, color=NEUTRAL, linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("time (ms)")
    axes[0].set_title(
        f"{wavelet.provenance}  |  phase {wavelet.estimated_phase():+.0f} deg", fontsize=8
    )

    frequency, spectrum = wavelet.amplitude_spectrum()
    peak = float(np.max(spectrum))
    axes[1].plot(frequency, spectrum / peak, color=SEISMIC, linewidth=1.2)
    low, high = wavelet.bandwidth()
    axes[1].axvspan(low, high, color=ACCENT, alpha=0.15)
    axes[1].axhline(10 ** (-6.0 / 20.0), color=NEUTRAL, linewidth=0.8, linestyle="--")
    axes[1].set_xlim(0, min(float(frequency[-1]), high * 3))
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_title(
        f"dominant {wavelet.dominant_frequency():.1f} Hz  |  "
        f"-6 dB band {low:.0f}-{high:.0f} Hz", fontsize=8,
    )

    fig.tight_layout()
    return _prepare(fig, axes, dark)


def tie_display(session, dark: bool = False, repeats: int = 4, zoom: tuple | None = None):
    """The tie itself: reflectivity, synthetic and seismic side by side in time.

    This is the display a tie is actually judged on.  The synthetic sits beside
    the seismic rather than on top of it -- the question is whether events line
    up, and overlaying two wiggles makes that harder to see, not easier.
    """
    result = session.result
    if result is None:
        raise ValueError("no tie has been run yet")

    t0, t1 = zoom if zoom else result.window_s
    fig, axes = plt.subplots(1, 4, figsize=(11, 8), sharey=True)

    reflectivity = result.reflectivity
    axes[0].hlines(reflectivity.twt, 0, reflectivity.rc, color=NEUTRAL, linewidth=0.6)
    axes[0].axvline(0, color=NEUTRAL, linewidth=0.6)
    axes[0].set_xlabel("reflectivity")
    axes[0].set_ylabel("TWT (s)")

    synthetic = result.synthetic.window(t0, t1)
    seismic = session.seismic.window(t0, t1)

    # Residual, on the seismic's axis and at the tie's own optimal scaling.
    predicted = synthetic.resample_to(seismic)
    scalar = result.metrics.optimal_scalar
    scaled_synthetic = Trace(seismic.twt, scalar * predicted.amplitude, "synthetic")
    residual = Trace(seismic.twt, seismic.amplitude - scaled_synthetic.amplitude, "residual")

    # One normalisation for all three panels, so the residual's size is read
    # against the data rather than against itself.
    norm = float(np.max(np.abs(seismic.amplitude))) or 1.0

    wiggle(axes[1], scaled_synthetic, colour=SYNTHETIC, repeats=repeats,
           spacing=1.2, scale=0.55, norm=norm)
    axes[1].set_xlabel("synthetic")

    wiggle(axes[2], seismic, colour=SEISMIC, repeats=repeats, spacing=1.2,
           scale=0.55, norm=norm)
    axes[2].set_xlabel("seismic")

    wiggle(axes[3], residual, colour=WARNING, repeats=repeats, spacing=1.2,
           scale=0.55, norm=norm)
    residual_rms = float(np.sqrt(np.mean(residual.amplitude**2)))
    seismic_rms = float(np.sqrt(np.mean(seismic.amplitude**2)))
    axes[3].set_xlabel(
        f"residual ({100.0 * residual_rms / seismic_rms:.0f}% of data rms)"
        if seismic_rms else "residual"
    )

    for ax in axes:
        ax.set_ylim(t1, t0)
        ax.set_xticks([])

    metrics = result.metrics
    fig.suptitle(
        f"correlation {metrics.correlation:.3f}   NRMS {metrics.nrms_percent:.1f}%   "
        f"PEP {metrics.pep:.3f}   shift {result.total_shift_s * 1e3:+.1f} ms   "
        f"phase {result.phase_deg:+.0f} deg",
        color=_style(dark)["fg"], fontsize=10,
    )
    fig.tight_layout()
    return _prepare(fig, axes, dark)


def crosscorrelation_panel(session, dark: bool = False):
    """The cross-correlation function and the constant-phase scan.

    Both are ambiguity displays.  A second peak in the cross-correlation about
    one dominant period from the first is a cycle-skip risk: the tie could
    plausibly sit a whole loop away, and a single reported number hides that.
    """
    result = session.result
    if result is None:
        raise ValueError("no tie has been run yet")

    from ..qc.metrics import cross_correlation

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    observed = session.seismic.window(*result.window_s)
    predicted = result.synthetic.resample_to(observed)
    lags, values = cross_correlation(observed.amplitude, predicted.amplitude,
                                     observed.dt, 0.100)

    axes[0].plot(lags * 1e3, values, color=SEISMIC, linewidth=1.2)
    axes[0].axvline(0.0, color=ACCENT, linewidth=1.0)
    best = int(np.argmax(values))
    axes[0].plot(lags[best] * 1e3, values[best], "o", color=SYNTHETIC, markersize=6)
    axes[0].set_xlabel("lag (ms)")
    axes[0].set_ylabel("correlation")
    axes[0].set_title(f"residual shift {lags[best] * 1e3:+.1f} ms", fontsize=8)

    if result.shift_search is not None and result.shift_search.is_ambiguous:
        axes[0].axvline(
            result.shift_search.runner_up_shift_s * 1e3,
            color=WARNING, linewidth=1.0, linestyle="--",
        )
        axes[0].set_title(
            f"AMBIGUOUS: second peak at "
            f"{result.shift_search.runner_up_shift_s * 1e3:+.0f} ms "
            f"(r={result.shift_search.runner_up_correlation:.2f}) -- cycle-skip risk",
            fontsize=7, color=WARNING,
        )

    scan = result.phase_scan
    if scan is not None:
        axes[1].plot(scan.phases_deg, scan.correlations, color=SEISMIC, linewidth=1.2)
        axes[1].plot(scan.best_phase_deg, scan.best_correlation, "o",
                     color=SYNTHETIC, markersize=6)
    axes[1].axvline(result.phase_deg, color=SYNTHETIC, linewidth=1.0, linestyle="--",
                    label=f"total {result.phase_deg:+.0f} deg")
    axes[1].set_xlabel("constant phase (deg)")
    axes[1].set_ylabel("correlation")
    axes[1].set_title("constant-phase scan", fontsize=8)
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    return _prepare(fig, axes, dark)


def warp_panel(session, dark: bool = False):
    """The warp, and the two ways it can be inadmissible.

    Three panels, because a warp cannot be judged by its shape alone:

    * the **shift field** -- what the warp does, coarse pass beside fine;
    * the **strain** against its limit -- where the warp is pinned, which is
      where it has been clipped rather than solved;
    * the **implied velocity change** against the sonic, which is the warp's
      actual geological claim and the only panel that can say it is wrong.

    A warp that looks smooth and improves the correlation can still be pinned
    at its limit across most of the log, which means the correction genuinely
    needed is larger than the permitted velocity change. That is visible here
    and nowhere else.
    """
    auto = getattr(session, "auto", None)
    if auto is None or auto.warp is None:
        raise ValueError("no warp has been computed yet")

    warp, guardrail = auto.warp, auto.guardrail
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    axes[0].plot(warp.shift_s * 1e3, warp.twt, color=SYNTHETIC, linewidth=1.2, label="fine")
    if auto.coarse_warp is not None:
        axes[0].plot(
            auto.coarse_warp.shift_s * 1e3, auto.coarse_warp.twt,
            color=NEUTRAL, linewidth=1.0, linestyle="--", label="coarse (envelope)",
        )
    axes[0].axvline(0.0, color=NEUTRAL, linewidth=0.8)
    axes[0].set_xlabel("shift (ms)")
    axes[0].set_ylabel("TWT (s)")
    axes[0].set_title("warp shift field", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].invert_yaxis()

    strain = warp.strain() * 100.0
    limit = warp.strain_limit * 100.0
    axes[1].plot(strain, warp.twt, color=SEISMIC, linewidth=0.8)
    for sign in (-1, 1):
        axes[1].axvline(sign * limit, color=SYNTHETIC, linewidth=1.0, linestyle="--")
    pinned = np.abs(strain) >= 0.95 * limit
    if np.any(pinned):
        axes[1].plot(strain[pinned], warp.twt[pinned], ".", color=WARNING, markersize=2)
    axes[1].set_xlabel("strain (%)")
    axes[1].set_title(
        f"strain vs limit -- {warp.saturated_fraction():.0%} pinned", fontsize=9,
        color=WARNING if warp.saturated_fraction() > 0.5 else _style(dark)["fg"],
    )
    axes[1].invert_yaxis()

    axes[2].plot(guardrail.change_percent, guardrail.depth, color=ACCENT, linewidth=0.6)
    for sign in (-1, 1):
        axes[2].axvline(sign * guardrail.limit_percent, color=SYNTHETIC,
                        linewidth=1.0, linestyle="--")
    for violation in guardrail.worst(8):
        axes[2].axhspan(violation.start_depth, violation.end_depth,
                        color=SYNTHETIC, alpha=0.25)
    axes[2].set_xlabel("implied velocity change vs sonic (%)")
    axes[2].set_ylabel("TVDSS (m)")
    axes[2].set_title(
        f"velocity claim -- max {guardrail.max_change_percent:.1f}%", fontsize=9,
        color=_style(dark)["fg"] if guardrail.passed else SYNTHETIC,
    )
    axes[2].invert_yaxis()

    headline = (
        auto.verdict() if auto.warp_accepted
        else f"WARP REJECTED -- {warp.saturated_fraction():.0%} pinned at the strain limit"
        if warp.saturated_fraction() > 0.5
        else "WARP REJECTED -- see the tie's verdict for why"
    )
    fig.suptitle(headline, color=ACCENT if auto.warp_accepted else SYNTHETIC, fontsize=9)
    fig.tight_layout()
    return _prepare(fig, axes, dark)


def truth_panel(session, dark: bool = False):
    """Error against ground truth -- synthetic cases only.

    On real data this panel does not exist, which is the whole argument for
    developing against a forward model: it is the only place the question "is
    the answer right?" can be asked at all.
    """
    if not session.truth or session.time_depth is None:
        raise ValueError("this session has no ground truth")

    fig, ax = plt.subplots(figsize=(5, 6))
    error = (session.time_depth.twt - session.truth["twt_at_log"]) * 1e3
    ax.plot(error, session.time_depth.depth_tvdss, color=SYNTHETIC, linewidth=0.8)
    ax.axvline(0.0, color=ACCENT, linewidth=1.0)
    ax.set_xlabel("time-depth error vs truth (ms)")
    ax.set_ylabel("TVDSS (m)")
    ax.set_title(
        f"rms {np.sqrt(np.mean(error ** 2)):.2f} ms   max {np.max(np.abs(error)):.2f} ms",
        fontsize=9,
    )
    ax.invert_yaxis()
    fig.tight_layout()
    return _prepare(fig, ax, dark)
