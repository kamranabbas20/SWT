"""Streamlit front end for SWT.

Deliberately thin.  Every calculation happens in :class:`swt.session.TieSession`
and every picture in :mod:`swt.viz.panels`; this file only wires widgets to
session methods and lays out the result.  That separation is what lets the
Claude copilot drive the same tie through the same methods later, rather than
through a parallel implementation that drifts out of step with this one.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swt.io.checkshot import Checkshots, load_csv  # noqa: E402
from swt.io.las import load_las  # noqa: E402
from swt.io.segy import extract_trace, survey_info  # noqa: E402
from swt.session import SessionError, TieSession  # noqa: E402
from swt.viz import panels  # noqa: E402

st.set_page_config(page_title="SWT -- seismic-to-well tie", layout="wide")


def dark_mode() -> bool:
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# session state


def session() -> TieSession | None:
    return st.session_state.get("tie_session")


def set_session(value: TieSession) -> None:
    st.session_state["tie_session"] = value


# ---------------------------------------------------------------------------
# sidebar: data source


st.sidebar.title("SWT")
st.sidebar.caption("Seismic-to-well tie")

source = st.sidebar.radio(
    "Data source",
    ["Synthetic (known answer)", "Load files"],
    help=(
        "Synthetic cases carry their own ground truth, so the tie can be graded "
        "rather than merely scored. Real data has no truth to grade against."
    ),
)

if source.startswith("Synthetic"):
    with st.sidebar.expander("Forward model", expanded=True):
        seed = st.number_input("seed", 0, 999, 3, 1)
        static_ms = st.slider("applied static (ms)", -40.0, 40.0, 12.0, 1.0)
        phase_deg = st.slider("wavelet phase (deg)", -180.0, 180.0, 30.0, 5.0)
        frequency = st.slider("wavelet frequency (Hz)", 10.0, 60.0, 28.0, 1.0)
        snr = st.slider("signal-to-noise", 1.0, 30.0, 8.0, 0.5)
        skips = st.number_input("cycle skips to inject", 0, 10, 2, 1)
        drift = st.slider("sonic drift (fraction)", 0.0, 0.08, 0.03, 0.005)
        checkshot_spacing = st.slider("checkshot spacing (m)", 100, 1000, 150, 50)
        anomaly = st.slider(
            "localised sonic anomaly (fraction, 0 = off)", 0.0, 0.25, 0.0, 0.01,
            help=(
                "A zone where the sonic reads wrong but the earth does not. Smooth "
                "drift is removed completely by a checkshot drift curve; a localised "
                "anomaly is not, and it leaves exactly the residual a warp exists to "
                "correct. Combine with a wide checkshot spacing to see the auto-tie "
                "earn its place."
            ),
        )
        if anomaly > 0:
            anomaly_top, anomaly_base = st.slider(
                "anomaly interval (m TVDSS)", 700, 3200, (1700, 2100), 50
            )

    if st.sidebar.button("Build case", type="primary", use_container_width=True):
        set_session(
            TieSession.from_synthetic(
                seed=int(seed),
                static_s=static_ms / 1e3,
                wavelet_phase_deg=phase_deg,
                wavelet_frequency=frequency,
                signal_to_noise=snr,
                n_cycle_skips=int(skips),
                drift_amplitude=drift,
                checkshot_spacing=float(checkshot_spacing),
                anomaly_zone=(
                    (float(anomaly_top), float(anomaly_base), anomaly)
                    if anomaly > 0 else None
                ),
            )
        )
        st.rerun()
else:
    with st.sidebar.expander("Files", expanded=True):
        las_path = st.text_input("LAS file", "")
        segy_path = st.text_input("SEG-Y file", "")
        checkshot_path = st.text_input("Checkshot CSV", "")
        st.caption("Addressing the trace")
        by_line = st.checkbox("address by inline/crossline", value=False)
        if by_line:
            # Seeded from the survey's own middle rather than 0, which is never a
            # valid line number -- so the first Load cannot fail on a placeholder.
            centre = st.session_state.get("line_centre", (0, 0))
            inline = st.number_input("inline", value=int(centre[0]), step=1)
            xline = st.number_input("crossline", value=int(centre[1]), step=1)
            st.caption(
                "Run **Inspect SEG-Y** first to seed these from the survey's "
                "actual range. Coordinates are the better route where the LAS "
                "carries them -- a line number picked at random is a real "
                "location, just not the well's."
            )
            well_x = well_y = None
        else:
            well_x = st.number_input("well X", value=0.0, format="%.2f")
            well_y = st.number_input("well Y", value=0.0, format="%.2f")
            inline = xline = None
        kb = st.number_input("KB elevation (m)", value=0.0, step=1.0)
        deviation_path = st.text_input("Deviation survey CSV (MD, INC, AZI)", "")
        follow_path = st.checkbox(
            "follow the well path when extracting",
            value=False,
            disabled=not deviation_path,
            help=(
                "A deviated well is not under its wellhead. By 3 km measured "
                "depth it can be a kilometre away, which is tens of traces from "
                "where a single-location extraction looks. This extracts the "
                "trace at the well's position at each depth instead \u2014 done "
                "as a second pass, because knowing where the well is at a given "
                "time needs a time-depth model."
            ),
        )

    if segy_path and st.sidebar.button("Inspect SEG-Y", use_container_width=True):
        try:
            info = survey_info(segy_path)
            if info.inline_range and info.xline_range:
                st.session_state["line_centre"] = (
                    sum(info.inline_range) // 2, sum(info.xline_range) // 2
                )
            st.sidebar.json(info.summary())
            for warning in info.warnings():
                st.sidebar.warning(warning)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            st.sidebar.error(f"{type(exc).__name__}: {exc}")

    if st.sidebar.button("Load", type="primary", use_container_width=True):
        try:
            new = TieSession(name=Path(las_path).stem if las_path else "well")
            logs = load_las(las_path)
            new.logs = logs
            st.sidebar.json(logs.summary())
            for warning in logs.warnings():
                st.sidebar.warning(warning)
            new.sonic_us_per_m = logs.sonic_us_per_m
            new.density_g_cm3 = logs.density_g_cm3
            if deviation_path:
                from swt.io.deviation import load_csv as load_deviation

                survey = load_deviation(
                    deviation_path, kb_elevation_m=kb,
                    wellhead_x=well_x or 0.0, wellhead_y=well_y or 0.0,
                )
                new.set_deviation(survey, md_m=logs.depth_md_m)
                st.sidebar.json(survey.summary())
            else:
                # Vertical-well assumption. Wrong for any deviated well, and
                # wrong by the borehole's excess length.
                new.depth_tvdss = logs.depth_md_m - kb
            if checkshot_path:
                new.checkshots = load_csv(checkshot_path)
            if segy_path:
                new.seismic = extract_trace(
                    segy_path, x=well_x, y=well_y, inline=inline, xline=xline
                )
            st.session_state["segy_path"] = segy_path
            st.session_state["follow_path"] = bool(follow_path and deviation_path)
            set_session(new)
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            st.sidebar.error(f"{type(exc).__name__}: {exc}")


current = session()
if current is None:
    st.title("Seismic-to-well tie")
    st.markdown(
        """
Build a synthetic case from the sidebar to get started, or load a LAS and a SEG-Y.

A synthetic case knows its own answer, so the tie can be **graded** rather than
merely scored. That distinction is the point of this tool: a pipeline with a sign
error or a phase/time confusion still reports a high correlation, because a high
correlation is what it optimised for.
"""
    )
    st.stop()


# ---------------------------------------------------------------------------
# sidebar: processing controls


st.sidebar.divider()
st.sidebar.subheader("1. Condition")
despike_threshold = st.sidebar.slider("despike threshold (sigma)", 2.0, 10.0, 4.0, 0.5)
max_gap = st.sidebar.slider("max gap to fill (m)", 0.0, 30.0, 5.0, 1.0)

st.sidebar.subheader("2. Time-depth")
shallow_mode = st.sidebar.radio(
    "Above the log top",
    ["Replacement velocity", "Fix TWT at log top"],
    help=(
        "The log starts below the seismic datum and the gap has to be bridged. "
        "This is where most bad ties are born: an error here shifts the whole "
        "synthetic rigidly."
    ),
)
if shallow_mode.startswith("Replacement"):
    replacement_velocity = st.sidebar.number_input(
        "replacement velocity (m/s)", 1000.0, 5000.0, 1900.0, 50.0
    )
    twt_at_top = None
else:
    replacement_velocity = None
    twt_at_top = st.sidebar.number_input("TWT at log top (ms)", 0.0, 5000.0, 700.0, 5.0) / 1e3

st.sidebar.subheader("3. Calibrate")
use_checkshots = st.sidebar.checkbox(
    "calibrate to checkshots", value=current.checkshots is not None,
    disabled=current.checkshots is None,
)
knot_spacing = st.sidebar.slider(
    "drift knot spacing (m, 0 = every checkshot)", 0, 1000, 0, 50,
    help=(
        "Honouring every checkshot exactly injects its noise into the velocity "
        "field. Widening the knots smooths the correction -- and is the fix when "
        "calibration reports an inverted time-depth."
    ),
)

st.sidebar.subheader("4. Tie")
wavelet_length = st.sidebar.slider("wavelet length (ms)", 40, 300, 128, 4)
max_shift = st.sidebar.slider("max bulk shift (ms)", 10, 200, 60, 5)
iterations = st.sidebar.slider("max iterations", 1, 6, 3, 1)
deterministic = st.sidebar.checkbox("re-estimate wavelet by least squares", value=True)
backus_length = st.sidebar.slider("Backus upscaling (m, 0 = off)", 0, 100, 0, 5)

st.sidebar.subheader("5. Auto-tie (warp)")
use_warp = st.sidebar.checkbox(
    "stretch and squeeze",
    value=False,
    help=(
        "Runs after the bulk shift and phase scan, never instead of them. The "
        "warp is kept only if the velocity change it implies is plausible, it is "
        "not merely clipped to look plausible, and it earns enough correlation to "
        "justify a degree of freedom per sample."
    ),
)
velocity_limit = st.sidebar.slider(
    "max velocity change vs sonic (%)", 2.0, 40.0, 15.0, 1.0, disabled=not use_warp,
    help=(
        "The one number that matters. It sets both the guardrail's acceptance "
        "test and the solver's strain limit, so the solver cannot search a region "
        "the guardrail would reject."
    ),
)
max_warp_shift = st.sidebar.slider(
    "max warp shift (ms)", 10, 150, 60, 5, disabled=not use_warp
)
saturation_limit = st.sidebar.slider(
    "max fraction pinned at the limit", 0.0, 1.0, 0.5, 0.05, disabled=not use_warp,
    help=(
        "A warp pinned at its strain limit over most of the log has been clipped, "
        "not solved. It passes the velocity check precisely because it was clipped "
        "to a passing value, so this is a separate test."
    ),
)

st.sidebar.subheader("6. Uncertainty")
use_uq = st.sidebar.checkbox(
    "run ensemble",
    value=False,
    help=(
        "Re-runs the whole tie over sampled interpreter choices \u2014 despike "
        "threshold, drift knots, wavelet length, upscaling \u2014 and reports a "
        "corridor instead of a curve. Slow: every member is a full pipeline run."
    ),
)
n_members = st.sidebar.slider("ensemble members", 4, 48, 16, 4, disabled=not use_uq)
probe_cycles = st.sidebar.checkbox(
    "probe cycle ambiguity", value=True, disabled=not use_uq,
    help=(
        "Start members from time-depth models offset by up to a wavelet period, "
        "so that if two alignments a cycle apart are both defensible the ensemble "
        "finds both. Without it the corridor measures precision around whichever "
        "loop the first run happened to reach."
    ),
)

run = st.sidebar.button("Run pipeline", type="primary", use_container_width=True)

if run:
    try:
        with st.spinner("Tying..."):
            current.condition(despike_threshold=despike_threshold, max_gap_m=max_gap)
            current.build_time_depth(
                replacement_velocity=replacement_velocity, twt_at_log_top=twt_at_top
            )
            if use_checkshots and current.checkshots is not None:
                current.calibrate(knot_spacing_m=knot_spacing or None)
            tie_options = dict(
                wavelet_length_s=wavelet_length / 1e3,
                max_shift_s=max_shift / 1e3,
                max_iterations=iterations,
                deterministic=deterministic,
            )
            # A deviated well needs the trace that follows the borehole, and
            # finding it needs a time-depth model -- so the path is followed on a
            # second pass, once the first tie has produced one.
            if (
                st.session_state.get("follow_path")
                and current.deviation is not None
                and st.session_state.get("segy_path")
            ):
                current.run_tie(**tie_options)
                current.reextract_along_path(st.session_state["segy_path"])

            if use_warp:
                current.run_auto_tie(
                    velocity_limit_percent=velocity_limit,
                    max_warp_shift_s=max_warp_shift / 1e3,
                    max_saturated_fraction=saturation_limit,
                    **tie_options,
                )
            else:
                current.run_tie(**tie_options)

            if use_uq:
                bar = st.progress(0.0, text="Running ensemble...")
                current.run_uncertainty(
                    n_members=int(n_members),
                    probe_cycle_ambiguity=probe_cycles,
                    progress=lambda done, total: bar.progress(
                        done / total, text=f"Ensemble member {done}/{total}"
                    ),
                )
                bar.empty()
        st.session_state["error"] = None
    except (SessionError, ValueError) as exc:
        st.session_state["error"] = f"{type(exc).__name__}: {exc}"

if st.session_state.get("error"):
    st.error(st.session_state["error"])


# ---------------------------------------------------------------------------
# main layout


st.title(current.name)

state = current.state()
columns = st.columns(5)
columns[0].metric("correlation", f"{state['correlation']:.3f}" if state["correlation"] else "--")
if current.result:
    columns[1].metric("NRMS", f"{current.result.metrics.nrms_percent:.1f}%")
    columns[2].metric("bulk shift", f"{current.result.total_shift_s * 1e3:+.1f} ms")
    columns[3].metric("phase", f"{current.result.phase_deg:+.0f}°")
graded = current.grade()
if graded:
    columns[4].metric(
        "error vs truth", f"{graded['truth_rms_error_ms']:.2f} ms",
        help="rms difference from the forward model's actual time-depth curve",
    )

tab_names = ["Tie", "Logs", "Time-depth", "Wavelet", "QC"]
if current.auto is not None and current.auto.warp is not None:
    tab_names.append("Warp")
if current.ensemble is not None and current.ensemble.members:
    tab_names.append("Uncertainty")
tab_names.append("Copilot")
tab_names.append("Journal")
if current.truth:
    tab_names.append("Truth")
tabs = dict(zip(tab_names, st.tabs(tab_names)))

dark = dark_mode()

with tabs["Tie"]:
    if current.result is None:
        st.info("Run the pipeline from the sidebar.")
    else:
        window = current.result.window_s
        zoom = st.slider(
            "display window (s)",
            float(window[0]), float(window[1]),
            (float(window[0]), float(window[1])), 0.01,
        )
        st.pyplot(panels.tie_display(current, dark=dark, zoom=zoom), use_container_width=True)

        verdict = current.result.significance.verdict()
        (st.success if current.result.significance.is_significant else st.warning)(verdict)

        if current.result.shift_search and current.result.shift_search.is_ambiguous:
            st.warning(
                "The cross-correlation has a second peak of comparable height. "
                "This tie may sit a whole cycle away from the right one -- check "
                "the QC tab before accepting it."
            )

with tabs["Logs"]:
    if current.sonic_us_per_m is None:
        st.info("No logs loaded.")
    else:
        st.pyplot(
            panels.log_tracks(current, dark=dark, backus_length_m=backus_length),
            use_container_width=True,
        )
        if current.cycle_skips:
            st.warning(f"{len(current.cycle_skips)} cycle skip(s) detected:")
            st.dataframe(
                [skip.summary() for skip in current.cycle_skips],
                use_container_width=True, hide_index=True,
            )
        if current.conditioning and current.conditioning.edits:
            with st.expander(f"{len(current.conditioning.edits)} conditioning edit(s)"):
                st.dataframe(
                    [edit.summary() for edit in current.conditioning.edits],
                    use_container_width=True, hide_index=True,
                )

with tabs["Time-depth"]:
    if current.time_depth is None:
        st.info("Build a time-depth model from the sidebar.")
    else:
        st.pyplot(panels.time_depth_panel(current, dark=dark), use_container_width=True)
        st.caption(
            "Read the drift panel first. A smooth ramp is ordinary dispersion; a "
            "**step** is a log problem at that depth; a change of slope localises "
            "where the sonic stops agreeing with the seismic."
        )
        if current.result and abs(current.result.total_shift_s) > 1e-4:
            st.info(
                f"Drift is shown against the **current** time-depth model, which "
                f"now carries the tie's {current.result.total_shift_s * 1e3:+.1f} ms "
                "bulk shift — so a uniform offset of about that size is expected "
                "here and is not a calibration failure. It says the static lives "
                "in the seismic (a datum or processing static the checkshots never "
                "saw), not in the well. A drift that is *sloped* or *stepped* after "
                "tying is the one to worry about."
            )
        if current.drift:
            st.json(current.drift.summary())
        if current.checkshots:
            implausible = current.checkshots.implausible_intervals()
            if implausible:
                st.warning("Checkshot intervals with implausible velocities:")
                st.dataframe(implausible, use_container_width=True, hide_index=True)

with tabs["Wavelet"]:
    if current.result is None:
        st.info("Run the pipeline to estimate a wavelet.")
    else:
        st.pyplot(panels.wavelet_panel(current.result.wavelet, dark=dark),
                  use_container_width=True)
        st.json(current.result.wavelet.summary())
        if current.truth:
            st.caption(
                f"The forward model used a "
                f"{current.truth['wavelet_frequency_hz']:.0f} Hz wavelet at "
                f"{current.truth['wavelet_phase_deg']:+.0f}°."
            )

with tabs["QC"]:
    if current.result is None:
        st.info("Run the pipeline to compute QC.")
    else:
        st.pyplot(panels.crosscorrelation_panel(current, dark=dark),
                  use_container_width=True)
        left, right = st.columns(2)
        left.subheader("Metrics")
        left.json(current.result.metrics.summary())
        right.subheader("Significance")
        right.json(current.result.significance.summary())
        st.caption(
            "A correlation coefficient means nothing without the window and "
            "bandwidth it was measured over. The significance block gives the "
            "chance level for *this* window -- correlations below it are "
            "indistinguishable from noise."
        )

if "Warp" in tabs:
    with tabs["Warp"]:
        auto = current.auto
        st.pyplot(panels.warp_panel(current, dark=dark), use_container_width=True)
        if auto.warp_accepted:
            st.success(auto.verdict())
        else:
            st.error(auto.rejection_reason or auto.verdict())
        st.caption(
            "A warp has one free parameter per sample, so it can align almost "
            "anything with almost anything and report a fine correlation. "
            "Correlation therefore stops being evidence here. What remains is the "
            "velocity change the warp claims against the sonic \u2014 the third "
            "panel \u2014 and whether the warp reached that claim or merely got "
            "clipped to it \u2014 the second."
        )
        left, right = st.columns(2)
        left.subheader("Warp")
        left.json(auto.warp.summary())
        right.subheader("Guardrail")
        right.json(auto.guardrail.summary())

if "Uncertainty" in tabs:
    with tabs["Uncertainty"]:
        ensemble = current.ensemble
        st.pyplot(panels.uncertainty_panel(current, dark=dark), use_container_width=True)
        (st.warning if ensemble.is_multimodal else st.success)(ensemble.verdict())

        st.subheader("Per-horizon tolerance")
        st.caption(
            "This is the deliverable \u2014 what whoever does the depth conversion "
            "needs, and what a single time-depth curve cannot give them."
        )
        depths = current.depth_tvdss
        marks = np.linspace(depths[0], depths[-1], 6)
        st.dataframe(ensemble.per_horizon(marks), use_container_width=True, hide_index=True)

        st.json(ensemble.summary())
        if ensemble.failures:
            with st.expander(f"{len(ensemble.failures)} member(s) failed"):
                st.dataframe(ensemble.failures, use_container_width=True, hide_index=True)
        st.caption(
            "The ensemble varies processing **choices**, so it measures how much the "
            "answer depends on decisions nobody can make uniquely. It does not vary "
            "the seismic's noise, the wavelet's own estimation error, or the sample "
            "interval, and so cannot see error that every member shares \u2014 which "
            "is why the corridor carries a resolution floor and why this is a "
            "precision estimate, not an accuracy one."
        )

if "Copilot" in tabs:
    with tabs["Copilot"]:
        st.caption(
            "Claude, wired to this session through the same methods the sidebar "
            "calls. It reads the engine's own QC \u2014 including the significance "
            "verdict and any warp rejection \u2014 and its job is diagnosis, not "
            "restating numbers you can already see."
        )

        allow_changes = st.toggle(
            "let the copilot change the tie",
            value=False,
            help=(
                "Off, it can look but not touch: any call that would change the "
                "tie is refused and it must explain what it wanted to do. On, it "
                "can run the pipeline. The engine's own guardrails still apply "
                "either way \u2014 it cannot exceed the velocity limit or invert "
                "the time-depth."
            ),
        )

        if "copilot_history" not in st.session_state:
            st.session_state["copilot_history"] = []

        for entry in st.session_state["copilot_history"]:
            with st.chat_message(entry["role"]):
                st.markdown(entry["text"])
                if entry.get("tools"):
                    st.caption("tools: " + ", ".join(entry["tools"]))

        question = st.chat_input("Ask about this tie...")
        report = st.button("Write the tie report")

        if question or report:
            try:
                from swt.copilot.agent import Copilot

                copilot = st.session_state.get("copilot")
                if copilot is None or copilot.session is not current:
                    copilot = Copilot(current, allow_changes=allow_changes)
                    st.session_state["copilot"] = copilot
                copilot.permit.allow = allow_changes

                prompt = question or "Write the tie report."
                st.session_state["copilot_history"].append(
                    {"role": "user", "text": prompt}
                )
                with st.spinner("Thinking..."):
                    turn = copilot.write_report() if report else copilot.ask(question)
                st.session_state["copilot_history"].append({
                    "role": "assistant",
                    "text": turn.text or "_(no text returned)_",
                    "tools": [c["name"] for c in turn.tool_calls],
                })
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                text = str(exc).lower()
                missing_credentials = (
                    "authentication" in text or "api_key" in text or "api key" in text
                )
                if missing_credentials:
                    st.info(
                        "The copilot needs Anthropic credentials. Set "
                        "`ANTHROPIC_API_KEY` in the environment, or run "
                        "`ant auth login`, then reload. Everything else in SWT "
                        "works without them."
                    )
                else:
                    st.error(f"{type(exc).__name__}: {exc}")

        denials = [
            d for d in (st.session_state.get("copilot").permit_log()
                        if st.session_state.get("copilot") else [])
            if not d["allowed"]
        ]
        if denials:
            with st.expander(f"{len(denials)} change(s) the copilot was stopped from making"):
                st.dataframe(denials, use_container_width=True, hide_index=True)
                st.caption(
                    "A refused call changes nothing, so the journal never records "
                    "it \u2014 but what the copilot *wanted* to do is worth keeping."
                )

with tabs["Journal"]:
    st.subheader("What was done")
    for entry in current.journal:
        with st.expander(entry["step"]):
            st.json({k: v for k, v in entry.items() if k != "step"})
    st.download_button(
        "Download journal (JSON)",
        data=__import__("json").dumps(current.journal, indent=2, default=str),
        file_name=f"{current.name.replace(' ', '_')}_journal.json",
        mime="application/json",
    )

if current.truth:
    with tabs["Truth"]:
        if current.time_depth is None:
            st.info("Build a time-depth model to grade it.")
        else:
            st.pyplot(panels.truth_panel(current, dark=dark), use_container_width=True)
            st.caption(
                "This panel does not exist on real data. It is the whole argument "
                "for developing against a forward model: it is the only place the "
                "question *is the answer right?* can be asked at all."
            )
