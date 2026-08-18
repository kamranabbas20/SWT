"""Tests for the session object and the display panels.

The session is the seam the UI and the future copilot both drive, so its
contract matters more than its internals: steps run in order, mutations
invalidate what depends on them, and **no arrays cross the boundary**.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from swt.session import SessionError, TieSession
from swt.viz import panels


@pytest.fixture
def session() -> TieSession:
    return TieSession.from_synthetic(
        seed=3, static_s=0.012, wavelet_phase_deg=30.0, n_cycle_skips=2
    )


@pytest.fixture
def tied(session: TieSession) -> TieSession:
    session.condition()
    session.build_time_depth(replacement_velocity=1900.0)
    session.calibrate()
    session.run_tie()
    return session


class TestPipelineOrder:
    def test_steps_refuse_to_run_out_of_order(self):
        empty = TieSession()
        with pytest.raises(SessionError, match="load logs"):
            empty.condition()
        with pytest.raises(SessionError, match="build a time-depth model"):
            empty.calibrate()
        with pytest.raises(SessionError, match="build a time-depth model"):
            empty.run_tie()

    def test_calibrating_without_checkshots_is_refused(self, session):
        session.checkshots = None
        session.condition()
        session.build_time_depth(replacement_velocity=1900.0)
        with pytest.raises(SessionError, match="no checkshots"):
            session.calibrate()

    def test_reconditioning_invalidates_downstream_state(self, tied):
        assert tied.result is not None
        tied.condition()
        assert tied.time_depth is None
        assert tied.drift is None
        assert tied.result is None

    def test_conditioning_twice_starts_from_the_raw_log(self, session):
        """Conditioning must not compound: two passes equal one."""
        first = session.condition()
        log_after_one = session.sonic_us_per_m.copy()
        second = session.condition()
        assert first["n_edits"] == second["n_edits"]
        assert session.sonic_us_per_m == pytest.approx(log_after_one)


class TestToolSurfaceContract:
    """Every method returns compact JSON. This is the copilot's contract."""

    def test_every_step_returns_json_safe_output(self, session):
        results = [
            session.condition(),
            session.build_time_depth(replacement_velocity=1900.0),
            session.calibrate(),
            session.run_tie(),
        ]
        for payload in results:
            encoded = json.dumps(payload)  # raises if anything is not JSON-safe
            assert "step" in payload
            # Nothing array-shaped may cross the boundary: a curve serialises
            # into thousands of characters, and the model can neither read nor
            # act on it.
            assert len(encoded) < 6000, f"payload too large ({len(encoded)} chars)"

    def test_no_step_returns_a_raw_array(self, session):
        session.condition()
        payload = session.build_time_depth(replacement_velocity=1900.0)
        for key, value in payload.items():
            assert not isinstance(value, np.ndarray), f"{key} leaked an array"

    def test_state_summarises_progress(self, tied):
        state = tied.state()
        assert state["tied"] is True
        assert state["correlation"] > 0.85
        assert state["steps_taken"] == ["load synthetic", "condition",
                                        "build time-depth", "calibrate", "tie"]
        json.dumps(state)

    def test_journal_records_every_mutation(self, tied):
        steps = [entry["step"] for entry in tied.journal]
        assert steps.count("tie") == 1
        assert "condition" in steps
        json.dumps(tied.journal, default=str)


class TestGrading:
    def test_synthetic_session_grades_itself(self, tied):
        graded = tied.grade()
        assert graded["truth_rms_error_ms"] < 3.0

    def test_error_falls_at_every_stage(self, session):
        """Each step must move the answer towards the truth, not away."""
        session.condition()
        after_integration = session.build_time_depth(replacement_velocity=1900.0)
        after_calibration = session.calibrate()
        after_tie = session.run_tie()

        assert (
            after_calibration["truth_rms_error_ms"]
            < after_integration["truth_rms_error_ms"] / 5.0
        )
        assert after_tie["truth_rms_error_ms"] < after_calibration["truth_rms_error_ms"]

    def test_real_data_has_nothing_to_grade(self, session):
        session.truth = {}
        session.condition()
        payload = session.build_time_depth(replacement_velocity=1900.0)
        assert "truth_rms_error_ms" not in payload
        assert session.grade() == {}


class TestDerivedQuantities:
    def test_backus_upscaling_reduces_impedance_variance(self, session):
        session.condition()
        raw = session.impedance()
        upscaled = session.impedance(backus_length_m=30.0)
        assert np.var(upscaled) < np.var(raw)
        assert upscaled.shape == raw.shape

    def test_velocity_is_the_reciprocal_of_slowness(self, session):
        session.condition()
        assert session.velocity() == pytest.approx(1e6 / session.sonic_us_per_m)


class TestPanels:
    """The panels must build without error on a real session.

    Rendering is not verified pixel by pixel -- these catch the shape and
    keyword errors that otherwise only appear when someone opens the app.
    """

    def test_every_panel_builds(self, tied):
        figures = [
            panels.log_tracks(tied),
            panels.log_tracks(tied, dark=True, backus_length_m=30.0),
            panels.time_depth_panel(tied),
            panels.wavelet_panel(tied.result.wavelet),
            panels.tie_display(tied),
            panels.crosscorrelation_panel(tied),
            panels.truth_panel(tied),
        ]
        for figure in figures:
            assert figure.get_axes()
            figure.clf()

    def test_tie_panels_refuse_before_a_tie(self, session):
        session.condition()
        session.build_time_depth(replacement_velocity=1900.0)
        with pytest.raises(ValueError, match="no tie"):
            panels.tie_display(session)

    def test_truth_panel_refuses_without_ground_truth(self, tied):
        tied.truth = {}
        with pytest.raises(ValueError, match="no ground truth"):
            panels.truth_panel(tied)

    def test_wiggle_shared_normalisation_keeps_relative_size(self):
        """A self-normalised residual would fill its track at any size."""
        import matplotlib.pyplot as plt

        from swt.trace import Trace

        twt = np.arange(0.0, 1.0, 0.002)
        big = Trace(twt, np.sin(2 * np.pi * 20 * twt), "big")
        small = Trace(twt, 0.05 * np.sin(2 * np.pi * 20 * twt), "small")

        figure, ax = plt.subplots()
        panels.wiggle(ax, big, norm=1.0)
        panels.wiggle(ax, small, norm=1.0)
        # Two lines drawn; the second must span ~5% of the first's width.
        spans = [
            float(np.ptp(line.get_xdata())) for line in ax.get_lines()
        ]
        assert spans[1] == pytest.approx(0.05 * spans[0], rel=0.01)
        figure.clf()


class TestUncertainty:
    def test_ensemble_runs_and_summarises(self, tied):
        payload = tied.run_uncertainty(n_members=6, seed=1)
        assert payload["n_members"] + payload["n_failed"] == 6
        json.dumps(payload)
        assert len(json.dumps(payload)) < 6000
        assert tied.state()["ensemble_members"] == tied.ensemble.n

    def test_ensemble_needs_seismic(self, session):
        session.seismic = None
        with pytest.raises(SessionError, match="no seismic"):
            session.run_uncertainty(n_members=4)

    def test_reconditioning_invalidates_the_ensemble(self, tied):
        tied.run_uncertainty(n_members=4, seed=0)
        assert tied.ensemble is not None
        tied.condition()
        assert tied.ensemble is None

    def test_uncertainty_panel_builds(self, tied):
        tied.run_uncertainty(n_members=6, seed=1)
        figure = panels.uncertainty_panel(tied)
        assert figure.get_axes()
        figure.clf()

    def test_uncertainty_panel_refuses_before_an_ensemble(self, tied):
        with pytest.raises(ValueError, match="no ensemble"):
            panels.uncertainty_panel(tied)
