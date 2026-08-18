"""Deviated wells: the survey, the path-following extraction, and the tie.

A deviated well breaks a tie in two independent ways, and both are silent.

The log is recorded against **measured depth** while the seismic is indexed by
**true vertical depth**. Integrating a sonic against MD stretches the time-depth
curve by exactly the difference, and nothing downstream recognises it: the drift
curve absorbs part, the bulk shift absorbs more, and the result is a plausible
wrong answer.

And the well is not under its wellhead. By 3 km measured depth it can be a
kilometre away, which is tens of traces from where a single-location extraction
looks -- so the synthetic is compared against rock the well never entered.
"""

from __future__ import annotations

import numpy as np
import pytest

from swt.forward import make_case
from swt.io import deviation as dev
from swt.io.segy import extract_along_path, extract_trace
from swt.timedepth.calibrate import calibrate_to_checkshots
from swt.timedepth.integrate import ShallowModel, integrate_sonic
from swt.timedepth.model import TimeDepth
from swt.tie.iterate import tie

BIN_M = 25.0
X0, Y0 = 620000.0, 6074000.0


@pytest.fixture(scope="module")
def volume(tmp_path_factory):
    """A 20x20 volume whose every trace carries a constant identifying it.

    Because each trace is a distinct constant, the composite extracted along a
    path says exactly which trace supplied each sample -- so the test can check
    the routing rather than merely that it produced numbers.
    """
    import segyio

    path = str(tmp_path_factory.mktemp("segy") / "volume.sgy")
    n_il = n_xl = 20
    n_samples, dt_ms = 200, 4

    spec = segyio.spec()
    spec.format = 5
    spec.samples = list(range(0, n_samples * dt_ms, dt_ms))
    spec.ilines = list(range(1, n_il + 1))
    spec.xlines = list(range(1, n_xl + 1))
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING

    with segyio.create(path, spec) as handle:
        index = 0
        for inline in spec.ilines:
            for xline in spec.xlines:
                handle.trace[index] = np.full(
                    n_samples, inline * 1000 + xline, dtype=np.float32
                )
                handle.header[index] = {
                    segyio.TraceField.INLINE_3D: inline,
                    segyio.TraceField.CROSSLINE_3D: xline,
                    segyio.TraceField.CDP_X: int((X0 + (inline - 1) * BIN_M) * 100),
                    segyio.TraceField.CDP_Y: int((Y0 + (xline - 1) * BIN_M) * 100),
                    segyio.TraceField.SourceGroupScalar: -100,
                    segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_ms * 1000,
                }
                index += 1
    return path


def straight_time_depth(v: float = 2500.0) -> TimeDepth:
    depth = np.linspace(0.0, 3000.0, 3001)
    return TimeDepth(depth[1:], 2.0 * depth[1:] / v, "test")


class TestSurveyInverse:
    def test_depth_to_md_round_trips(self):
        survey = dev.minimum_curvature(
            np.array([0.0, 1000.0, 3000.0]),
            np.array([0.0, 0.0, 60.0]),
            np.array([0.0, 90.0, 90.0]),
        )
        md = np.array([500.0, 1500.0, 2500.0])
        assert survey.md_at_tvdss(survey.tvdss_at_md(md)) == pytest.approx(md, abs=1.0)

    def test_kb_elevation_is_honoured_in_both_directions(self):
        survey = dev.vertical(np.array([0.0, 2000.0]), kb_elevation_m=30.0)
        assert survey.tvdss_at_md(np.array([1000.0]))[0] == pytest.approx(970.0)
        assert survey.md_at_tvdss(np.array([970.0]))[0] == pytest.approx(1000.0)

    def test_a_horizontal_section_is_excluded_rather_than_averaged(self):
        """Past 90 degrees the well stops going down; TVD is no longer invertible."""
        survey = dev.minimum_curvature(
            np.array([0.0, 1000.0, 2000.0, 3000.0]),
            np.array([0.0, 0.0, 90.0, 90.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
        )
        # The build section still inverts; the horizontal tail cannot add TVD.
        deepest = float(np.max(survey.tvd_m))
        md = survey.md_at_tvdss(np.array([deepest]))
        assert np.all(np.isfinite(md))
        assert float(md[0]) <= 2001.0

    def test_a_well_that_never_goes_down_is_refused(self):
        survey = dev.Deviation(
            md_m=np.array([0.0, 100.0, 200.0]),
            tvd_m=np.zeros(3),
            east_m=np.array([0.0, 100.0, 200.0]),
            north_m=np.zeros(3),
            name="horizontal",
        )
        with pytest.raises(ValueError, match="never increases"):
            survey.md_at_tvdss(np.array([50.0]))

    def test_absolute_position_adds_the_wellhead(self):
        survey = dev.minimum_curvature(
            np.array([0.0, 1000.0]), np.array([30.0, 30.0]), np.array([90.0, 90.0]),
            wellhead_x=620300.0, wellhead_y=6074500.0,
        )
        x, y = survey.absolute_position_at_md(np.array([1000.0]))
        assert x[0] > 620300.0            # moved east
        assert y[0] == pytest.approx(6074500.0, abs=1e-6)


class TestPathExtraction:
    def test_a_vertical_well_matches_single_point_extraction(self, volume):
        """The deviated path must reduce to the simple case when there is no deviation."""
        x, y = X0 + 5 * BIN_M, Y0 + 7 * BIN_M
        survey = dev.vertical(np.linspace(0.0, 3000.0, 31), wellhead_x=x, wellhead_y=y)

        along = extract_along_path(volume, survey, straight_time_depth())
        single = extract_trace(volume, x=x, y=y)

        assert along.amplitude == pytest.approx(single.amplitude)
        assert along.meta["n_distinct_traces"] == 1

    def test_a_deviated_well_reads_from_many_traces(self, volume):
        """The point of the exercise: the composite follows the borehole."""
        survey = dev.minimum_curvature(
            np.array([0.0, 500.0, 3000.0]),
            np.array([0.0, 0.0, 70.0]),
            np.array([0.0, 0.0, 0.0]),          # due north -> crossline direction
            wellhead_x=X0 + 2 * BIN_M, wellhead_y=Y0 + 2 * BIN_M,
        )
        trace = extract_along_path(volume, survey, straight_time_depth())

        assert trace.meta["n_distinct_traces"] > 5, trace.meta
        assert trace.meta["lateral_travel_m"] > 100.0
        # Every sample must come from a real trace in the volume.
        codes = np.unique(trace.amplitude)
        for code in codes:
            inline, xline = divmod(int(round(float(code))), 1000)
            assert 1 <= inline <= 20 and 1 <= xline <= 20

    def test_the_composite_routes_each_sample_to_the_nearest_trace(self, volume):
        """Check the routing itself, not just that traces were read."""
        survey = dev.minimum_curvature(
            np.array([0.0, 500.0, 3000.0]),
            np.array([0.0, 0.0, 70.0]),
            np.array([0.0, 0.0, 0.0]),
            wellhead_x=X0 + 2 * BIN_M, wellhead_y=Y0 + 2 * BIN_M,
        )
        time_depth = straight_time_depth()
        trace = extract_along_path(volume, survey, time_depth)

        depth = time_depth.depth_at(trace.twt)
        well_x, well_y = survey.position_at_tvdss(depth)
        expected_inline = np.round((well_x - X0) / BIN_M) + 1
        expected_xline = np.round((well_y - Y0) / BIN_M) + 1

        inside = (
            (expected_inline >= 1) & (expected_inline <= 20)
            & (expected_xline >= 1) & (expected_xline <= 20)
        )
        expected = expected_inline[inside] * 1000 + expected_xline[inside]
        assert trace.amplitude[inside] == pytest.approx(expected)

    def test_the_well_moves_deeper_down_the_hole(self, volume):
        """Shallow samples come from near the wellhead, deep ones from far away."""
        survey = dev.minimum_curvature(
            np.array([0.0, 500.0, 3000.0]),
            np.array([0.0, 0.0, 70.0]),
            np.array([0.0, 0.0, 0.0]),
            wellhead_x=X0 + 2 * BIN_M, wellhead_y=Y0 + 2 * BIN_M,
        )
        trace = extract_along_path(volume, survey, straight_time_depth())
        first, last = float(trace.amplitude[0]), float(trace.amplitude[-1])
        assert first != last, "the composite never left the wellhead"

    def test_a_volume_without_geometry_is_refused(self, tmp_path):
        """Following a path needs coordinates; guessing them would be worse."""
        import segyio

        path = str(tmp_path / "nogeom.sgy")
        spec = segyio.spec()
        spec.format = 5
        spec.samples = list(range(0, 400, 4))
        spec.ilines = [1, 2]
        spec.xlines = [1, 2]
        spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
        with segyio.create(path, spec) as handle:
            for i in range(4):
                handle.trace[i] = np.zeros(100, dtype=np.float32)
                handle.header[i] = {segyio.TraceField.SourceGroupScalar: 1}

        survey = dev.vertical(np.linspace(0.0, 3000.0, 31))
        with pytest.raises(ValueError, match="geometry is missing"):
            extract_along_path(path, survey, straight_time_depth())


class TestDeviatedTie:
    """The failure that motivates all of the above."""

    @staticmethod
    def deviated_case(seed: int = 1):
        survey = dev.minimum_curvature(
            np.array([0.0, 800.0, 3600.0]),
            np.array([0.0, 0.0, 55.0]),
            np.array([0.0, 45.0, 45.0]),
        )
        case = make_case(
            seed=seed, deviation=survey, log_top=700.0, log_base=2800.0,
            static_s=0.010, drift_amplitude=0.02,
        )
        return case, survey

    def test_the_case_is_genuinely_deviated(self):
        case, survey = self.deviated_case()
        assert case.log_depth_md_m is not None
        separation = case.log_depth_md_m[-1] - case.log_depth_tvdss[-1]
        assert separation > 200.0, f"only {separation:.0f} m of MD-TVD separation"

    def test_integrating_against_measured_depth_corrupts_the_tie(self):
        """The headline failure, demonstrated rather than asserted in a comment.

        Using MD as if it were TVDSS stretches the time-depth curve by the
        borehole's excess length -- hundreds of milliseconds here -- and the
        error is smooth, so nothing downstream flags it as anomalous.
        """
        # Isolate the confusion: no drift, no log noise, and the true time at
        # the log top, so the *only* difference between the two integrations is
        # which depth axis they used.
        survey = dev.minimum_curvature(
            np.array([0.0, 800.0, 3600.0]),
            np.array([0.0, 0.0, 55.0]),
            np.array([0.0, 45.0, 45.0]),
        )
        case = make_case(
            seed=1, deviation=survey, log_top=700.0, log_base=2800.0,
            drift_amplitude=0.0, sonic_noise_fraction=0.0,
        )
        truth = case.true_twt_at_log()
        shallow = ShallowModel(twt_at_log_top=float(truth[0]))

        wrong = integrate_sonic(case.log_depth_md_m, case.log_dt_us_per_m, shallow)
        right = integrate_sonic(case.log_depth_tvdss, case.log_dt_us_per_m, shallow)

        wrong_error = float(np.max(np.abs(wrong.twt - truth)))
        right_error = float(np.max(np.abs(right.twt - truth)))

        # Using MD as TVDSS adds the borehole's excess length as extra time.
        assert wrong_error > 0.150, f"only {wrong_error * 1e3:.0f} ms of damage"
        # With the same log on the right axis, the integration is essentially exact.
        assert right_error < 0.002, f"TVDSS integration off by {right_error * 1e3:.1f} ms"

    def test_a_deviated_well_ties_correctly_in_tvdss(self):
        case, _ = self.deviated_case()
        calibrated = calibrate_to_checkshots(
            integrate_sonic(
                case.log_depth_tvdss, case.log_dt_us_per_m,
                ShallowModel(replacement_velocity=1900.0),
            ),
            case.checkshot_depth_tvdss, case.checkshot_twt,
        ).time_depth

        result = tie(
            case.seismic, calibrated, case.log_dt_us_per_m, case.log_rho_g_cm3
        )

        truth = case.true_twt_at_log() + 0.010
        error = float(np.sqrt(np.mean((result.time_depth.twt - truth) ** 2)))
        assert error < 0.004, f"deviated tie off by {error * 1e3:.1f} ms rms"
        assert result.total_shift_s == pytest.approx(0.010, abs=case.seismic.dt)
        assert result.metrics.correlation > 0.85
