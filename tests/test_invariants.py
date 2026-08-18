"""Property and invariant tests for the deterministic core.

These check the guarantees the rest of the package is allowed to rely on -- and,
in several cases, that the failure modes the docstrings warn about are actually
prevented rather than merely described.
"""

from __future__ import annotations

import numpy as np
import pytest

from swt.io import deviation as dev
from swt.logs.condition import despike, detect_cycle_skips, fill_gaps, flag_bad_hole
from swt.petro.elastic import acoustic_impedance, backus_average, reflectivity
from swt.qc import significance as sig
from swt.qc.metrics import correlation_coefficient, nrms
from swt.synth.resample import reflectivity_in_time
from swt.timedepth.calibrate import CalibrationError, calibrate_to_checkshots
from swt.timedepth.integrate import ShallowModel, integrate_sonic
from swt.timedepth.model import MonotonicityError, TimeDepth
from swt.trace import Trace
from swt.units import UnitError, slowness_to_us_per_m, depth_to_m
from swt.wavelet.core import Wavelet
from swt.wavelet.parametric import ormsby, ricker


class TestTimeDepthInvariant:
    def test_rejects_non_monotonic_time(self):
        with pytest.raises(MonotonicityError, match="not strictly increasing"):
            TimeDepth(np.array([100.0, 200.0, 300.0]), np.array([0.1, 0.3, 0.2]))

    def test_rejects_non_monotonic_depth(self):
        with pytest.raises(MonotonicityError):
            TimeDepth(np.array([100.0, 90.0, 300.0]), np.array([0.1, 0.2, 0.3]))

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            TimeDepth(np.array([100.0, 200.0]), np.array([0.1, np.nan]))

    def test_round_trips_depth_and_time(self):
        td = TimeDepth(np.array([100.0, 200.0, 400.0]), np.array([0.1, 0.18, 0.32]))
        depths = np.array([120.0, 250.0, 390.0])
        assert td.depth_at(td.time_at(depths)) == pytest.approx(depths)

    def test_extrapolates_rather_than_clamping(self):
        """Clamping would create a zero-velocity layer outside the model."""
        td = TimeDepth(np.array([100.0, 200.0]), np.array([0.1, 0.2]))
        assert td.time_at(np.array([300.0]))[0] == pytest.approx(0.3)
        assert td.time_at(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_interval_velocity_uses_two_way_time(self):
        # 100 m in 0.1 s two-way is 2000 m/s, not 1000.
        td = TimeDepth(np.array([0.0, 100.0]), np.array([0.0, 0.1]))
        _, velocity = td.interval_velocity()
        assert velocity[0] == pytest.approx(2000.0)


class TestSonicIntegration:
    def test_matches_analytic_answer_for_constant_velocity(self):
        depth = np.linspace(1000.0, 2000.0, 5001)
        velocity = 2500.0
        dt = np.full_like(depth, 1e6 / velocity)
        td = integrate_sonic(depth, dt, ShallowModel(twt_at_log_top=0.8))
        expected = 0.8 + 2.0 * (depth - depth[0]) / velocity
        assert td.twt == pytest.approx(expected, abs=1e-9)

    def test_refuses_to_guess_the_shallow_section(self):
        with pytest.raises(ValueError, match="exactly one"):
            ShallowModel()
        with pytest.raises(ValueError, match="exactly one"):
            ShallowModel(replacement_velocity=1800.0, twt_at_log_top=0.5)

    def test_refuses_nan_slowness(self):
        depth = np.linspace(1000.0, 1100.0, 101)
        dt = np.full_like(depth, 400.0)
        dt[50] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            integrate_sonic(depth, dt, ShallowModel(replacement_velocity=1800.0))

    def test_replacement_velocity_sets_the_start_time(self):
        depth = np.linspace(800.0, 900.0, 101)
        dt = np.full_like(depth, 400.0)
        td = integrate_sonic(depth, dt, ShallowModel(replacement_velocity=2000.0))
        assert td.twt[0] == pytest.approx(2.0 * 800.0 / 2000.0)


class TestCalibration:
    def test_raises_when_correction_would_invert_time(self):
        depth = np.linspace(1000.0, 2000.0, 1001)
        td = integrate_sonic(
            depth, np.full_like(depth, 400.0), ShallowModel(twt_at_log_top=0.8)
        )
        # Checkshot times that run backwards against depth are rejected outright.
        with pytest.raises(ValueError, match="must increase with depth"):
            calibrate_to_checkshots(
                td, np.array([1200.0, 1600.0]), np.array([1.2, 1.0])
            )

    def test_detects_a_drift_that_outruns_the_sonic(self):
        """Tight knots either side of a slow zone over-correct the fast rock.

        Note the shape of this failure: with a *constant-velocity* sonic it
        cannot happen at all, because the corrected curve is then linear between
        knots and monotone whenever the checkshots are.  It takes a varying
        sonic gradient -- here a 20 m washed-out interval -- plus knots placed
        tightly around it.  The whole drift then has to be removed across a
        short interval, and the correction's slope exceeds the sonic's own time
        gradient in the fast rock either side.
        """
        depth = np.linspace(1000.0, 2000.0, 1001)
        slowness = np.full_like(depth, 400.0)
        slow_zone = (depth >= 1500.0) & (depth <= 1520.0)
        slowness[slow_zone] = 1500.0

        td = integrate_sonic(depth, slowness, ShallowModel(twt_at_log_top=0.8))

        # Checkshots consistent with 2500 m/s throughout -- they never saw the
        # washout -- placed just either side of it.
        cs_depth = np.array([1400.0, 1495.0, 1525.0, 1900.0])
        cs_twt = 0.8 + 2.0 * (cs_depth - 1000.0) / 2500.0

        with pytest.raises(CalibrationError, match="inverted the time-depth"):
            calibrate_to_checkshots(td, cs_depth, cs_twt)

    def test_the_same_drift_is_fine_with_wider_knots(self):
        """...and the error message's advice actually works."""
        depth = np.linspace(1000.0, 2000.0, 1001)
        slowness = np.full_like(depth, 400.0)
        slowness[(depth >= 1500.0) & (depth <= 1520.0)] = 1500.0
        td = integrate_sonic(depth, slowness, ShallowModel(twt_at_log_top=0.8))

        cs_depth = np.array([1100.0, 1400.0, 1800.0])
        cs_twt = 0.8 + 2.0 * (cs_depth - 1000.0) / 2500.0

        result = calibrate_to_checkshots(td, cs_depth, cs_twt)
        assert result.max_abs_residual_ms < 0.01


class TestResampling:
    def test_bin_averaging_beats_naive_interpolation_on_a_thin_bed_stack(self):
        """The anti-aliasing claim in swt.synth.resample, made concrete.

        A stack of beds far below seismic resolution should produce almost no
        energy at the seismic sample rate: the beds average out.  Point-sampling
        the impedance instead aliases them into the band, and the aliased series
        carries far more energy than it should.
        """
        twt = np.linspace(1.0, 2.0, 200001)
        # Beds ~0.2 ms thick, i.e. a tenth of the 2 ms sample interval.
        impedance = 6000.0 + 600.0 * np.sign(np.sin(2 * np.pi * twt / 0.0004))

        proper = reflectivity_in_time(twt, impedance, dt=0.002)
        proper_energy = float(np.sum(proper.rc**2))

        axis = proper.twt
        point_sampled = np.interp(axis, twt, impedance)
        naive = reflectivity(point_sampled)
        naive_energy = float(np.sum(naive**2))

        assert proper_energy < naive_energy / 50.0

    def test_preserves_a_single_resolvable_interface(self):
        twt = np.linspace(1.0, 2.0, 100001)
        impedance = np.where(twt < 1.5, 5000.0, 7000.0)
        result = reflectivity_in_time(twt, impedance, dt=0.002)
        expected = (7000.0 - 5000.0) / (7000.0 + 5000.0)
        assert result.rc.sum() == pytest.approx(expected, rel=0.02)

    def test_marks_padding_as_invalid(self):
        twt = np.linspace(1.0, 1.5, 5001)
        impedance = np.full_like(twt, 6000.0)
        result = reflectivity_in_time(twt, impedance, dt=0.002, t_start=0.5, t_end=2.0)
        t0, t1 = result.valid_window()
        assert t0 >= 1.0 and t1 <= 1.5
        assert not result.valid[0] and not result.valid[-1]

    def test_rejects_non_monotonic_log_time(self):
        twt = np.array([1.0, 1.1, 1.05, 1.2])
        with pytest.raises(ValueError, match="strictly increasing"):
            reflectivity_in_time(twt, np.full(4, 6000.0), dt=0.002)


class TestElastic:
    def test_reflectivity_signs(self):
        assert reflectivity(np.array([5000.0, 7000.0]))[0] > 0
        assert reflectivity(np.array([7000.0, 5000.0]))[0] < 0

    def test_reflectivity_is_scale_invariant(self):
        ai = np.array([5000.0, 7000.0, 6000.0])
        assert reflectivity(ai) == pytest.approx(reflectivity(ai * 3.7))

    def test_backus_uses_harmonic_modulus_not_arithmetic_velocity(self):
        """Harmonic averaging must give a slower medium than arithmetic."""
        depth = np.arange(0.0, 100.0, 0.1)
        velocity = np.where((depth // 5) % 2 == 0, 2000.0, 4000.0)
        density = np.full_like(velocity, 2.4)

        upscaled_v, _ = backus_average(depth, velocity, density, length_m=30.0)
        middle = slice(200, 800)
        assert upscaled_v[middle].mean() < velocity[middle].mean()

    def test_backus_leaves_a_homogeneous_medium_alone(self):
        depth = np.arange(0.0, 100.0, 0.1)
        velocity = np.full_like(depth, 3000.0)
        density = np.full_like(depth, 2.4)
        upscaled_v, upscaled_rho = backus_average(depth, velocity, density, 30.0)
        assert upscaled_v == pytest.approx(velocity, rel=1e-9)
        assert upscaled_rho == pytest.approx(density, rel=1e-9)


class TestWavelets:
    def test_length_is_forced_odd_so_the_centre_is_exact(self):
        with pytest.raises(ValueError, match="odd"):
            Wavelet(np.zeros(10) + 1.0, dt=0.002)
        assert ricker(30.0, 0.002).n % 2 == 1

    def test_ricker_peaks_at_zero_time(self):
        w = ricker(30.0, 0.002)
        assert int(np.argmax(w.samples)) == w.centre
        assert w.time()[w.centre] == 0.0

    def test_ricker_dominant_frequency_matches_its_parameter(self):
        for frequency in (15.0, 30.0, 45.0):
            assert ricker(frequency, 0.001).dominant_frequency() == pytest.approx(
                frequency, rel=0.1
            )

    def test_rotation_by_360_is_the_identity(self):
        w = ricker(30.0, 0.002)
        assert w.rotate(360.0).samples == pytest.approx(w.samples, abs=1e-9)

    def test_rotation_by_180_negates(self):
        w = ricker(30.0, 0.002)
        assert w.rotate(180.0).samples == pytest.approx(-w.samples, abs=1e-9)

    def test_rotation_composes(self):
        w = ricker(30.0, 0.002)
        assert w.rotate(30.0).rotate(60.0).samples == pytest.approx(
            w.rotate(90.0).samples, abs=1e-9
        )

    def test_ormsby_band_matches_its_corners(self):
        w = ormsby(8.0, 12.0, 40.0, 50.0, 0.001)
        low, high = w.bandwidth(level_db=-6.0)
        assert 8.0 <= low <= 16.0
        assert 36.0 <= high <= 52.0

    def test_ormsby_rejects_corners_above_nyquist(self):
        with pytest.raises(ValueError, match="Nyquist"):
            ormsby(8.0, 12.0, 200.0, 300.0, 0.002)


class TestSignificance:
    def test_dof_ignores_the_sample_rate(self):
        """Resampling adds samples and no information; DOF must not move."""
        assert sig.effective_dof(0.2, 40.0) == sig.effective_dof(0.2, 40.0)
        assert sig.effective_dof(0.4, 40.0) > sig.effective_dof(0.2, 40.0)
        assert sig.effective_dof(0.2, 80.0) > sig.effective_dof(0.2, 40.0)

    def test_short_narrowband_window_makes_a_good_looking_correlation_meaningless(self):
        # 0.6 over 100 ms of 10-30 Hz data: ~5 independent samples.
        assert not sig.assess(0.6, window_length_s=0.1, bandwidth_hz=20.0).is_significant
        # The same 0.6 over a second of broadband data is solid.
        assert sig.assess(0.6, window_length_s=1.0, bandwidth_hz=60.0).is_significant

    def test_critical_correlation_falls_as_the_window_grows(self):
        short = sig.assess(0.6, 0.1, 20.0).critical_correlation
        long = sig.assess(0.6, 2.0, 20.0).critical_correlation
        assert short > long

    def test_verdict_is_a_sentence_a_human_can_act_on(self):
        verdict = sig.assess(0.6, 0.1, 20.0).verdict()
        assert "NOT significant" in verdict and "independent samples" in verdict


class TestConditioning:
    def test_despike_removes_spikes_and_reports_them(self):
        depth = np.arange(1000.0, 1100.0, 0.15)
        log = 400.0 + 5.0 * np.sin(depth / 10.0)
        log[100] = 1200.0
        log[300] = 40.0

        report = despike(log, depth)
        assert len(report.edits) == 2
        assert all(e.kind == "spike" for e in report.edits)
        assert report.log.max() < 500.0 and report.log.min() > 300.0

    def test_despike_leaves_a_clean_log_alone(self):
        depth = np.arange(1000.0, 1100.0, 0.15)
        log = 400.0 + 5.0 * np.sin(depth / 10.0)
        assert despike(log, depth).is_clean

    def test_despike_leaves_bed_boundaries_and_cycle_skips_alone(self):
        """A spike is narrow; a sustained step is geology or a skip.

        Both must survive despiking -- geology because it is the signal, and
        cycle skips because they are reported for a human to judge, which
        cannot happen if despiking has already smoothed them away.
        """
        rng = np.random.default_rng(0)
        depth = np.arange(1000.0, 1200.0, 0.15)
        log = np.where(depth < 1100.0, 400.0, 470.0)      # a bed boundary
        log = log * (1.0 + 0.005 * rng.standard_normal(depth.size))
        log[600:640] *= 2.0                                # a 6 m cycle skip
        log[50] = 1500.0                                   # an actual spike

        report = despike(log, depth)

        assert any(e.n_samples <= 5 for e in report.edits), "the real spike was missed"
        assert report.log[50] < 600.0, "the real spike was not removed"
        # The skip and the bed boundary are still there.
        assert report.log[620] > 700.0, "a cycle skip was silently smoothed away"
        assert report.log[1000] == pytest.approx(470.0, rel=0.05)
        assert detect_cycle_skips(report.log, depth), "the skip is no longer detectable"

    def test_detects_a_sustained_cycle_skip_but_not_a_single_spike(self):
        depth = np.arange(1000.0, 1100.0, 0.15)
        log = np.full_like(depth, 400.0)
        log[200:230] *= 2.0  # a skip: sustained
        log[500] *= 2.0      # a spike: one sample

        skips = detect_cycle_skips(log, depth, min_samples=3)
        assert len(skips) == 1
        assert skips[0].start_depth == pytest.approx(depth[200], abs=0.2)

    def test_fill_gaps_interpolates_short_and_refuses_long(self):
        depth = np.arange(1000.0, 1100.0, 0.15)
        log = np.full_like(depth, 400.0)
        log[100:110] = np.nan   # ~1.5 m
        log[300:400] = np.nan   # ~15 m

        report = fill_gaps(log, depth, max_gap_m=5.0)
        kinds = {e.kind for e in report.edits}
        assert kinds == {"gap_filled", "gap_left"}
        assert np.isfinite(report.log[100:110]).all()
        assert not np.isfinite(report.log[300:400]).any()
        assert report.remaining_nan == 100

    def test_flags_washed_out_hole(self):
        depth = np.arange(1000.0, 1100.0, 0.15)
        caliper = np.full_like(depth, 8.5)
        caliper[200:260] = 12.0
        edits = flag_bad_hole(caliper, depth, bit_size=8.5)
        assert len(edits) == 1 and edits[0].kind == "bad_hole"


class TestUnits:
    def test_refuses_unknown_units(self):
        with pytest.raises(UnitError, match="unrecognised"):
            slowness_to_us_per_m(np.array([100.0]), "furlongs/fortnight")

    def test_refuses_missing_units(self):
        with pytest.raises(UnitError):
            depth_to_m(np.array([100.0]), None)

    def test_us_per_ft_conversion(self):
        assert slowness_to_us_per_m(np.array([100.0]), "us/ft")[0] == pytest.approx(
            100.0 / 0.3048
        )

    def test_tolerates_punctuation_and_case(self):
        for spelling in ("US/FT", "us/ft", " us / ft ", "USFT"):
            assert slowness_to_us_per_m(np.array([1.0]), spelling)[0] == pytest.approx(
                1.0 / 0.3048
            )


class TestDeviation:
    def test_vertical_well_has_tvd_equal_to_md(self):
        survey = dev.minimum_curvature(
            np.array([0.0, 1000.0, 2000.0]),
            np.zeros(3),
            np.zeros(3),
        )
        assert survey.tvd_m == pytest.approx(np.array([0.0, 1000.0, 2000.0]))
        assert survey.is_vertical
        assert survey.max_departure_m() == pytest.approx(0.0)

    def test_sixty_degree_hold_section(self):
        """A straight 60-degree section gains MD*cos(60) = MD/2 of TVD."""
        survey = dev.minimum_curvature(
            np.array([1000.0, 2000.0]),
            np.array([60.0, 60.0]),
            np.array([90.0, 90.0]),
        )
        assert survey.tvd_m[1] - survey.tvd_m[0] == pytest.approx(500.0)
        # Azimuth 90 is due east.
        assert survey.east_m[1] == pytest.approx(1000.0 * np.sin(np.deg2rad(60.0)))
        assert survey.north_m[1] == pytest.approx(0.0, abs=1e-9)
        assert not survey.is_vertical

    def test_kb_elevation_is_removed_for_tvdss(self):
        survey = dev.vertical(np.array([0.0, 1000.0]), kb_elevation_m=25.0)
        assert survey.tvdss_at_md(np.array([1000.0]))[0] == pytest.approx(975.0)


class TestTrace:
    def test_rejects_irregular_time_axis(self):
        with pytest.raises(ValueError, match="uniformly sampled"):
            Trace(np.array([0.0, 0.002, 0.005]), np.zeros(3))

    def test_shift_moves_events_later(self):
        twt = np.arange(0.0, 1.0, 0.002)
        amplitude = np.zeros_like(twt)
        amplitude[100] = 1.0
        shifted = Trace(twt, amplitude).shifted(0.020)
        assert int(np.argmax(shifted.amplitude)) == 110

    def test_metrics_behave_as_documented(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(500)
        assert correlation_coefficient(a, a) == pytest.approx(1.0)
        assert correlation_coefficient(a, -a) == pytest.approx(-1.0)
        assert nrms(a, a) == pytest.approx(0.0)
        assert nrms(a, -a) == pytest.approx(200.0, rel=0.01)


class TestWaveletRecentring:
    """A wavelet must not be allowed to carry a bulk shift in its shape."""

    def test_recentring_a_centred_wavelet_is_a_no_op(self):
        w = ricker(30.0, 0.002)
        recentred, shift = w.recentred()
        assert shift == 0.0
        assert recentred is w

    def test_recentring_reports_and_removes_an_offset(self):
        w = ricker(30.0, 0.002)
        offset_samples = 5
        shifted = Wavelet(np.roll(w.samples, offset_samples), w.dt)

        assert shifted.energy_centre_s() == pytest.approx(offset_samples * w.dt, abs=w.dt)

        recentred, shift = shifted.recentred()
        assert shift == pytest.approx(offset_samples * w.dt, abs=1e-12)
        assert abs(recentred.energy_centre_s()) < 0.5 * w.dt
        assert int(np.argmax(np.abs(recentred.samples))) == w.centre

    def test_energy_centre_is_phase_blind(self):
        """A 90-degree wavelet has no obvious peak but a well defined centre."""
        w = ricker(30.0, 0.002)
        for phase in (0.0, 45.0, 90.0, 180.0):
            assert abs(w.rotate(phase).energy_centre_s()) < 0.5 * w.dt


class TestDuplicateMnemonics:
    """A LAS may name two curves the same thing. Found on real F3 data.

    lasio makes duplicates unique by appending ":1", ":2", so a lookup for "DT"
    matched neither and the sonic vanished -- while the load reported success
    with a null curve. A load that silently drops the most important log in the
    file is the exact failure this package is arranged against.
    """

    @staticmethod
    def write_las(tmp_path, curve_lines, data_lines):
        path = tmp_path / "well.las"
        path.write_text(
            "~Version\nVERS. 2.0 :\nWRAP. NO :\n"
            "~Well\nSTRT .M 30.0 :\nSTOP .M 200.0 :\nSTEP .M 0.15 :\n"
            "NULL . -999.25 :\nWELL . TEST :\n"
            "~Curve\n" + curve_lines + "~A\n" + data_lines
        )
        return str(path)

    def test_a_duplicated_sonic_is_still_found(self, tmp_path):
        path = self.write_las(
            tmp_path,
            "DEPTH .M   : 1\nRHOB .g/cc : 2\nDT .us/ft : 3 raw\nDT .us/ft : 4 corrected\n",
            "100.0 2.35 150.0 149.0\n150.0 2.40 145.0 144.0\n200.0 2.45 140.0 139.0\n",
        )
        from swt.io.las import load_las

        logs = load_las(path)
        assert logs.sonic_us_per_m is not None, "the duplicated sonic was dropped"
        assert logs.density_g_cm3 is not None

    def test_the_later_duplicate_is_used_and_the_choice_reported(self, tmp_path):
        path = self.write_las(
            tmp_path,
            "DEPTH .M   : 1\nDT .us/ft : 2 raw\nDT .us/ft : 3 corrected\n",
            "100.0 150.0 149.0\n150.0 145.0 144.0\n200.0 140.0 139.0\n",
        )
        from swt.io.las import load_las
        from swt.units import slowness_to_us_per_m

        logs = load_las(path)
        # The second DT column (149, 144, 139) is the one that should be used.
        assert logs.sonic_us_per_m[0] == pytest.approx(
            slowness_to_us_per_m(np.array([149.0]), "us/ft")[0]
        )
        assert any("2 curves for 'sonic'" in w for w in logs.warnings())

    def test_an_explicit_override_can_pick_the_other_one(self, tmp_path):
        path = self.write_las(
            tmp_path,
            "DEPTH .M   : 1\nDT .us/ft : 2 raw\nDT .us/ft : 3 corrected\n",
            "100.0 150.0 149.0\n150.0 145.0 144.0\n200.0 140.0 139.0\n",
        )
        from swt.io.las import load_las
        from swt.units import slowness_to_us_per_m

        logs = load_las(path, curves={"sonic": "DT:1"})
        assert logs.sonic_us_per_m[0] == pytest.approx(
            slowness_to_us_per_m(np.array([150.0]), "us/ft")[0]
        )

    def test_a_missing_sonic_is_announced_not_returned_as_none(self, tmp_path):
        path = self.write_las(
            tmp_path, "DEPTH .M : 1\nRHOB .g/cc : 2\n",
            "100.0 2.35\n150.0 2.40\n200.0 2.45\n",
        )
        from swt.io.las import load_las

        logs = load_las(path)
        assert logs.sonic_us_per_m is None
        warnings = logs.warnings()
        assert any("No sonic curve" in w for w in warnings)
        # The message must name what *is* in the file, so the fix is obvious.
        assert any("RHOB" in w for w in warnings)

    def test_the_tie_interval_is_where_both_curves_exist(self, tmp_path):
        """Real logs start at different depths; only the overlap can be tied."""
        path = self.write_las(
            tmp_path, "DEPTH .M : 1\nDT .us/ft : 2\nRHOB .g/cc : 3\n",
            "50.0 150.0 -999.25\n100.0 148.0 -999.25\n"
            "150.0 145.0 2.40\n200.0 140.0 2.45\n",
        )
        from swt.io.las import load_las

        logs = load_las(path)
        interval = logs.tie_interval()
        assert interval == (150.0, 200.0), interval
        assert any("both sonic and density" in w for w in logs.warnings())
