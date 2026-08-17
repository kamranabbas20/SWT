"""The headline tests: does the pipeline recover a known answer?

Everything else in the suite checks that a function does what its docstring
says.  These check the only thing that actually matters -- that a tie built from
a forward-modelled earth lands where the forward model put it.

A correlation coefficient cannot do this job.  A pipeline with a sign error, a
half-sample shift or a phase/time confusion will still report a high
correlation, because a high correlation is exactly what it optimised for.  Only
ground truth distinguishes "tied well" from "tied well to the wrong place".
"""

from __future__ import annotations

import numpy as np
import pytest

from swt.forward import make_case
from swt.timedepth.calibrate import calibrate_to_checkshots
from swt.timedepth.integrate import ShallowModel, integrate_sonic
from swt.tie.iterate import tie


def build_calibrated(case, replacement_velocity: float = 1900.0):
    """The standard M1 front half: integrate the sonic, calibrate to checkshots."""
    raw = integrate_sonic(
        case.log_depth_tvdss,
        case.log_dt_us_per_m,
        ShallowModel(replacement_velocity=replacement_velocity),
    )
    return calibrate_to_checkshots(
        raw, case.checkshot_depth_tvdss, case.checkshot_twt
    ).time_depth


class TestCheckshotCalibration:
    """The sonic disagrees with the earth; the checkshots must fix it."""

    def test_calibration_recovers_true_time_depth(self):
        case = make_case(seed=1, drift_amplitude=0.03)
        truth = case.true_twt_at_log()

        raw = integrate_sonic(
            case.log_depth_tvdss,
            case.log_dt_us_per_m,
            ShallowModel(replacement_velocity=1900.0),
        )
        calibrated = build_calibrated(case)

        raw_error = np.abs(raw.twt - truth).max()
        calibrated_error = np.abs(calibrated.twt - truth).max()

        # The uncalibrated sonic should be badly wrong -- otherwise the test is
        # not exercising anything.
        assert raw_error > 0.030, "forward model produced no meaningful drift"
        assert calibrated_error < 0.005
        assert calibrated_error < raw_error / 10.0

    def test_calibration_honours_every_checkshot(self):
        case = make_case(seed=2)
        raw = integrate_sonic(
            case.log_depth_tvdss,
            case.log_dt_us_per_m,
            ShallowModel(replacement_velocity=1900.0),
        )
        result = calibrate_to_checkshots(
            raw, case.checkshot_depth_tvdss, case.checkshot_twt
        )
        assert result.max_abs_residual_ms < 0.01

    @pytest.mark.parametrize("drift", [0.0, 0.01, 0.05])
    def test_calibration_works_across_drift_magnitudes(self, drift):
        case = make_case(seed=4, drift_amplitude=drift)
        calibrated = build_calibrated(case)
        error = np.abs(calibrated.twt - case.true_twt_at_log()).max()
        assert error < 0.006


class TestTieRecovery:
    """The full loop must recover the imposed static and phase, not just correlate."""

    @pytest.mark.parametrize("phase", [0.0, 30.0, 90.0, 180.0, -60.0])
    @pytest.mark.parametrize("static_ms", [0.0, 12.0, -20.0])
    def test_recovers_static_and_phase(self, phase, static_ms):
        static = static_ms / 1e3
        case = make_case(seed=3, static_s=static, wavelet_phase_deg=phase)
        result = tie(
            case.seismic,
            build_calibrated(case),
            case.log_dt_us_per_m,
            case.log_rho_g_cm3,
        )

        # The seismic was moved by `static`, so the tie must move the well by
        # the same amount in the same direction.
        assert result.total_shift_s == pytest.approx(static, abs=0.003)

        # Phase is recovered modulo 360; +180 and -180 are the same rotation.
        phase_error = (result.phase_deg - phase + 180.0) % 360.0 - 180.0
        assert abs(phase_error) < 12.0

        assert result.metrics.correlation > 0.95
        assert result.significance.is_significant

    @pytest.mark.parametrize("phase", [0.0, 90.0, 180.0])
    def test_time_depth_lands_on_truth(self, phase):
        """The real acceptance test: is the *curve* right, not the correlation?"""
        static = 0.012
        case = make_case(seed=3, static_s=static, wavelet_phase_deg=phase)
        result = tie(
            case.seismic,
            build_calibrated(case),
            case.log_dt_us_per_m,
            case.log_rho_g_cm3,
        )

        truth = case.true_twt_at_log() + static
        rms_error = float(np.sqrt(np.mean((result.time_depth.twt - truth) ** 2)))
        assert rms_error < 0.003, f"time-depth off by {rms_error * 1e3:.1f} ms rms"

    def test_high_correlation_is_not_enough_on_its_own(self):
        """Regression guard for the phase-scan bug that this suite first caught.

        A 180 degree wavelet paired with a half-period time shift correlates
        beautifully and puts the well in the wrong place.  Before the fix, this
        case reported correlation 0.995 with a time-depth 16 ms out.
        """
        case = make_case(seed=3, static_s=0.0, wavelet_phase_deg=180.0)
        result = tie(
            case.seismic,
            build_calibrated(case),
            case.log_dt_us_per_m,
            case.log_rho_g_cm3,
        )
        assert result.metrics.correlation > 0.95
        assert abs(result.total_shift_s) < 0.003, (
            "a high correlation was achieved by shifting half a period -- "
            "the phase and time estimates have been confused again"
        )

    @pytest.mark.parametrize("snr", [None, 20.0, 4.0, 2.0])
    def test_degrades_gracefully_with_noise(self, snr):
        case = make_case(seed=5, signal_to_noise=snr, static_s=0.008)
        result = tie(
            case.seismic,
            build_calibrated(case),
            case.log_dt_us_per_m,
            case.log_rho_g_cm3,
        )
        assert result.total_shift_s == pytest.approx(0.008, abs=0.004)
        # Noisier data must tie worse, but must still tie.
        assert result.metrics.correlation > (0.5 if snr and snr <= 2.0 else 0.8)

    @pytest.mark.parametrize("seed", range(10))
    def test_stable_across_earth_models(self, seed):
        """Not one lucky random earth: the result must hold across many."""
        case = make_case(seed=seed, static_s=0.010)
        result = tie(
            case.seismic,
            build_calibrated(case),
            case.log_dt_us_per_m,
            case.log_rho_g_cm3,
        )
        # Within one seismic sample of the truth, on every earth model.
        assert result.total_shift_s == pytest.approx(0.010, abs=case.seismic.dt)
        assert result.metrics.correlation > 0.85

    @pytest.mark.parametrize("seed", range(10))
    def test_wavelet_never_absorbs_the_static(self, seed):
        """A least-squares wavelet must not hide a residual shift in its shape.

        Before the fix this caught, the estimator returned a wavelet whose
        energy was offset by the residual static.  The loop then converged --
        the next shift search found nothing, because the error was inside the
        wavelet -- on a time-depth several milliseconds out.
        """
        case = make_case(seed=seed, static_s=0.010)
        result = tie(
            case.seismic,
            build_calibrated(case),
            case.log_dt_us_per_m,
            case.log_rho_g_cm3,
        )
        offset_ms = abs(result.wavelet.energy_centre_s()) * 1e3
        assert offset_ms < 0.5 * result.wavelet.dt * 1e3, (
            f"wavelet energy sits {offset_ms:.1f} ms off centre -- it has absorbed "
            "a time shift that belongs in the time-depth model"
        )


class TestWaveletRecovery:
    def test_deterministic_wavelet_matches_the_true_one(self):
        case = make_case(seed=7, signal_to_noise=None, wavelet_frequency=25.0)
        result = tie(
            case.seismic,
            build_calibrated(case),
            case.log_dt_us_per_m,
            case.log_rho_g_cm3,
        )
        estimated = result.wavelet.dominant_frequency()
        assert estimated == pytest.approx(case.true_wavelet.dominant_frequency(), abs=6.0)

    def test_statistical_wavelet_finds_the_band(self):
        from swt.wavelet.statistical import statistical_wavelet

        case = make_case(seed=8, wavelet_frequency=35.0, signal_to_noise=None)
        window = case.logged_time_window()
        wavelet = statistical_wavelet(
            case.seismic.window(*window).amplitude, case.seismic.dt, 0.128
        )
        assert wavelet.dominant_frequency() == pytest.approx(35.0, abs=8.0)
