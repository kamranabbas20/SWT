"""Ground-truth tests for the warping solver and the auto-tie.

The property that matters here is **not** "the warp improves the correlation".
It always does; that is what it optimises. The property that matters is that the
warp improves, or declines to touch, the *time-depth curve* -- and never
degrades it while reporting a better correlation, which is the specific way a
warping auto-tie fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from swt.forward import make_case
from swt.tie.auto import auto_tie
from swt.tie.dtw import apply_warp, auto_warp, dynamic_warp
from swt.tie.guardrail import strain_limit_for
from swt.timedepth.calibrate import calibrate_to_checkshots
from swt.timedepth.integrate import ShallowModel, integrate_sonic
from swt.trace import Trace
from swt.wavelet.parametric import ricker

STRAIN = strain_limit_for(15.0)


def warped_pair(true_shift: np.ndarray, seed: int = 0, frequency: float = 28.0):
    """A synthetic and a copy of it displaced by a known, time-varying shift."""
    dt = 0.002
    twt = np.arange(0.6, 2.4, dt)
    rng = np.random.default_rng(seed)
    rc = np.zeros(twt.size)
    rc[rng.choice(twt.size, 120, replace=False)] = rng.standard_normal(120)
    synthetic = np.convolve(rc, ricker(frequency, dt).samples, mode="same")
    seismic = np.interp(twt - true_shift, twt, synthetic)
    return Trace(twt, seismic, "seismic"), Trace(twt, synthetic, "synthetic")


def anomaly_case(seed: int):
    """A tie carrying a localised sonic anomaly -- what a warp is actually for."""
    case = make_case(
        seed=seed, static_s=0.008, drift_amplitude=0.01,
        checkshot_spacing=600.0, anomaly_zone=(1700.0, 2100.0, 0.12),
    )
    raw = integrate_sonic(
        case.log_depth_tvdss, case.log_dt_us_per_m,
        ShallowModel(replacement_velocity=1900.0),
    )
    calibrated = calibrate_to_checkshots(
        raw, case.checkshot_depth_tvdss, case.checkshot_twt
    ).time_depth
    return case, calibrated, case.true_twt_at_log() + 0.008


def rms_ms(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error**2))) * 1e3


class TestWarpRecovery:
    """The solver must recover a shift field it was given."""

    @pytest.mark.parametrize(
        "name,builder",
        [
            ("zero", lambda t: np.zeros(t.size)),
            ("constant", lambda t: np.full(t.size, 0.010)),
            ("ramp up", lambda t: 0.030 * (t - t[0]) / (t[-1] - t[0])),
            ("ramp down", lambda t: -0.025 * (t - t[0]) / (t[-1] - t[0])),
            ("sinusoid", lambda t: 0.015 * np.sin(2 * np.pi * (t - t[0]) / 1.2)),
        ],
    )
    def test_recovers_a_known_shift_field(self, name, builder):
        twt = np.arange(0.6, 2.4, 0.002)
        true_shift = builder(twt)
        seismic, synthetic = warped_pair(true_shift)

        warp = dynamic_warp(seismic, synthetic, strain_limit=STRAIN, max_shift_s=0.060)

        error = (warp.shift_s - true_shift) * 1e3
        # Within a couple of milliseconds -- the sample interval is 2 ms.
        assert float(np.sqrt(np.mean(error**2))) < 2.0, f"{name}: {error.std():.1f} ms"

    def test_never_exceeds_the_strain_limit(self):
        """Structural, not checked-and-hoped-for: no path can violate it."""
        twt = np.arange(0.6, 2.4, 0.002)
        for limit in (5.0, 10.0, 15.0, 25.0):
            strain = strain_limit_for(limit)
            seismic, synthetic = warped_pair(
                0.040 * (twt - twt[0]) / (twt[-1] - twt[0])
            )
            warp = dynamic_warp(seismic, synthetic, strain_limit=strain, max_shift_s=0.060)
            assert warp.max_strain() <= strain * 1.001, (
                f"strain {warp.max_strain():.4f} exceeds limit {strain:.4f}"
            )

    def test_strain_limit_holds_at_the_trace_ends(self):
        """The ends are where a control-point scheme leaks, and nobody looks."""
        twt = np.arange(0.6, 2.4, 0.002)
        seismic, synthetic = warped_pair(0.030 * (twt - twt[0]) / (twt[-1] - twt[0]))
        warp = dynamic_warp(seismic, synthetic, strain_limit=STRAIN, max_shift_s=0.060)
        strain = np.abs(warp.strain())
        assert strain[:20].max() <= STRAIN * 1.001
        assert strain[-20:].max() <= STRAIN * 1.001

    def test_does_not_taper_to_zero_at_the_ends(self):
        """Out-of-range comparisons must be neutral, not scored as bad.

        Scoring them badly makes large shifts impossible near the ends, so the
        warp invents a taper back to zero that the data never asked for.
        """
        twt = np.arange(0.6, 2.4, 0.002)
        seismic, synthetic = warped_pair(np.full(twt.size, 0.012))
        warp = dynamic_warp(seismic, synthetic, strain_limit=STRAIN, max_shift_s=0.060)
        assert warp.shift_s[-1] * 1e3 == pytest.approx(12.0, abs=2.0)
        assert warp.shift_s[0] * 1e3 == pytest.approx(12.0, abs=2.0)

    def test_envelope_pass_survives_a_polarity_flip(self):
        twt = np.arange(0.6, 2.4, 0.002)
        true_shift = np.full(twt.size, 0.010)
        seismic, synthetic = warped_pair(true_shift)
        flipped = Trace(seismic.twt, -seismic.amplitude, "flipped")

        envelope = dynamic_warp(
            flipped, synthetic, strain_limit=STRAIN, max_shift_s=0.060, on_envelope=True
        )
        assert float(np.mean(envelope.shift_s)) * 1e3 == pytest.approx(10.0, abs=3.0)

    def test_anchors_are_honoured(self):
        twt = np.arange(0.6, 2.4, 0.002)
        seismic, synthetic = warped_pair(np.zeros(twt.size))
        warp = dynamic_warp(
            seismic, synthetic, strain_limit=STRAIN, max_shift_s=0.060,
            anchors={1.5: 0.008}, error_smoothing_s=0.0,
        )
        at_anchor = float(warp.shift_at(1.5)) * 1e3
        assert at_anchor == pytest.approx(8.0, abs=2.5)

    def test_saturation_is_reported(self):
        """A warp asked for more than it may give must say it was clipped."""
        twt = np.arange(0.6, 2.4, 0.002)
        # A shift far steeper than a 2% strain limit can follow.
        seismic, synthetic = warped_pair(0.200 * (twt - twt[0]) / (twt[-1] - twt[0]))
        warp = dynamic_warp(
            seismic, synthetic, strain_limit=0.02, max_shift_s=0.200
        )
        assert warp.saturated_fraction() > 0.8
        assert warp.max_strain() <= 0.02 * 1.001


class TestWarpApplication:
    def test_applying_a_warp_keeps_the_depth_axis(self):
        case, calibrated, _ = anomaly_case(0)
        _, fine = auto_warp(
            case.seismic,
            _synthetic_for(case, calibrated),
            strain_limit=STRAIN,
        )
        warped = apply_warp(calibrated, fine)
        assert warped.depth_tvdss is calibrated.depth_tvdss or np.allclose(
            warped.depth_tvdss, calibrated.depth_tvdss
        )

    def test_applying_a_warp_preserves_monotonicity(self):
        """Guaranteed by the strain limit; asserted because it is load-bearing."""
        case, calibrated, _ = anomaly_case(1)
        _, fine = auto_warp(
            case.seismic, _synthetic_for(case, calibrated), strain_limit=STRAIN
        )
        warped = apply_warp(calibrated, fine)  # TimeDepth raises if not monotonic
        assert np.all(np.diff(warped.twt) > 0)


class TestAutoTie:
    """The headline property: the warp never degrades the time-depth."""

    @pytest.mark.parametrize("seed", range(12))
    def test_never_makes_the_time_depth_worse(self, seed):
        case, calibrated, truth = anomaly_case(seed)
        result = auto_tie(
            case.seismic, calibrated, case.log_dt_us_per_m, case.log_rho_g_cm3
        )

        before = rms_ms(result.before_warp.time_depth.twt - truth)
        after = rms_ms(result.result.time_depth.twt - truth)

        assert after <= before + 0.5, (
            f"seed {seed}: warp degraded the time-depth {before:.2f} -> {after:.2f} ms "
            f"rms while reporting correlation "
            f"{result.result.metrics.correlation:.3f} "
            f"(was {result.before_warp.metrics.correlation:.3f}). "
            f"accepted={result.warp_accepted}"
        )

    def test_usually_improves_a_localised_anomaly(self):
        """Declining to act is acceptable; failing to act on most cases is not."""
        improved = 0
        for seed in range(12):
            case, calibrated, truth = anomaly_case(seed)
            result = auto_tie(
                case.seismic, calibrated, case.log_dt_us_per_m, case.log_rho_g_cm3
            )
            before = rms_ms(result.before_warp.time_depth.twt - truth)
            after = rms_ms(result.result.time_depth.twt - truth)
            if after < before - 0.3:
                improved += 1
        assert improved >= 7, f"only {improved}/12 improved"

    def test_rejected_warps_leave_the_tie_untouched(self):
        case, calibrated, _ = anomaly_case(1)
        result = auto_tie(
            case.seismic, calibrated, case.log_dt_us_per_m, case.log_rho_g_cm3,
            max_saturated_fraction=0.0,  # reject everything
        )
        assert not result.warp_accepted
        assert result.result is result.before_warp
        assert result.correlation_gain == 0.0
        assert "clipped" in result.rejection_reason or "pinned" in result.rejection_reason

    def test_a_clipped_warp_is_rejected_even_though_it_passes_the_guardrail(self):
        """The case the guardrail alone cannot catch.

        A warp clipped by the strain limit passes the velocity check *because* it
        was clipped to a passing value. Only the saturation test distinguishes it
        from a warp that solved the problem.
        """
        case = make_case(seed=3, static_s=0.0, drift_amplitude=0.03)
        uncalibrated = integrate_sonic(
            case.log_depth_tvdss, case.log_dt_us_per_m,
            ShallowModel(replacement_velocity=1900.0),
        )
        result = auto_tie(
            case.seismic, uncalibrated, case.log_dt_us_per_m, case.log_rho_g_cm3
        )
        assert result.guardrail.passed, "this case is meant to pass the velocity check"
        assert not result.warp_accepted, "but it must still be rejected as clipped"
        assert result.warp.saturated_fraction() > 0.5

    def test_declines_a_warp_that_buys_nothing(self):
        """A well-calibrated tie has no residual for a warp to remove."""
        case = make_case(seed=2, static_s=0.008, drift_amplitude=0.02)
        raw = integrate_sonic(
            case.log_depth_tvdss, case.log_dt_us_per_m,
            ShallowModel(replacement_velocity=1900.0),
        )
        calibrated = calibrate_to_checkshots(
            raw, case.checkshot_depth_tvdss, case.checkshot_twt
        ).time_depth
        result = auto_tie(
            case.seismic, calibrated, case.log_dt_us_per_m, case.log_rho_g_cm3
        )
        assert not result.warp_accepted
        assert result.result.metrics.correlation > 0.95

    def test_summary_is_compact_and_json_safe(self):
        import json

        case, calibrated, _ = anomaly_case(0)
        result = auto_tie(
            case.seismic, calibrated, case.log_dt_us_per_m, case.log_rho_g_cm3
        )
        encoded = json.dumps(result.summary())
        assert len(encoded) < 6000
        assert "warp_accepted" in result.summary()
        assert isinstance(result.verdict(), str)


def _synthetic_for(case, time_depth):
    """The synthetic a tie would build from this model -- for warp-only tests."""
    from swt.petro.elastic import acoustic_impedance
    from swt.synth.convolve import synthesise
    from swt.synth.resample import reflectivity_in_time
    from swt.units import slowness_to_velocity

    impedance = acoustic_impedance(
        slowness_to_velocity(case.log_dt_us_per_m), case.log_rho_g_cm3
    )
    reflectivity = reflectivity_in_time(
        time_depth.twt, impedance, case.seismic.dt,
        t_start=float(case.seismic.twt[0]), t_end=float(case.seismic.twt[-1]),
    )
    return synthesise(reflectivity, ricker(28.0, case.seismic.dt))


class TestWarpDeclinesWhenItShould:
    """The regime where warping is most dangerous: a tie that is already good.

    With little left to fix, a warp can still buy a hundredth or two of
    correlation by stretching locally onto an artefact -- typically a spurious
    event that a cycle skip put into the synthetic. Every check passes: the
    velocity claim is admissible, the warp is nowhere near its strain limit, the
    correlation goes up. And the time-depth gets worse. This is the case that
    sets `require_improvement`.
    """

    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("cycle_skips", [0, 2])
    def test_never_degrades_a_well_calibrated_tie(self, seed, cycle_skips):
        case = make_case(
            seed=seed, static_s=0.012, wavelet_phase_deg=30.0, signal_to_noise=8.0,
            n_cycle_skips=cycle_skips, drift_amplitude=0.03, checkshot_spacing=150.0,
        )
        raw = integrate_sonic(
            case.log_depth_tvdss, case.log_dt_us_per_m,
            ShallowModel(replacement_velocity=1900.0),
        )
        calibrated = calibrate_to_checkshots(
            raw, case.checkshot_depth_tvdss, case.checkshot_twt
        ).time_depth
        truth = case.true_twt_at_log() + 0.012

        result = auto_tie(
            case.seismic, calibrated, case.log_dt_us_per_m, case.log_rho_g_cm3
        )
        before = rms_ms(result.before_warp.time_depth.twt - truth)
        after = rms_ms(result.result.time_depth.twt - truth)

        assert after <= before + 0.3, (
            f"seed {seed}, {cycle_skips} skips: warp degraded a good tie "
            f"{before:.2f} -> {after:.2f} ms rms for "
            f"{result.correlation_gain:+.3f} correlation"
        )

    def test_a_small_correlation_gain_is_not_worth_a_warp(self):
        """The threshold's rationale, asserted rather than only documented."""
        case = make_case(
            seed=3, static_s=0.012, wavelet_phase_deg=30.0, signal_to_noise=8.0,
            n_cycle_skips=2, drift_amplitude=0.03, checkshot_spacing=150.0,
        )
        raw = integrate_sonic(
            case.log_depth_tvdss, case.log_dt_us_per_m,
            ShallowModel(replacement_velocity=1900.0),
        )
        calibrated = calibrate_to_checkshots(
            raw, case.checkshot_depth_tvdss, case.checkshot_twt
        ).time_depth

        # This warp gains ~0.035 correlation and costs ~3.5 ms of time-depth.
        permissive = auto_tie(
            case.seismic, calibrated, case.log_dt_us_per_m, case.log_rho_g_cm3,
            require_improvement=0.001,
        )
        assert permissive.warp_accepted
        assert permissive.correlation_gain < 0.05

        default = auto_tie(
            case.seismic, calibrated, case.log_dt_us_per_m, case.log_rho_g_cm3
        )
        assert not default.warp_accepted
        assert "correlation" in default.rejection_reason
