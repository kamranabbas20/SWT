"""Tests for ensemble uncertainty.

The test that matters is **coverage**. An uncertainty estimate is only worth
reporting if it is calibrated: a corridor claiming 80% must contain the truth
about 80% of the time. One that claims 80% and delivers 40% is not conservative
or approximate, it is a false statement, and it is worse than reporting no
uncertainty at all because it will be believed.

These tests are slower than the rest of the suite because each one runs the whole
pipeline many times. That is inherent -- an ensemble is not cheap, and testing it
on a single run would test nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from swt.forward import make_case
from swt.uq.ensemble import Ensemble, Member, run_ensemble


def case_and_ensemble(seed: int, n_members: int = 10, **kwargs):
    case = make_case(
        seed=seed, static_s=0.012, wavelet_phase_deg=30.0, signal_to_noise=8.0,
        n_cycle_skips=2, drift_amplitude=0.03,
    )
    truth = case.true_twt_at_log() + 0.012
    ensemble = run_ensemble(
        case.seismic, case.log_depth_tvdss, case.log_dt_us_per_m, case.log_rho_g_cm3,
        case.checkshot_depth_tvdss, case.checkshot_twt,
        n_members=n_members, seed=100 + seed, truth_twt=truth, **kwargs,
    )
    return case, ensemble, truth


@pytest.fixture(scope="module")
def coverage_ensembles():
    """Six cases, built once. Each is a full pipeline run per member."""
    return [case_and_ensemble(seed, n_members=8)[1] for seed in range(6)]


class TestCoverage:
    """Is the corridor calibrated? Nothing else about a UQ system matters more."""

    def test_corridor_is_calibrated_across_cases(self, coverage_ensembles):
        coverages = [e.coverage() for e in coverage_ensembles]

        mean = float(np.mean(coverages))
        # Nominal 0.80. Wide bounds because six cases of eight members is a
        # small sample -- but a mean outside this range means the corridor is
        # systematically lying in one direction or the other.
        assert 0.60 <= mean <= 0.97, f"P10-P90 corridor covers {mean:.2f}, nominal 0.80"

    def test_most_cases_are_substantially_covered(self, coverage_ensembles):
        """Shared bias can defeat any ensemble; it must be the exception."""
        covered = sum(1 for e in coverage_ensembles if e.coverage() > 0.5)
        assert covered >= 4, f"only {covered}/6 cases half-covered"

    def test_a_wider_band_covers_more(self):
        _, ensemble, _ = case_and_ensemble(2, n_members=10)
        narrow = ensemble.coverage(40.0, 60.0)
        wide = ensemble.coverage(2.0, 98.0)
        assert wide >= narrow

    def test_real_data_has_no_coverage_to_report(self):
        case = make_case(seed=0)
        ensemble = run_ensemble(
            case.seismic, case.log_depth_tvdss, case.log_dt_us_per_m,
            case.log_rho_g_cm3, case.checkshot_depth_tvdss, case.checkshot_twt,
            n_members=4, seed=0,
        )
        assert ensemble.coverage() is None
        assert "coverage_vs_truth" not in ensemble.summary()


class TestResolutionFloor:
    """The floor exists because the ensemble cannot see its own blind spots."""

    def test_floor_widens_an_over_tight_corridor(self):
        _, ensemble, _ = case_and_ensemble(8, n_members=8)
        ensemble.resolution_floor_s = 0.005  # 5 ms, far wider than any spread
        band = ensemble.corridor()
        half_width = (band["high"] - band["low"]) / 2.0
        assert np.all(half_width >= 0.005 - 1e-12)

    def test_floor_never_narrows_a_wide_corridor(self):
        _, ensemble, _ = case_and_ensemble(0, n_members=8)
        raw = ensemble.ensemble_half_width_ms()
        ensemble.resolution_floor_s = 0.0
        unfloored = ensemble.width_ms() / 2.0
        assert np.allclose(unfloored, raw)

        ensemble.resolution_floor_s = 0.001
        floored = ensemble.width_ms() / 2.0
        assert np.all(floored >= unfloored - 1e-9)

    def test_raw_spread_is_reported_separately(self):
        """The floor must never be mistaken for measured agreement."""
        _, ensemble, _ = case_and_ensemble(8, n_members=8)
        summary = ensemble.summary()
        assert "resolution_floor_ms" in summary
        assert "ensemble_half_width_ms" in summary
        assert summary["ensemble_half_width_ms"]["median"] <= (
            summary["corridor_width_ms"]["median"] / 2.0 + 1e-6
        )

    def test_floor_defaults_to_half_a_sample(self):
        case, ensemble, _ = case_and_ensemble(1, n_members=4)
        assert ensemble.resolution_floor_s == pytest.approx(case.seismic.dt / 2.0)


class TestMultimodality:
    def test_a_consistent_ensemble_is_unimodal(self):
        _, ensemble, _ = case_and_ensemble(3, n_members=10)
        assert not ensemble.is_multimodal
        assert len(ensemble.modes()) == 1

    def test_a_split_ensemble_is_detected_and_not_averaged(self):
        """Two alignments a cycle apart must be reported as two, not as a mean."""
        depth = np.linspace(1000.0, 2000.0, 101)
        members = []
        for i in range(10):
            shift = 0.0 if i < 6 else 0.036  # one 28 Hz period apart
            members.append(
                Member(draw=None, twt=0.8 + depth * 1e-4 + shift,
                       correlation=0.9, total_shift_s=shift, phase_deg=0.0)
            )
        ensemble = Ensemble(depth_tvdss=depth, members=members)

        found = ensemble.modes()
        assert len(found) == 2
        assert ensemble.is_multimodal
        assert {m["members"] for m in found} == {6, 4}
        assert "AMBIGUOUS" in ensemble.verdict()
        assert "not the average" in ensemble.verdict()

    def test_a_lone_outlier_does_not_make_an_ensemble_ambiguous(self):
        depth = np.linspace(1000.0, 2000.0, 101)
        members = [
            Member(draw=None, twt=0.8 + depth * 1e-4 + (0.040 if i == 0 else 0.0),
                   correlation=0.9, total_shift_s=(0.040 if i == 0 else 0.0),
                   phase_deg=0.0)
            for i in range(20)
        ]
        ensemble = Ensemble(depth_tvdss=depth, members=members)
        assert len(ensemble.modes()) == 2      # the split is seen
        assert not ensemble.is_multimodal      # but 1 in 20 is not an ambiguity


class TestReporting:
    def test_summary_is_compact_and_json_safe(self):
        _, ensemble, _ = case_and_ensemble(0, n_members=8)
        encoded = json.dumps(ensemble.summary())
        assert len(encoded) < 6000

    def test_per_horizon_gives_the_depth_conversion_answer(self):
        _, ensemble, _ = case_and_ensemble(0, n_members=8)
        horizons = ensemble.per_horizon([1200.0, 1800.0, 2400.0])
        assert len(horizons) == 3
        for horizon in horizons:
            assert horizon["p_low_ms"] <= horizon["twt_ms"] <= horizon["p_high_ms"]
            assert horizon["half_width_ms"] > 0
        # Uncertainty should not shrink with depth.
        assert horizons[-1]["half_width_ms"] >= horizons[0]["half_width_ms"] * 0.5

    def test_verdict_quotes_a_tolerance_a_human_can_use(self):
        _, ensemble, _ = case_and_ensemble(0, n_members=8)
        verdict = ensemble.verdict()
        assert "ms" in verdict and "+/-" in verdict

    def test_failed_members_are_recorded_not_raised(self):
        """A draw that fails is data about the tie, not a crash."""
        case = make_case(seed=0)
        ensemble = run_ensemble(
            case.seismic, case.log_depth_tvdss, case.log_dt_us_per_m,
            case.log_rho_g_cm3, case.checkshot_depth_tvdss, case.checkshot_twt,
            n_members=4, seed=0,
        )
        assert isinstance(ensemble.failures, list)
        assert ensemble.n + len(ensemble.failures) == 4

    def test_empty_ensemble_reports_rather_than_crashes(self):
        ensemble = Ensemble(depth_tvdss=np.linspace(1000.0, 2000.0, 11), members=[])
        summary = ensemble.summary()
        assert summary["n_members"] == 0
        assert "error" in summary
        assert "no uncertainty" in ensemble.verdict()


class TestEnsembleMechanics:
    def test_cycle_probe_changes_the_starting_point_not_the_bookkeeping(self):
        """Members must report shifts comparable to each other.

        The start offset is applied to the model *before* the tie, so the tie's
        own shift is relative to the offset model. Getting that sign wrong makes
        every member look like it landed on a different cycle and turns the
        multimodality test into a permanent false alarm.
        """
        _, ensemble, _ = case_and_ensemble(3, n_members=10)
        offsets = np.array([m.draw.start_offset_s for m in ensemble.members]) * 1e3
        shifts = np.array([m.total_shift_s for m in ensemble.members]) * 1e3
        assert np.ptp(offsets) > 20.0, "the cycle probe is not varying the start"
        # Reported shifts must be far tighter than the offsets they started from.
        assert np.ptp(shifts) < np.ptp(offsets)

    def test_more_members_do_not_shift_the_answer(self):
        _, small, _ = case_and_ensemble(2, n_members=6)
        _, large, _ = case_and_ensemble(2, n_members=14)
        difference = np.abs(small.corridor()["mid"] - large.corridor()["mid"]) * 1e3
        assert float(np.median(difference)) < 3.0

    def test_rejects_a_trivial_ensemble(self):
        case = make_case(seed=0)
        with pytest.raises(ValueError, match="at least two"):
            run_ensemble(
                case.seismic, case.log_depth_tvdss, case.log_dt_us_per_m,
                case.log_rho_g_cm3, n_members=1,
            )


class TestRobustSpread:
    """A percentile corridor must not be reported beside a non-robust sigma.

    A single member landing on a different cycle sits far from the rest. The
    percentile band ignores it; the ordinary standard deviation does not. Before
    this was fixed the per-horizon table read `half_width_ms: 1.98` next to
    `std_ms: 4.43` -- two numbers differing by a factor of two, with nothing to
    tell a reader which one to believe.
    """

    def test_a_lone_outlier_does_not_inflate_the_reported_spread(self):
        depth = np.linspace(1000.0, 2000.0, 51)
        base = 0.8 + depth * 1e-4
        members = [
            Member(draw=None, twt=base + (0.014 if i == 0 else 0.0005 * (i % 3)),
                   correlation=0.9, total_shift_s=0.0, phase_deg=0.0)
            for i in range(10)
        ]
        ensemble = Ensemble(depth_tvdss=depth, members=members, resolution_floor_s=0.0)

        report = ensemble.uncertainty_at(1500.0)

        # The 14 ms outlier must not set the reported spread...
        assert report["robust_std_ms"] < 3.0
        # ...but it must be reported.
        assert report["n_outlier_members"] == 1

    def test_clean_ensembles_report_no_outliers(self):
        rng = np.random.default_rng(0)
        depth = np.linspace(1000.0, 2000.0, 51)
        base = 0.8 + depth * 1e-4
        members = [
            Member(draw=None, twt=base + 0.001 * rng.standard_normal(),
                   correlation=0.9, total_shift_s=0.0, phase_deg=0.0)
            for _ in range(20)
        ]
        ensemble = Ensemble(depth_tvdss=depth, members=members, resolution_floor_s=0.0)
        report = ensemble.uncertainty_at(1500.0)
        assert report["n_outlier_members"] <= 1
        assert report["robust_std_ms"] > 0

    def test_the_two_spread_measures_agree_on_clean_data(self):
        """Half-width should be ~1.28x the robust sigma for Gaussian members."""
        rng = np.random.default_rng(1)
        depth = np.linspace(1000.0, 2000.0, 51)
        base = 0.8 + depth * 1e-4
        members = [
            Member(draw=None, twt=base + 0.002 * rng.standard_normal(),
                   correlation=0.9, total_shift_s=0.0, phase_deg=0.0)
            for _ in range(60)
        ]
        ensemble = Ensemble(depth_tvdss=depth, members=members, resolution_floor_s=0.0)
        report = ensemble.uncertainty_at(1500.0)
        ratio = report["half_width_ms"] / report["robust_std_ms"]
        assert 0.9 < ratio < 1.8, f"half-width / robust sigma = {ratio:.2f}"
