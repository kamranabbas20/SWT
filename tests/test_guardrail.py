"""Tests for the velocity guardrail -- written before the solver it guards.

The guardrail is the only thing standing between a warping algorithm and a
beautifully correlated fiction, so it is tested against hand-built warps whose
implied velocity change is known analytically, not against solver output. If it
only worked on warps the solver happens to produce, it would not be a guardrail.
"""

from __future__ import annotations

import numpy as np
import pytest

from swt.tie.guardrail import check_warp, strain_limit_for
from swt.timedepth.model import TimeDepth


def constant_velocity_model(velocity: float = 2500.0, n: int = 2001) -> TimeDepth:
    depth = np.linspace(1000.0, 3000.0, n)
    twt = 0.8 + 2.0 * (depth - depth[0]) / velocity
    return TimeDepth(depth, twt, "test")


def stretched(model: TimeDepth, factor: float) -> TimeDepth:
    """Scale every time increment by ``factor`` -- a uniform stretch."""
    increments = np.diff(model.twt) * factor
    twt = np.concatenate([[model.twt[0]], model.twt[0] + np.cumsum(increments)])
    return TimeDepth(model.depth_tvdss, twt, "stretched")


class TestVelocityArithmetic:
    def test_a_uniform_stretch_implies_the_inverse_velocity_change(self):
        """Stretching time by 1+r slows the rock by r/(1+r), exactly."""
        model = constant_velocity_model()
        for r in (0.05, 0.10, 0.25):
            report = check_warp(model, stretched(model, 1.0 + r), limit_percent=100.0)
            expected = 100.0 * (-r / (1.0 + r))
            assert report.max_change_percent == pytest.approx(abs(expected), rel=1e-6)

    def test_squeezing_speeds_the_rock_up(self):
        model = constant_velocity_model()
        report = check_warp(model, stretched(model, 0.9), limit_percent=100.0)
        assert np.all(report.change_percent > 0)
        assert report.max_change_percent == pytest.approx(100.0 * (1 / 0.9 - 1), rel=1e-6)

    def test_no_warp_implies_no_velocity_change(self):
        model = constant_velocity_model()
        report = check_warp(model, model, limit_percent=15.0)
        assert report.passed
        assert report.max_change_percent == pytest.approx(0.0, abs=1e-9)
        assert report.violations == []

    def test_a_pure_bulk_shift_is_not_a_velocity_claim(self):
        """Moving every sample by the same amount changes no interval velocity."""
        model = constant_velocity_model()
        report = check_warp(model, model.shifted(0.030), limit_percent=1.0)
        assert report.passed
        assert report.max_change_percent == pytest.approx(0.0, abs=1e-9)


class TestAdmissibility:
    def test_accepts_a_modest_stretch(self):
        model = constant_velocity_model()
        report = check_warp(model, stretched(model, 1.05), limit_percent=15.0)
        assert report.passed
        assert "admissible" in report.verdict()

    def test_rejects_an_extravagant_stretch(self):
        model = constant_velocity_model()
        report = check_warp(model, stretched(model, 1.40), limit_percent=15.0)
        assert not report.passed
        assert "REJECTED" in report.verdict()
        assert "cycle skip" in report.verdict()

    def test_localises_the_offending_interval(self):
        """The report must say *where*, not just that something is wrong."""
        model = constant_velocity_model()
        increments = np.diff(model.twt).copy()
        bad = (model.depth_tvdss[:-1] > 1800.0) & (model.depth_tvdss[:-1] < 2000.0)
        increments[bad] *= 1.5
        warped = TimeDepth(
            model.depth_tvdss,
            np.concatenate([[model.twt[0]], model.twt[0] + np.cumsum(increments)]),
            "locally stretched",
        )

        report = check_warp(model, warped, limit_percent=15.0)
        assert not report.passed
        worst = report.worst(1)[0]
        assert 1780.0 < worst.start_depth < 1820.0
        assert 1980.0 < worst.end_depth < 2020.0
        # Only that interval is at fault; the rest of the log is clean.
        assert report.fraction_violating < 0.2

    def test_tolerated_fraction_allows_a_known_bad_zone(self):
        model = constant_velocity_model()
        increments = np.diff(model.twt).copy()
        increments[(model.depth_tvdss[:-1] > 1900.0) & (model.depth_tvdss[:-1] < 1950.0)] *= 1.5
        warped = TimeDepth(
            model.depth_tvdss,
            np.concatenate([[model.twt[0]], model.twt[0] + np.cumsum(increments)]),
            "locally stretched",
        )
        assert not check_warp(model, warped, 15.0).passed
        assert check_warp(model, warped, 15.0, tolerated_fraction=0.05).passed

    def test_rejects_a_time_inverting_warp_outright(self):
        """A non-monotonic warp is not a bad tie, it is not a tie at all.

        TimeDepth cannot represent one, so the check is that the guardrail
        refuses to be handed the pieces of one rather than quietly accepting it.
        """
        model = constant_velocity_model()
        increments = np.diff(model.twt).copy()
        increments[500:520] *= -1.0
        with pytest.raises(Exception):
            TimeDepth(
                model.depth_tvdss,
                np.concatenate([[model.twt[0]], model.twt[0] + np.cumsum(increments)]),
                "inverted",
            )

    def test_refuses_models_on_different_depth_axes(self):
        model = constant_velocity_model()
        other = constant_velocity_model(n=1001)
        with pytest.raises(ValueError, match="different depth axes"):
            check_warp(model, other)


class TestStrainCoupling:
    """The solver's constraint and the guardrail's limit must be one number."""

    def test_strain_limit_binds_on_the_squeeze_side(self):
        """Squeezing at the strain limit lands exactly on the velocity limit.

        Stretch and squeeze are not symmetric in velocity even though the strain
        bound is: at strain r, stretching implies r/(1+r) and squeezing r/(1-r).
        The solver's bound must be set by the worse (squeeze) side, or it can
        squeeze past the guardrail and have its work thrown away.
        """
        model = constant_velocity_model()
        for limit in (5.0, 10.0, 15.0, 30.0):
            r = strain_limit_for(limit)
            squeeze = check_warp(model, stretched(model, 1.0 - r), limit_percent=limit)
            assert squeeze.max_change_percent == pytest.approx(limit, rel=1e-6)
            assert squeeze.passed

            # The stretch side at the same strain is strictly inside the limit.
            stretch = check_warp(model, stretched(model, 1.0 + r), limit_percent=limit)
            assert stretch.max_change_percent < limit
            assert stretch.passed

    def test_a_warp_within_the_strain_limit_always_passes(self):
        limit = 15.0
        r = strain_limit_for(limit)
        model = constant_velocity_model()
        for factor in (1.0 + 0.99 * r, 1.0, 1.0 - 0.99 * r):
            assert check_warp(model, stretched(model, factor), limit_percent=limit).passed

    def test_rejects_nonsense_limits(self):
        with pytest.raises(ValueError):
            strain_limit_for(0.0)
        with pytest.raises(ValueError):
            strain_limit_for(150.0)
        with pytest.raises(ValueError):
            check_warp(constant_velocity_model(), constant_velocity_model(), limit_percent=-1)
