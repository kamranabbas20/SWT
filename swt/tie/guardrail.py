"""The velocity guardrail: is a proposed warp geologically admissible?

This module exists because of a single asymmetry. A bulk shift has one free
parameter and a phase rotation has one; a warp has **one per sample**. Given
that freedom, a warping algorithm can align almost any synthetic with almost any
seismic and report a superb correlation, because a high correlation is precisely
what it optimised for. Correlation therefore stops being evidence at exactly the
moment warping is introduced.

What remains as evidence is physics. A warp is not a free reparameterisation of
time -- it is a *claim about velocity*. Stretching the synthetic by 20% over an
interval asserts that the true interval velocity there is 20% different from what
the sonic measured. Sometimes that is right (invasion, dispersion, a bad hole).
At 40% it is almost never right, and a tie that needs it is telling you something
is wrong upstream -- a cycle skip, a mis-picked checkshot, the wrong trace, the
wrong well.

So every warp is converted back into the velocity change it implies and checked
against what rock can actually do. A warp that fails this check is rejected even
when it correlates better than the one that passes, and the rejection is reported
in velocity terms, because that is the language in which a geophysicist can
judge whether the software or the data is at fault.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..timedepth.model import TimeDepth, is_monotonic


@dataclass(frozen=True)
class Violation:
    """One depth interval where a warp implies an implausible velocity change."""

    start_depth: float
    end_depth: float
    change_percent: float

    def summary(self) -> dict:
        return {
            "interval_m": [round(self.start_depth, 1), round(self.end_depth, 1)],
            "velocity_change_percent": round(self.change_percent, 1),
        }


@dataclass(frozen=True)
class GuardrailReport:
    """Whether a warp is admissible, and the evidence either way."""

    passed: bool
    max_change_percent: float
    mean_abs_change_percent: float
    limit_percent: float
    violations: list[Violation]
    fraction_violating: float
    monotonic: bool
    depth: np.ndarray
    change_percent: np.ndarray

    def summary(self) -> dict:
        return {
            "passed": self.passed,
            "limit_percent": self.limit_percent,
            "max_velocity_change_percent": round(self.max_change_percent, 1),
            "mean_abs_velocity_change_percent": round(self.mean_abs_change_percent, 2),
            "fraction_of_log_violating": round(self.fraction_violating, 4),
            "monotonic": self.monotonic,
            "n_violating_intervals": len(self.violations),
            "worst_intervals": [v.summary() for v in self.worst(5)],
            "verdict": self.verdict(),
        }

    def worst(self, n: int = 5) -> list[Violation]:
        return sorted(self.violations, key=lambda v: -abs(v.change_percent))[:n]

    def verdict(self) -> str:
        """One sentence an interpreter can act on."""
        if not self.monotonic:
            return (
                "REJECTED: the warp inverts time with depth. This is not a tie, it is "
                "a reordering of the earth."
            )
        if self.passed:
            return (
                f"Warp is admissible: it implies at most {self.max_change_percent:.1f}% "
                f"velocity change against the sonic (limit {self.limit_percent:.0f}%), "
                f"{self.mean_abs_change_percent:.1f}% on average."
            )
        worst = self.worst(1)[0]
        return (
            f"REJECTED: the warp implies up to {self.max_change_percent:.1f}% velocity "
            f"change against the sonic, over the {self.limit_percent:.0f}% limit -- worst "
            f"at {worst.start_depth:.0f}-{worst.end_depth:.0f} m. A tie needing this much "
            "stretch is usually a symptom, not a solution: check that interval for a cycle "
            "skip or bad hole, check the checkshots, and check the trace is at the right well."
        )


def check_warp(
    original: TimeDepth,
    warped: TimeDepth,
    limit_percent: float = 15.0,
    tolerated_fraction: float = 0.0,
) -> GuardrailReport:
    """Convert a warp into implied velocity change and judge it.

    Both models must share a depth axis -- a warp changes time, never depth, so
    a differing axis means something upstream is wrong.

    The implied interval velocity change is

    ``(v_warped - v_original) / v_original``

    computed interval by interval. Because ``v = 2 dz / dt`` on a shared depth
    axis, this reduces to the ratio of time gradients: the warp's local stretch
    *is* the velocity claim, which is the whole point.

    Parameters
    ----------
    limit_percent:
        Largest velocity change any interval may imply. The default of 15% is
        deliberately generous for real rock and still far below what an
        unconstrained warp will reach for.
    tolerated_fraction:
        Fraction of the log allowed to exceed the limit before the warp is
        rejected outright. Defaults to zero -- one bad interval fails the warp,
        because one bad interval is usually where the real problem is. Raise it
        to accept a warp that is sound apart from a known bad zone.
    """
    if limit_percent <= 0:
        raise ValueError("limit_percent must be positive")
    if not 0.0 <= tolerated_fraction < 1.0:
        raise ValueError("tolerated_fraction must lie in [0, 1)")
    if original.depth_tvdss.shape != warped.depth_tvdss.shape or not np.allclose(
        original.depth_tvdss, warped.depth_tvdss
    ):
        raise ValueError(
            "the original and warped time-depth models are on different depth axes; "
            "a warp changes time, never depth"
        )

    midpoints, v_original = original.interval_velocity()
    _, v_warped = warped.interval_velocity()

    with np.errstate(divide="ignore", invalid="ignore"):
        change = 100.0 * (v_warped - v_original) / v_original
    change = np.where(np.isfinite(change), change, 0.0)

    # A warp sitting exactly on the limit is admissible -- that is what a limit
    # means -- so compare with a relative tolerance. Without it a solver whose
    # strain bound is derived from this very limit (see `strain_limit_for`) has
    # its output rejected by floating-point noise in the last decimal place.
    threshold = limit_percent * (1.0 + 1e-9) + 1e-12
    exceeded = np.abs(change) > threshold
    violations = [
        Violation(
            start_depth=float(original.depth_tvdss[lo]),
            end_depth=float(original.depth_tvdss[min(hi, original.depth_tvdss.size - 1)]),
            change_percent=float(change[lo:hi][np.argmax(np.abs(change[lo:hi]))]),
        )
        for lo, hi in _runs(exceeded)
    ]

    fraction = float(np.count_nonzero(exceeded)) / max(exceeded.size, 1)
    monotonic = is_monotonic(warped.twt)

    return GuardrailReport(
        passed=monotonic and fraction <= tolerated_fraction,
        max_change_percent=float(np.max(np.abs(change))) if change.size else 0.0,
        mean_abs_change_percent=float(np.mean(np.abs(change))) if change.size else 0.0,
        limit_percent=float(limit_percent),
        violations=violations,
        fraction_violating=fraction,
        monotonic=monotonic,
        depth=midpoints,
        change_percent=change,
    )


def strain_limit_for(limit_percent: float) -> float:
    """The symmetric warp strain limit that keeps velocity change inside a bound.

    Velocity is inversely proportional to the time gradient on a fixed depth
    axis, so a warp with local strain ``r`` (that is, ``dt_warped/dt_original =
    1 + r``) implies a velocity change of ``1/(1 + r) - 1``.

    **Stretch and squeeze are not symmetric in velocity, though the strain bound
    is.** For a solver constrained to ``|r| <= R``:

    ===========  ===================  ==========================
    strain       velocity change      at R = 0.15
    ===========  ===================  ==========================
    ``r = +R``   ``-R / (1 + R)``     -13.0%  (stretch, slower)
    ``r = -R``   ``+R / (1 - R)``     +17.6%  (squeeze, faster)
    ===========  ===================  ==========================

    The squeeze side is therefore the binding one, and solving ``R / (1 - R) =
    L`` for it gives ``R = L / (1 + L)``. Using the stretch side instead would
    let the solver squeeze past the guardrail's limit and have its work rejected
    -- the asymmetry is small but it lands entirely on the side that fails.

    Wiring the solver's constraint to the guardrail's limit through this function
    -- rather than setting the two by hand -- is what stops them drifting apart.
    """
    if not 0.0 < limit_percent < 100.0:
        raise ValueError("limit_percent must lie strictly between 0 and 100")
    fraction = limit_percent / 100.0
    return fraction / (1.0 + fraction)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs as half-open ``(start, stop)`` index pairs."""
    if not np.any(mask):
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[::2].tolist(), changes[1::2].tolist()))
