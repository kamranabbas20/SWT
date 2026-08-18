"""Ensemble uncertainty: a tie is not one answer.

A well tie is reported as a single time-depth curve, and that curve is presented
to whoever does the depth conversion as though it were a measurement. It is not.
It is the output of a dozen judgement calls -- where the wavelet design window
sits, how hard to despike, how finely to honour the checkshots, how much to
upscale -- each of which a competent interpreter could have made differently, and
several of which move the answer by more than the tie's own QC would suggest.

So the deliverable here is a **corridor**, not a curve, and a per-horizon
uncertainty in milliseconds, which is what the person doing the depth conversion
actually needs.

The method is deliberately the cheap one. Perturb the interpreter's choices,
re-run the whole pipeline, and look at the spread of answers. It is not a
posterior -- it makes no claim about the probability of the earth -- and this
module does not pretend otherwise. It measures how much the answer depends on
decisions nobody can make uniquely, which is a smaller question than full
Bayesian inversion and the one that actually changes what a well tie is used for.

**The corridor is tested for coverage**, which is the only thing that makes an
uncertainty estimate worth reporting: on forward-modelled cases where the truth
is known, an 80% corridor must contain the truth about 80% of the time. A
corridor that claims 80% and delivers 40% is not conservative or approximate --
it is a false statement, and worse than reporting no uncertainty at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..logs.condition import despike
from ..petro.elastic import backus_average
from ..timedepth.calibrate import calibrate_to_checkshots
from ..timedepth.integrate import ShallowModel, integrate_sonic
from ..timedepth.model import TimeDepth
from ..tie.iterate import tie
from ..trace import Trace
from ..units import slowness_to_velocity, velocity_to_slowness


@dataclass(frozen=True)
class Draw:
    """One sampled set of interpreter choices.

    Every field is a decision a human makes and cannot make uniquely. The ranges
    are meant to span what two competent interpreters would plausibly choose --
    not the full space of legal values, which would overstate the uncertainty as
    surely as a single run understates it.
    """

    despike_threshold: float
    knot_spacing_m: float | None
    wavelet_length_s: float
    ridge: float
    backus_length_m: float
    start_offset_s: float
    index: int

    def summary(self) -> dict:
        return {
            "despike_threshold": round(self.despike_threshold, 2),
            "knot_spacing_m": self.knot_spacing_m,
            "wavelet_length_ms": round(self.wavelet_length_s * 1e3, 1),
            "ridge": round(self.ridge, 5),
            "backus_length_m": round(self.backus_length_m, 1),
            "start_offset_ms": round(self.start_offset_s * 1e3, 1),
        }


@dataclass
class Member:
    """One completed ensemble run."""

    draw: Draw
    twt: np.ndarray
    correlation: float
    total_shift_s: float
    phase_deg: float


@dataclass
class Ensemble:
    """The spread of time-depth curves over sampled interpreter choices."""

    depth_tvdss: np.ndarray
    members: list[Member]
    failures: list[dict] = field(default_factory=list)
    truth_twt: np.ndarray | None = None
    #: Half-width (s) below which the corridor is widened, whatever the members
    #: did.  See :meth:`corridor` for why this is not a fudge.
    resolution_floor_s: float = 0.001

    @property
    def n(self) -> int:
        return len(self.members)

    def stack(self) -> np.ndarray:
        """``(n_members, n_depth)`` array of two-way times."""
        if not self.members:
            raise ValueError("the ensemble is empty; every member failed")
        return np.vstack([m.twt for m in self.members])

    # -- corridors ---------------------------------------------------------

    def corridor(self, lower: float = 10.0, upper: float = 90.0) -> dict:
        """Percentile band of two-way time at every depth.

        Returns arrays keyed ``low``, ``mid``, ``high`` -- the only place in the
        package where arrays cross a public boundary, because a corridor *is* an
        array and there is no useful scalar summary of one. The copilot-facing
        summaries below return scalars.

        **The resolution floor.** The band is widened to at least
        ``resolution_floor_s`` either side of the median, however tightly the
        members agree. This is not a device for making the coverage statistic
        look better; it corrects a real omission. The ensemble varies processing
        *choices*, so it can only see uncertainty those choices create. It never
        varies the seismic's noise realisation, the wavelet's own estimation
        error, or the sample interval, and it therefore cannot represent them --
        a corridor of +/-0.4 ms on 2 ms data is over-precise no matter how well
        the members agree. The floor is a crude acknowledgement of the error
        sources outside the ensemble's reach, and
        :meth:`Ensemble.ensemble_half_width_ms` still reports the raw spread so
        the two are never confused.
        """
        stack = self.stack()
        low = np.percentile(stack, lower, axis=0)
        mid = np.percentile(stack, 50.0, axis=0)
        high = np.percentile(stack, upper, axis=0)

        floor = float(self.resolution_floor_s)
        if floor > 0:
            low = np.minimum(low, mid - floor)
            high = np.maximum(high, mid + floor)

        return {
            "depth_tvdss": self.depth_tvdss,
            "low": low,
            "mid": mid,
            "high": high,
            "lower_percentile": lower,
            "upper_percentile": upper,
            "resolution_floor_ms": floor * 1e3,
        }

    def width_ms(self, lower: float = 10.0, upper: float = 90.0) -> np.ndarray:
        """Reported corridor width in milliseconds, floor included."""
        band = self.corridor(lower, upper)
        return (band["high"] - band["low"]) * 1e3

    def ensemble_half_width_ms(self, lower: float = 10.0, upper: float = 90.0) -> np.ndarray:
        """Raw half-width from the members alone, with no resolution floor.

        Reported alongside the corridor so the two are never conflated. Where
        this is much smaller than the floor, the ensemble is telling you that
        processing choices barely matter here -- which is a statement about the
        choices, not a claim that the tie is accurate to that precision.
        """
        stack = self.stack()
        low, high = np.percentile(stack, [lower, upper], axis=0)
        return (high - low) * 1e3 / 2.0

    def uncertainty_at(self, depth_tvdss: float, lower: float = 10.0,
                       upper: float = 90.0) -> dict:
        """Time and its uncertainty at one depth -- the depth-conversion answer.

        The spread is reported as a **robust** scale (1.4826 x the median absolute
        deviation, which equals the standard deviation for clean Gaussian data)
        rather than as the ordinary standard deviation, and outlier members are
        counted separately.

        The reason is a real inconsistency found in this report. A single member
        that lands on a different cycle sits ten or more milliseconds from the
        rest. The percentile band correctly ignores it, but the standard
        deviation does not -- so the table read
        ``half_width_ms: 1.98, std_ms: 4.43``, two numbers differing by a factor
        of two with nothing to say which to believe. Mixing a percentile corridor
        with a non-robust scale invites the reader to trust the wrong one.

        The outlier count carries the information the standard deviation was
        smuggling in, and carries it legibly: *one member tied elsewhere* is a
        statement someone can act on, where an inflated sigma is not.
        """
        stack = self.stack()
        index = int(np.argmin(np.abs(self.depth_tvdss - float(depth_tvdss))))
        times = stack[:, index]
        low, mid, high = np.percentile(times, [lower, 50.0, upper])

        deviation = np.abs(times - mid)
        robust = 1.4826 * float(np.median(deviation))
        outliers = int(np.count_nonzero(deviation > 3.0 * robust)) if robust > 0 else 0

        return {
            "depth_m": round(float(self.depth_tvdss[index]), 1),
            "twt_ms": round(float(mid) * 1e3, 2),
            "p_low_ms": round(float(low) * 1e3, 2),
            "p_high_ms": round(float(high) * 1e3, 2),
            "half_width_ms": round(float(high - low) * 1e3 / 2.0, 2),
            "robust_std_ms": round(robust * 1e3, 2),
            "n_outlier_members": outliers,
        }

    def per_horizon(self, depths, lower: float = 10.0, upper: float = 90.0) -> list[dict]:
        """Uncertainty at a set of horizon depths, for the depth conversion."""
        return [self.uncertainty_at(d, lower, upper) for d in np.atleast_1d(depths)]

    # -- multimodality -----------------------------------------------------

    def modes(self, gap_ms: float = 8.0) -> list[dict]:
        """Clusters in the ensemble's bulk shift, found by gaps in the sorted values.

        A tie can be genuinely ambiguous: two alignments a cycle apart can both
        look good, and an interpreter starting from a different place lands on a
        different one. Averaging across that ambiguity produces a mean nobody
        believes and a corridor that spans the gap as though every value in
        between were plausible, when in fact none of them is.

        So the split is detected and reported rather than smoothed over. A
        one-dimensional gap-based split is enough here: the modes are separated
        by roughly a wavelet period, which is far larger than the within-mode
        spread, so nothing subtler is needed to see them.
        """
        if not self.members:
            return []
        shifts = np.sort(np.array([m.total_shift_s for m in self.members]) * 1e3)
        splits = np.flatnonzero(np.diff(shifts) > gap_ms)
        groups = np.split(shifts, splits + 1)
        return [
            {
                "shift_ms": round(float(np.median(g)), 2),
                "members": int(g.size),
                "fraction": round(float(g.size) / shifts.size, 3),
                "spread_ms": round(float(g.max() - g.min()), 2),
            }
            for g in groups
        ]

    @property
    def is_multimodal(self) -> bool:
        """True when a second cluster holds a non-trivial share of the ensemble.

        Both a fraction *and* an absolute count are required. On a small ensemble
        a fraction alone is far too easy to trip: with eight members one stray
        draw is 12.5%, which would raise a cycle-skip alarm on every ensemble
        that happened to contain a single bad run. Two members landing together
        is the weakest evidence worth calling an ambiguity.
        """
        found = sorted(self.modes(), key=lambda m: -m["members"])
        return len(found) > 1 and found[1]["members"] >= 2 and found[1]["fraction"] >= 0.1

    # -- coverage ----------------------------------------------------------

    def coverage(self, lower: float = 10.0, upper: float = 90.0) -> float | None:
        """Fraction of depths where the corridor actually contains the truth.

        ``None`` on real data, where there is no truth. The number this returns
        should be close to ``(upper - lower) / 100``; if it is far below, the
        ensemble is not sampling the choices that actually matter, and the
        corridor is a false statement rather than a conservative one.
        """
        if self.truth_twt is None:
            return None
        band = self.corridor(lower, upper)
        inside = (self.truth_twt >= band["low"]) & (self.truth_twt <= band["high"])
        return float(np.mean(inside))

    # -- reporting ---------------------------------------------------------

    def summary(self, lower: float = 10.0, upper: float = 90.0) -> dict:
        """Compact JSON-safe summary -- the shape the copilot tools return."""
        if not self.members:
            return {"n_members": 0, "n_failed": len(self.failures),
                    "error": "every ensemble member failed"}

        width = self.width_ms(lower, upper)
        correlations = np.array([m.correlation for m in self.members])
        shifts = np.array([m.total_shift_s for m in self.members]) * 1e3
        found = self.modes()

        out = {
            "n_members": self.n,
            "n_failed": len(self.failures),
            "band": f"P{lower:g}-P{upper:g}",
            "corridor_width_ms": {
                "median": round(float(np.median(width)), 2),
                "max": round(float(np.max(width)), 2),
                "at_log_top": round(float(width[0]), 2),
                "at_log_base": round(float(width[-1]), 2),
            },
            "resolution_floor_ms": round(self.resolution_floor_s * 1e3, 2),
            "ensemble_half_width_ms": {
                "median": round(float(np.median(self.ensemble_half_width_ms(lower, upper))), 3),
                "max": round(float(np.max(self.ensemble_half_width_ms(lower, upper))), 3),
            },
            "bulk_shift_ms": {
                "median": round(float(np.median(shifts)), 2),
                "spread": round(float(np.max(shifts) - np.min(shifts)), 2),
            },
            "correlation": {
                "median": round(float(np.median(correlations)), 4),
                "min": round(float(np.min(correlations)), 4),
            },
            "multimodal": self.is_multimodal,
            "modes": found,
            "verdict": self.verdict(lower, upper),
        }
        covered = self.coverage(lower, upper)
        if covered is not None:
            out["coverage_vs_truth"] = round(covered, 3)
        return out

    def verdict(self, lower: float = 10.0, upper: float = 90.0) -> str:
        """One sentence for the report and the copilot."""
        if not self.members:
            return "Every ensemble member failed; no uncertainty can be reported."
        width = self.width_ms(lower, upper)
        median = float(np.median(width))
        if self.is_multimodal:
            found = sorted(self.modes(), key=lambda m: -m["fraction"])
            gap = abs(found[0]["shift_ms"] - found[1]["shift_ms"])
            return (
                f"AMBIGUOUS: the ensemble splits into {len(found)} alignments about "
                f"{gap:.0f} ms apart ({found[0]['fraction']:.0%} / "
                f"{found[1]['fraction']:.0%}). This is a cycle-skip ambiguity, not a "
                "spread -- the answer is one of them, not the average. Resolve it with "
                "a checkshot, a formation top, or a marker you trust before using this tie."
            )
        return (
            f"Time-depth is determined to about +/-{median / 2:.1f} ms "
            f"(P{lower:g}-P{upper:g} corridor, {self.n} members), widening to "
            f"+/-{float(np.max(width)) / 2:.1f} ms at worst."
        )


def run_ensemble(
    seismic: Trace,
    depth_tvdss: np.ndarray,
    sonic_us_per_m: np.ndarray,
    density_g_cm3: np.ndarray,
    checkshot_depth: np.ndarray | None = None,
    checkshot_twt: np.ndarray | None = None,
    replacement_velocity: float = 1900.0,
    n_members: int = 24,
    seed: int = 0,
    probe_cycle_ambiguity: bool = True,
    truth_twt: np.ndarray | None = None,
    resolution_floor_s: float | None = None,
    progress=None,
) -> Ensemble:
    """Re-run the tie over sampled interpreter choices.

    Parameters
    ----------
    n_members:
        Ensemble size. 24 is enough for a P10-P90 corridor; percentiles that far
        into the tails of a 24-member sample are noisy, which is one reason the
        default band is P10-P90 rather than P5-P95.
    probe_cycle_ambiguity:
        Start members from time-depth models offset by up to a wavelet period, so
        that if two alignments a cycle apart are both defensible the ensemble
        finds both. Without this the ensemble measures precision around whichever
        alignment the first run happened to reach, and reports a tight corridor
        around a possibly wrong loop -- confident and incorrect, the failure this
        whole package is arranged against.
    truth_twt:
        The true two-way time at each log sample, when known. Enables
        :meth:`Ensemble.coverage`, and is available only for forward-modelled
        cases.
    resolution_floor_s:
        Half-width below which the reported corridor is widened regardless of
        member agreement. Defaults to half the seismic sample interval. See
        :meth:`Ensemble.corridor` for why this exists.
    progress:
        Optional ``callable(done, total)`` for a UI progress bar.
    """
    if n_members < 2:
        raise ValueError("an ensemble needs at least two members")

    rng = np.random.default_rng(seed)
    members: list[Member] = []
    failures: list[dict] = []

    # A period of the dominant frequency, used to size the ambiguity probe.
    period_s = _dominant_period(seismic)

    for index in range(n_members):
        draw = _sample(rng, index, period_s if probe_cycle_ambiguity else 0.0)
        try:
            members.append(
                _run_member(
                    draw, seismic, depth_tvdss, sonic_us_per_m, density_g_cm3,
                    checkshot_depth, checkshot_twt, replacement_velocity,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a failed draw is data, not a crash
            failures.append({"index": index, "error": f"{type(exc).__name__}: {exc}",
                             "draw": draw.summary()})
        if progress is not None:
            progress(index + 1, n_members)

    return Ensemble(
        depth_tvdss=np.asarray(depth_tvdss, dtype=float),
        members=members,
        failures=failures,
        truth_twt=None if truth_twt is None else np.asarray(truth_twt, dtype=float),
        # Half a sample interval: the data cannot locate an event more precisely
        # than this, so neither can the tie, whatever the ensemble spread says.
        resolution_floor_s=(
            resolution_floor_s if resolution_floor_s is not None else seismic.dt / 2.0
        ),
    )


def _sample(rng: np.random.Generator, index: int, period_s: float) -> Draw:
    """Draw one set of interpreter choices."""
    return Draw(
        despike_threshold=float(rng.uniform(3.0, 6.0)),
        knot_spacing_m=(None if rng.random() < 0.4
                        else float(rng.uniform(200.0, 800.0))),
        wavelet_length_s=float(rng.uniform(0.080, 0.180)),
        ridge=float(10.0 ** rng.uniform(-2.5, -1.5)),
        backus_length_m=(0.0 if rng.random() < 0.5 else float(rng.uniform(10.0, 40.0))),
        start_offset_s=float(rng.uniform(-1.0, 1.0) * period_s),
        index=index,
    )


def _run_member(
    draw: Draw,
    seismic: Trace,
    depth_tvdss: np.ndarray,
    sonic_us_per_m: np.ndarray,
    density_g_cm3: np.ndarray,
    checkshot_depth: np.ndarray | None,
    checkshot_twt: np.ndarray | None,
    replacement_velocity: float,
) -> Member:
    """The whole pipeline, once, under one set of choices."""
    conditioned = despike(
        sonic_us_per_m, depth_tvdss, threshold=draw.despike_threshold
    ).log

    velocity = slowness_to_velocity(conditioned)
    density = np.asarray(density_g_cm3, dtype=float)
    if draw.backus_length_m > 0:
        velocity, density = backus_average(
            depth_tvdss, velocity, density, draw.backus_length_m
        )

    time_depth = integrate_sonic(
        depth_tvdss, velocity_to_slowness(velocity),
        ShallowModel(replacement_velocity=replacement_velocity),
    )

    if checkshot_depth is not None and checkshot_twt is not None:
        knots = None
        if draw.knot_spacing_m:
            knots = np.arange(
                checkshot_depth[0], checkshot_depth[-1] + draw.knot_spacing_m,
                draw.knot_spacing_m,
            )
            knots = knots[knots <= checkshot_depth[-1]]
            if knots.size < 2:
                knots = None
        time_depth = calibrate_to_checkshots(
            time_depth, checkshot_depth, checkshot_twt, knot_depth=knots
        ).time_depth

    if draw.start_offset_s:
        time_depth = time_depth.shifted(draw.start_offset_s)

    result = tie(
        seismic, time_depth, velocity_to_slowness(velocity), density,
        wavelet_length_s=draw.wavelet_length_s, ridge=draw.ridge,
    )

    return Member(
        draw=draw,
        twt=result.time_depth.twt,
        correlation=float(result.metrics.correlation),
        # The member started from a model already offset by `start_offset_s`,
        # and the tie reports its shift relative to *that* model. The quantity
        # comparable across members is therefore the sum, not the difference --
        # getting the sign wrong makes every member look like it landed on a
        # different cycle and turns the multimodality test into a false alarm.
        total_shift_s=float(result.total_shift_s + draw.start_offset_s),
        phase_deg=float(result.phase_deg),
    )


def _dominant_period(seismic: Trace) -> float:
    """Period of the seismic's dominant frequency, for sizing the cycle probe."""
    amplitude = seismic.amplitude - np.mean(seismic.amplitude)
    spectrum = np.abs(np.fft.rfft(amplitude))
    frequency = np.fft.rfftfreq(amplitude.size, d=seismic.dt)
    # Ignore DC and the lowest bins, which carry residual trend rather than signal.
    usable = frequency > 2.0
    if not np.any(usable):
        return 0.0
    peak = float(frequency[usable][int(np.argmax(spectrum[usable]))])
    return 1.0 / peak if peak > 0 else 0.0
