"""The tie session: one mutable object holding the state of a well tie.

This is the seam between the deterministic core and everything above it.  The
UI drives a session; later the Claude copilot will drive the *same* session
through the same methods, so there is exactly one path by which a tie can be
changed, and it is a path that records what it did.

Two rules govern every method here, and both exist for the copilot's sake even
though the UI is the first client:

1. **Arrays never cross this boundary.**  Every method returns a compact,
   JSON-safe dict -- statistics, flags, intervals, verdicts.  The 16,000-sample
   curves stay in the session.  A tool surface that returns arrays burns the
   model's context on numbers it cannot read and cannot act on.
2. **Every mutation is journalled.**  ``session.journal`` is an ordered record
   of what was done and what it produced, which is what a tie report is made of
   and what makes a tie reproducible by someone who was not there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .io.checkshot import Checkshots
from .io.deviation import Deviation
from .io.las import WellLogs
from .logs.condition import ConditioningReport, despike, detect_cycle_skips, fill_gaps
from .petro.elastic import acoustic_impedance, backus_average
from .qc.metrics import TieMetrics
from .timedepth.calibrate import DriftResult, calibrate_to_checkshots, drift_curve
from .timedepth.integrate import ShallowModel, integrate_sonic
from .timedepth.model import TimeDepth
from .tie.auto import AutoTieResult, auto_tie
from .tie.iterate import TieResult, tie
from .uq.ensemble import Ensemble, run_ensemble
from .trace import Trace
from .units import slowness_to_velocity


class SessionError(RuntimeError):
    """Raised when a step is attempted before the step it depends on."""


@dataclass
class TieSession:
    """Everything known about one well tie, and the operations that change it.

    The pipeline has a strict order -- conditioning, then time-depth, then
    calibration, then the tie -- and each method checks that its inputs exist
    rather than failing obscurely three steps later.
    """

    name: str = "session"

    # inputs
    logs: WellLogs | None = None
    deviation: Deviation | None = None
    checkshots: Checkshots | None = None
    seismic: Trace | None = None

    # working state
    depth_tvdss: np.ndarray | None = None
    sonic_us_per_m: np.ndarray | None = None
    density_g_cm3: np.ndarray | None = None
    conditioning: ConditioningReport | None = None
    cycle_skips: list = field(default_factory=list)
    time_depth: TimeDepth | None = None
    drift: DriftResult | None = None
    result: TieResult | None = None
    auto: AutoTieResult | None = None
    ensemble: Ensemble | None = None

    # grading, present only for synthetic cases that know their own answer
    truth: dict = field(default_factory=dict)

    journal: list[dict] = field(default_factory=list)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_synthetic(cls, seed: int = 3, **kwargs: Any) -> "TieSession":
        """Build a session from a forward-modelled case with a known answer."""
        from .forward import make_case

        case = make_case(seed=seed, **kwargs)
        session = cls(name=f"synthetic (seed {seed})")
        session.depth_tvdss = case.log_depth_tvdss
        session.sonic_us_per_m = case.log_dt_us_per_m
        session.density_g_cm3 = case.log_rho_g_cm3
        session.checkshots = Checkshots(
            case.checkshot_depth_tvdss, case.checkshot_twt, "synthetic checkshots"
        )
        session.seismic = case.seismic
        session.truth = {
            "twt_at_log": case.true_twt_at_log() + case.applied_static_s,
            "static_s": case.applied_static_s,
            "wavelet_phase_deg": case.params["wavelet_phase_deg"],
            "wavelet_frequency_hz": case.params["wavelet_frequency"],
            "case": case,
        }
        session._record("load synthetic", {"seed": seed, **{
            k: v for k, v in case.params.items() if k != "case"
        }})
        return session

    def set_deviation(self, deviation: Deviation, md_m: np.ndarray | None = None) -> dict:
        """Put the log on a TVDSS axis using a deviation survey.

        The log is recorded against measured depth; everything downstream works
        in TVDSS. In a deviated well those differ by hundreds of metres, and
        integrating a sonic against MD stretches the time-depth curve by exactly
        that difference -- an error the drift curve and the bulk shift will
        partly absorb, leaving a tie that is plausible and wrong.

        Parameters
        ----------
        md_m:
            The log's measured-depth axis. Defaults to the LAS file's own depth
            curve when logs were loaded from one.
        """
        if md_m is None:
            if self.logs is None or self.logs.depth_md_m is None:
                raise SessionError(
                    "no measured-depth axis available; pass md_m explicitly"
                )
            md_m = self.logs.depth_md_m
        md = np.asarray(md_m, dtype=float)

        tvdss = deviation.tvdss_at_md(md)
        if tvdss.size < 2 or np.any(np.diff(tvdss) <= 0):
            raise SessionError(
                f"{deviation.name}: TVDSS does not increase monotonically over the "
                "logged interval -- the well levels off or rises within the log, "
                "and a tie needs a well that goes down through it."
            )

        self.deviation = deviation
        self.depth_tvdss = tvdss
        self.time_depth = self.drift = self.result = self.auto = self.ensemble = None

        return self._record("set deviation", {
            **deviation.summary(),
            "md_range_m": [round(float(md[0]), 1), round(float(md[-1]), 1)],
            "tvdss_range_m": [round(float(tvdss[0]), 1), round(float(tvdss[-1]), 1)],
            "md_minus_tvdss_at_base_m": round(float(md[-1] - tvdss[-1]), 1),
        })

    def reextract_along_path(self, segy_path: str, aperture: int = 1) -> dict:
        """Re-extract the seismic following the borehole, using the current T-D.

        A deviated well is not under its wellhead, so the trace it should be tied
        to changes with depth. Knowing *where* the well is at a given time needs a
        time-depth model, which is what the tie produces -- so this is run after a
        first pass and the tie repeated. One iteration is normally enough: lateral
        position changes slowly with time, so a time-depth error of tens of
        milliseconds moves the well by metres, usually inside a single bin.
        """
        from .io.segy import extract_along_path

        if self.deviation is None:
            raise SessionError("no deviation survey loaded; nothing to follow")
        self._require("time_depth", "build a time-depth model before following the path")

        self.seismic = extract_along_path(
            segy_path, self.deviation, self.time_depth, aperture=aperture
        )
        self.result = self.auto = self.ensemble = None
        return self._record("re-extract along path", dict(self.seismic.meta))

    # -- pipeline steps -----------------------------------------------------

    def condition(
        self,
        despike_threshold: float = 4.0,
        despike_window: int = 15,
        max_gap_m: float = 5.0,
        detect_skips: bool = True,
    ) -> dict:
        """Despike the sonic, fill short gaps, and report cycle skips."""
        self._require("sonic_us_per_m", "load logs before conditioning them")

        raw = self._raw_sonic()
        filled = fill_gaps(raw, self.depth_tvdss, max_gap_m=max_gap_m)
        report = despike(
            filled.log, self.depth_tvdss, window=despike_window, threshold=despike_threshold
        )
        report = ConditioningReport(
            log=report.log,
            edits=filled.edits + report.edits,
            remaining_nan=report.remaining_nan,
        )

        self.conditioning = report
        self.sonic_us_per_m = report.log
        self.cycle_skips = (
            detect_cycle_skips(report.log, self.depth_tvdss) if detect_skips else []
        )

        summary = {
            **report.summary(),
            "cycle_skips": [skip.summary() for skip in self.cycle_skips],
        }
        # Conditioning invalidates everything derived from the log.
        self.time_depth = self.drift = self.result = self.auto = self.ensemble = None
        return self._record("condition", summary)

    def build_time_depth(
        self,
        replacement_velocity: float | None = None,
        twt_at_log_top: float | None = None,
    ) -> dict:
        """Integrate the conditioned sonic into a time-depth curve."""
        self._require("sonic_us_per_m", "load logs before building a time-depth model")

        shallow = ShallowModel(
            replacement_velocity=replacement_velocity, twt_at_log_top=twt_at_log_top
        )
        self.time_depth = integrate_sonic(self.depth_tvdss, self.sonic_us_per_m, shallow)
        self.drift = self.result = self.auto = self.ensemble = None

        return self._record("build time-depth", {
            "shallow_model": shallow.describe(float(self.depth_tvdss[0])),
            "twt_at_log_top_ms": round(float(self.time_depth.twt[0]) * 1e3, 1),
            "twt_at_log_base_ms": round(float(self.time_depth.twt[-1]) * 1e3, 1),
            **self._truth_check(self.time_depth),
        })

    def calibrate(self, knot_spacing_m: float | None = None) -> dict:
        """Correct the time-depth model onto the checkshots.

        ``knot_spacing_m`` coarsens the drift curve.  Honouring every checkshot
        exactly (the default) injects any checkshot noise straight into the
        velocity field, so a wider spacing is often the better answer -- and it
        is the fix when calibration reports an inverted time-depth.
        """
        self._require("time_depth", "build a time-depth model before calibrating")
        if self.checkshots is None:
            raise SessionError("no checkshots loaded; nothing to calibrate to")

        knots = None
        if knot_spacing_m:
            depth = self.checkshots.depth_tvdss_m
            knots = np.arange(depth[0], depth[-1] + knot_spacing_m, knot_spacing_m)
            knots = knots[knots <= depth[-1]]
            if knots.size < 2:
                raise SessionError(
                    f"knot spacing {knot_spacing_m} m is wider than the checkshot range"
                )

        self.drift = calibrate_to_checkshots(
            self.time_depth,
            self.checkshots.depth_tvdss_m,
            self.checkshots.twt_s,
            knot_depth=knots,
        )
        self.time_depth = self.drift.time_depth
        self.result = self.auto = self.ensemble = None

        return self._record("calibrate", {
            **self.drift.summary(),
            **self._truth_check(self.time_depth),
        })

    def run_tie(self, **kwargs: Any) -> dict:
        """Run the deterministic tie loop: bulk shift, phase, wavelet. No warping."""
        self._require("time_depth", "build a time-depth model before tying")
        if self.seismic is None:
            raise SessionError("no seismic loaded; nothing to tie to")

        self.result = tie(
            self.seismic,
            self.time_depth,
            self.sonic_us_per_m,
            self.density_g_cm3,
            **kwargs,
        )
        self.auto = None
        self.time_depth = self.result.time_depth

        return self._record("tie", {
            **self.result.summary(),
            **self._truth_check(self.time_depth),
        })

    def run_auto_tie(self, **kwargs: Any) -> dict:
        """Run the tie loop, then a warp that must pass the velocity guardrail.

        The warp is kept only if it is geologically admissible, is not merely
        the constraint's own limit clipped to look admissible, and buys enough
        correlation to justify a per-sample degree of freedom.  When it is
        rejected the session keeps the unwarped tie and the journal records why.
        """
        self._require("time_depth", "build a time-depth model before tying")
        if self.seismic is None:
            raise SessionError("no seismic loaded; nothing to tie to")

        self.auto = auto_tie(
            self.seismic,
            self.time_depth,
            self.sonic_us_per_m,
            self.density_g_cm3,
            **kwargs,
        )
        self.result = self.auto.result
        self.time_depth = self.result.time_depth

        return self._record("auto-tie", {
            **self.auto.summary(),
            "verdict_warp": self.auto.verdict(),
            **self._truth_check(self.time_depth),
        })

    def run_uncertainty(
        self,
        n_members: int = 24,
        seed: int = 0,
        probe_cycle_ambiguity: bool = True,
        progress=None,
        **kwargs: Any,
    ) -> dict:
        """Re-run the tie over sampled interpreter choices and report a corridor.

        The deliverable is a band and a per-horizon tolerance in milliseconds,
        not a curve. What the ensemble measures is how much the answer depends
        on decisions no one can make uniquely -- which is a narrower question
        than "how uncertain is the earth", and the one that changes what a tie
        can be used for.
        """
        self._require("sonic_us_per_m", "load logs before running an ensemble")
        if self.seismic is None:
            raise SessionError("no seismic loaded; nothing to tie to")

        self.ensemble = run_ensemble(
            self.seismic,
            self.depth_tvdss,
            self._raw_sonic(),
            self.density_g_cm3,
            checkshot_depth=(
                self.checkshots.depth_tvdss_m if self.checkshots is not None else None
            ),
            checkshot_twt=(
                self.checkshots.twt_s if self.checkshots is not None else None
            ),
            n_members=n_members,
            seed=seed,
            probe_cycle_ambiguity=probe_cycle_ambiguity,
            truth_twt=self.truth.get("twt_at_log") if self.truth else None,
            progress=progress,
            **kwargs,
        )
        return self._record("uncertainty", self.ensemble.summary())

    # -- derived quantities for display -------------------------------------

    def velocity(self) -> np.ndarray:
        self._require("sonic_us_per_m", "no sonic log")
        return slowness_to_velocity(self.sonic_us_per_m)

    def impedance(self, backus_length_m: float = 0.0) -> np.ndarray:
        """Acoustic impedance, optionally upscaled to seismic scale."""
        self._require("density_g_cm3", "no density log")
        velocity, density = self.velocity(), self.density_g_cm3
        if backus_length_m > 0:
            velocity, density = backus_average(
                self.depth_tvdss, velocity, density, backus_length_m
            )
        return acoustic_impedance(velocity, density)

    def observed_drift(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Drift at the checkshots against the current model, for plotting."""
        if self.time_depth is None or self.checkshots is None:
            return None
        return drift_curve(
            self.time_depth, self.checkshots.depth_tvdss_m, self.checkshots.twt_s
        )

    # -- reporting ----------------------------------------------------------

    def state(self) -> dict:
        """Compact description of where the tie has got to."""
        return {
            "name": self.name,
            "has_logs": self.sonic_us_per_m is not None,
            "has_checkshots": self.checkshots is not None,
            "has_seismic": self.seismic is not None,
            "deviated": (
                None if self.deviation is None else not self.deviation.is_vertical
            ),
            "depth_range_m": (
                [round(float(self.depth_tvdss[0]), 1), round(float(self.depth_tvdss[-1]), 1)]
                if self.depth_tvdss is not None else None
            ),
            "time_depth": self.time_depth.provenance if self.time_depth else None,
            "conditioned": self.conditioning is not None,
            "n_cycle_skips": len(self.cycle_skips),
            "tied": self.result is not None,
            "warp_accepted": self.auto.warp_accepted if self.auto else None,
            "ensemble_members": self.ensemble.n if self.ensemble else None,
            "correlation": (
                round(self.result.metrics.correlation, 4) if self.result else None
            ),
            "steps_taken": [entry["step"] for entry in self.journal],
        }

    def metrics(self) -> TieMetrics | None:
        return self.result.metrics if self.result else None

    def grade(self) -> dict:
        """Error against ground truth, or ``{}`` when there is none.

        Empty on real data, by construction -- there is nothing to grade
        against. That asymmetry is the argument for developing against a
        forward model rather than a field dataset.
        """
        if not self.truth or self.time_depth is None:
            return {}
        return self._truth_check(self.time_depth)

    def _record(self, step: str, payload: dict) -> dict:
        entry = {"step": step, **payload}
        self.journal.append(entry)
        return entry

    def _truth_check(self, time_depth: TimeDepth) -> dict:
        """Grade against ground truth, when the case has one.

        Present only for synthetic sessions.  On real data there is no truth and
        this returns nothing -- which is exactly the point of developing against
        a forward model.
        """
        if not self.truth:
            return {}
        error = time_depth.twt - self.truth["twt_at_log"]
        return {
            "truth_rms_error_ms": round(float(np.sqrt(np.mean(error**2))) * 1e3, 2),
            "truth_max_error_ms": round(float(np.max(np.abs(error))) * 1e3, 2),
        }

    def _raw_sonic(self) -> np.ndarray:
        """The sonic as first loaded, so conditioning is never applied twice."""
        if self.conditioning is None:
            return self.sonic_us_per_m
        if self.logs is not None and self.logs.sonic_us_per_m is not None:
            return self.logs.sonic_us_per_m
        if self.truth:
            return self.truth["case"].log_dt_us_per_m
        return self.sonic_us_per_m

    def _require(self, attribute: str, message: str) -> None:
        if getattr(self, attribute, None) is None:
            raise SessionError(message)
