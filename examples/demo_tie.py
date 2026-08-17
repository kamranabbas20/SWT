#!/usr/bin/env python3
"""End-to-end demonstration of the M1 tie on synthetic data with a known answer.

Run it with no arguments::

    python examples/demo_tie.py

It builds a forward-modelled earth, hands the pipeline only what an interpreter
would actually have -- a drifted sonic, a density log, a checkshot survey and a
seismic trace -- then reports both the tie's own QC and, because this is
synthetic, how far the answer is from the truth.

The second number is the one that matters.  A pipeline can be made to report a
high correlation while putting the well in the wrong place; only ground truth
tells the two apart.
"""

from __future__ import annotations

import argparse

import numpy as np

from swt.forward import make_case
from swt.logs.condition import despike, detect_cycle_skips
from swt.timedepth.calibrate import calibrate_to_checkshots
from swt.timedepth.integrate import ShallowModel, integrate_sonic
from swt.tie.iterate import tie


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--static-ms", type=float, default=12.0)
    parser.add_argument("--phase-deg", type=float, default=30.0)
    parser.add_argument("--snr", type=float, default=8.0)
    parser.add_argument("--cycle-skips", type=int, default=2)
    parser.add_argument("--replacement-velocity", type=float, default=1900.0)
    args = parser.parse_args()

    static = args.static_ms / 1e3
    case = make_case(
        seed=args.seed,
        static_s=static,
        wavelet_phase_deg=args.phase_deg,
        signal_to_noise=args.snr,
        n_cycle_skips=args.cycle_skips,
    )

    rule("The problem")
    print(f"  log        {case.log_depth_tvdss[0]:.0f}-{case.log_depth_tvdss[-1]:.0f} m TVDSS, "
          f"{case.log_depth_tvdss.size:,} samples")
    print(f"  unlogged   0-{case.log_depth_tvdss[0]:.0f} m above the log top")
    print(f"  checkshots {case.checkshot_depth_tvdss.size}")
    print(f"  seismic    {case.seismic}")
    print(f"  truth      static {static * 1e3:+.1f} ms, wavelet phase {args.phase_deg:+.0f} deg, "
          f"S/N {args.snr}")

    rule("1. Condition the sonic")
    skips = detect_cycle_skips(case.log_dt_us_per_m, case.log_depth_tvdss)
    print(f"  cycle skips detected: {len(skips)} (injected: {args.cycle_skips})")
    for skip in skips:
        print(f"    {skip.start_depth:.1f}-{skip.end_depth:.1f} m -- {skip.detail}")
    report = despike(case.log_dt_us_per_m, case.log_depth_tvdss)
    print(f"  despike: {len(report.edits)} interval(s) edited")

    rule("2. Integrate and calibrate")
    raw = integrate_sonic(
        case.log_depth_tvdss,
        report.log,
        ShallowModel(replacement_velocity=args.replacement_velocity),
    )
    drift = calibrate_to_checkshots(raw, case.checkshot_depth_tvdss, case.checkshot_twt)
    truth = case.true_twt_at_log()
    print(f"  drift: {drift.summary()}")
    print(f"  T-D error before calibration: {rms(raw.twt - truth):7.2f} ms rms")
    print(f"  T-D error after  calibration: {rms(drift.time_depth.twt - truth):7.2f} ms rms")

    rule("3. Tie")
    result = tie(case.seismic, drift.time_depth, report.log, case.log_rho_g_cm3)
    for step in result.history:
        name = step.pop("step")
        print(f"  {name:<28} {compact(step)}")

    rule("Result")
    summary = result.summary()
    print(f"  correlation      {summary['correlation']:.3f}")
    print(f"  NRMS             {summary['nrms_percent']:.1f}%")
    print(f"  PEP              {summary['pep']:.3f}")
    print(f"  bulk shift       {summary['total_shift_ms']:+.1f} ms   (truth {static * 1e3:+.1f})")
    print(f"  wavelet phase    {summary['phase_deg']:+.0f} deg  (truth {args.phase_deg:+.0f})")
    print(f"  wavelet          {result.wavelet.summary()['dominant_frequency_hz']:.1f} Hz dominant, "
          f"band {result.wavelet.summary()['bandwidth_hz']}")
    print(f"\n  significance     {result.significance.verdict()}")

    rule("Graded against the truth")
    final_error = rms(result.time_depth.twt - (truth + static))
    print(f"  time-depth error {final_error:.2f} ms rms")
    print(f"  shift error      {abs(result.total_shift_s - static) * 1e3:.2f} ms")
    phase_error = (result.phase_deg - args.phase_deg + 180.0) % 360.0 - 180.0
    print(f"  phase error      {abs(phase_error):.1f} deg")
    verdict = "PASS" if final_error < 3.0 else "FAIL"
    print(f"\n  {verdict}")
    return 0 if verdict == "PASS" else 1


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)) * 1e3)


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def compact(step: dict) -> str:
    keep = ("shift_ms", "correlation", "best_phase_deg", "best_correlation",
            "dominant_frequency_hz", "nrms_percent")
    return "  ".join(f"{k}={step[k]}" for k in keep if k in step)


if __name__ == "__main__":
    raise SystemExit(main())
