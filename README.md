# SWT — Smart Seismic-to-Well Tie

A seismic-to-well tie engine built on a deterministic geophysics core, with an
AI layer on top.

> The physics is deterministic and must be exactly right. The AI makes it *fast*
> and *honest* — it does not make it *correct*.

An auto-tie that reaches correlation 0.95 by applying 40% stretch is a worse
answer than a manual tie at 0.7, and this software is built to say so.

See [`docs/PLAN.md`](docs/PLAN.md) for the full design and roadmap.

## Status

| Milestone | State |
|---|---|
| **M0** Scaffolding, forward model, test harness | done |
| **M1** Deterministic tie, validated against ground truth | done |
| **M2** Streamlit UI | done |
| **M3** Auto-tie: constrained DTW + velocity guardrail | done |
| **M4** Uncertainty quantification | done |
| **M5** Claude copilot over the engine | done |

## Quick start

```bash
pip install -e ".[dev,app]"
streamlit run app/streamlit_app.py   # the interactive tie
python examples/demo_tie.py          # the same tie, headless, graded against truth
python -m pytest -q                  # 235 tests
```

The app opens on a synthetic case builder: choose a static, a wavelet phase, a
signal-to-noise ratio and how many cycle skips to inject, then run the pipeline
and watch each stage. Because the case knows its own answer, the header carries
an **error vs truth** metric next to the correlation — the two disagree more
often than is comfortable, which is the point.

`demo_tie.py` builds a forward-modelled earth, hands the pipeline only what an
interpreter would actually have — a drifted sonic, a density log, checkshots and
a seismic trace — then reports both the tie's own QC and how far the answer is
from the truth. The second number is the one that matters.

## Using it

```python
from swt.io.las import load_las
from swt.io.segy import survey_info, extract_trace
from swt.timedepth.integrate import integrate_sonic, ShallowModel
from swt.timedepth.calibrate import calibrate_to_checkshots
from swt.tie.iterate import tie

print(survey_info("volume.sgy").summary())      # always look before extracting
seismic = extract_trace("volume.sgy", x=..., y=...)
logs = load_las("well.las")

raw = integrate_sonic(depth_tvdss, logs.sonic_us_per_m,
                      ShallowModel(replacement_velocity=1900.0))
calibrated = calibrate_to_checkshots(raw, cs_depth, cs_twt).time_depth

result = tie(seismic, calibrated, logs.sonic_us_per_m, logs.density_g_cm3)
print(result.summary())
print(result.significance.verdict())
```

With no data to hand, `swt.forward.make_case()` builds a complete tie problem
that knows its own answer.

## What the core does

| Module | Job |
|---|---|
| `swt.units` | Unit detection. Refuses to guess — a sonic read as µs/m when it is µs/ft is wrong by 3.28× and still looks plausible |
| `swt.io` | LAS, SEG-Y, checkshots, deviation surveys, path-following extraction |
| `swt.logs.condition` | Despiking, cycle-skip and bad-hole detection, gap filling — nothing silently edited |
| `swt.timedepth` | Sonic integration, checkshot drift calibration, and the strict-monotonicity invariant |
| `swt.petro` | Impedance, reflectivity, acoustic Backus upscaling |
| `swt.synth` | Depth→time resampling (anti-aliased) and the convolutional model |
| `swt.wavelet` | Ricker/Ormsby, statistical, and least-squares deterministic estimation |
| `swt.tie` | Bulk shift, constant-phase scan, constrained DTW, the velocity guardrail |
| `swt.qc` | Correlation, NRMS, PEP — and whether any of them is *significant* |
| `swt.forward` | The synthetic earth with a known answer |
| `swt.uq` | Ensemble over interpreter choices: corridor, per-horizon tolerance, multimodality |
| `swt.session` | The one mutable tie state — and the copilot's eventual tool surface |
| `swt.viz` | Matplotlib panels, shared by the app, notebooks and reports |
| `swt.copilot` | Claude's tool surface over the session, the permit gate, and the loop |

`TieSession` is the seam between the deterministic core and everything above it.
The UI drives it; the copilot will drive the *same* methods. Two rules hold at
that boundary: **arrays never cross it** (every method returns compact JSON —
statistics, intervals, verdicts, never a 16,000-sample curve), and **every
mutation is journalled**, which is what a tie report is made of.

## Deviated wells

A deviated well breaks a tie in two independent ways, and both are silent.

The log is recorded against **measured depth**; the seismic is indexed by **true
vertical depth**. In a well with 800 m of departure those differ by hundreds of
metres, and a sonic integrated against MD stretches the time-depth curve by
exactly that difference. Nothing downstream flags it — the drift curve absorbs
part, the bulk shift absorbs more, and the tie ends up plausible and wrong. The
test suite demonstrates this rather than asserting it: on an isolated case, MD
integration is 150+ ms out where TVDSS integration is exact.

And the well is **not under its wellhead**. `extract_along_path` assembles the
trace sample by sample — at each output time the well is at some TVDSS, hence
some (x, y), and the amplitude comes from the trace nearest *that* point. The
circularity (knowing where the well is at time *t* needs the time-depth model the
tie produces) is handled by iterating, not by pretending: extract, tie, re-extract.
One pass suffices, because lateral position changes slowly enough with time that
tens of milliseconds of error moves the well a few metres.

## The copilot

`swt.copilot` wires Claude (`claude-opus-5`, adaptive thinking) to a session
through **the same methods the UI calls** — there is no second implementation of
the pipeline to drift out of step, and no path by which the model touches a numpy
array. Fifteen tools: nine read-only, six that change the tie.

Its job is diagnosis, not narration. A dashboard already shows every number it can
see; what it adds is *"correlation 0.42, the drift steps 8 ms at 2100 m and DT
doubles over 2080–2140 m — that reads as a cycle skip, not geology"*.

Two things are enforced rather than requested:

- **Mutating tools are gated.** By default the copilot can look but not touch; a
  refused call returns an explanation it must relay rather than retry. The
  engine's guardrails still apply on top, so it cannot exceed the velocity limit
  or invert the time-depth even with permission.
- **Every refusal is logged.** A denied call changes nothing, so the session's
  journal never sees it — but what the copilot *wanted* to do belongs in the audit
  trail.

The tool surface, the gate and the loop are tested against a stub client that
plays scripted tool calls and executes them for real against a real session. **The
live API call is not tested** — it needs credentials this environment lacks. Set
`ANTHROPIC_API_KEY` to use it; everything else in SWT works without it.

## Six things this does that most well-tie code does not

**Significance testing.** A seismic trace is band-limited, so a 200 ms window of
10–50 Hz data holds roughly 16 independent numbers, not 100. Correlating two
random band-limited series of that length gives ~0.5 routinely. `swt.qc.significance`
computes the chance level for the actual window and bandwidth, so a tie at 0.6
over a short narrowband window is reported as *not significant* rather than
coloured green.

**Correct depth→time resampling.** Reading impedance at each 2 ms tick is
decimation without an anti-alias filter: it aliases every thin bed into the
seismic band and changes the reflectivity spectrum, invisibly. `swt.synth.resample`
averages impedance over each output bin instead — an exact boxcar low-pass, in
closed form. The test suite shows the naive route carrying >50× the energy it should.

**Phase-blind coarse alignment.** The first shift is found on the analytic-signal
envelope, not the trace. A trace-domain correlation against data carrying 180° of
residual phase locks onto the wrong half-cycle and converts a phase error into a
time error that no later phase scan can undo.

**The wavelet is not allowed to hide a static.** A 128 ms least-squares wavelet can
shift its own energy by several milliseconds to fit a misaligned synthetic. The
loop converges, the next shift search finds nothing, and the time-depth is quietly
wrong. Every deterministic wavelet is recentred and its offset pushed back into the
time-depth model, where it stays visible.

**A warp has to earn its place.** Warping is where an auto-tie stops being
trustworthy: a bulk shift has one free parameter, a warp has one per sample, and
with that freedom correlation stops being evidence. So a warp is converted back
into the velocity change it claims against the sonic and must pass three separate
tests — the claim is plausible (≤15% by default), the warp *reached* that claim
rather than being clipped to it, and it buys enough correlation to justify the
freedom. A rejected warp is discarded even when it correlates better.

**Uncertainty is tested for calibration, not just produced.** An ensemble re-runs
the whole tie over sampled interpreter choices — despike threshold, drift knots,
wavelet length, upscaling — and reports a corridor and a per-horizon tolerance in
milliseconds rather than a curve. Crucially the corridor's *coverage* is measured
against ground truth: a P10–P90 band must contain the truth about 80% of the time,
and it measures 0.815 across cases. A band that claims 80% and delivers 40% is not
conservative, it is a false statement that will be believed.

Each of these was found by the ground-truth tests, not by inspection — three were
live bugs that produced high correlations on a wrong time-depth.

## Testing

235 tests, in seven tiers:

- **Ground truth** (`tests/test_ground_truth.py`) — a forward-modelled earth is
  tied, and the recovered time-depth, static and wavelet phase are graded against
  what the model actually used. This is the tier that matters: a pipeline with a
  sign error or a phase/time confusion still reports a high correlation, because
  a high correlation is what it optimised for.
- **Invariants** (`tests/test_invariants.py`) — the guarantees the rest of the
  package relies on, and, in several cases, proof that the failure modes the
  docstrings warn about are actually prevented.
- **Session and panels** (`tests/test_session.py`) — the tool-surface contract
  (ordering, invalidation, JSON-only output) and that every display builds.
- **Guardrail and auto-tie** (`tests/test_guardrail.py`, `tests/test_autotie.py`) —
  the velocity arithmetic against hand-built warps, and the property that the
  warp never degrades the time-depth across both regimes.
- **Uncertainty** (`tests/test_uq.py`) — corridor *coverage* against ground truth.
  An uncertainty estimate is only worth reporting if it is calibrated.
- **Copilot** (`tests/test_copilot.py`) — the tool surface, the permit gate and
  the loop, driven by a stub client executing real tools against a real session.
- **Deviated wells** (`tests/test_deviated.py`) — the survey inverse, and
  path-following extraction verified trace-by-trace against a synthetic volume
  whose every trace is individually identifiable.

## Data

Development runs against synthetic data with a known answer. Real data is a
format smoke test, not the validation target — see [`data/README.md`](data/README.md),
which also lists which open datasets are reachable from a restricted network and
how to drop F3 or Volve in manually.
