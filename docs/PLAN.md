# SWT — Smart Seismic-to-Well Tie

A geophysically rigorous seismic-to-well tie engine with an AI layer on top.

**Decisions made** (2026-08-17):

| Question | Answer |
|---|---|
| Deliverable | Python core package + Streamlit UI |
| AI scope | Constrained-DTW auto-tie, Claude copilot, uncertainty quantification |
| Data | Public open datasets (see §7 — partially blocked in the dev environment) |

Explicitly **out of scope for now**: ML log prediction (synthesising missing DT/RHOB from
other curves). Simple deterministic fallbacks (Gardner, Faust) are in scope; a trained
model is not.

---

## 1. Guiding principle

> The physics is deterministic and must be exactly right. The AI makes it *fast* and
> *honest* — it does not make it *correct*.

Every "smart" feature sits behind a guardrail derived from geophysics, not from a loss
function. An auto-tie that reaches correlation 0.95 by applying 40% stretch is a worse
answer than a manual tie at 0.7, and the software must say so.

The whole engine is therefore built in two strictly separated layers:

```
   deterministic core  ──  every step reproducible, unit-tested against
   (swt/*)                 a forward-modelled earth with a known answer
        ▲
        │  narrow, typed tool surface (JSON in / JSON out, no arrays)
        │
   AI layer            ──  DTW solver, ensemble UQ, Claude copilot
   (swt/tie, swt/uq, swt/copilot)
```

The copilot can only reach the core through the same tool surface a human reaches
through the UI. There is no path where the model touches numpy arrays directly.

---

## 2. Package layout

```
swt/
  io/          las.py  segy.py  checkshot.py  deviation.py     load + unit normalisation
  logs/        condition.py  upscale.py                        despike, gap-fill, TVD, Backus
  petro/       elastic.py                                      AI, reflectivity series
  timedepth/   integrate.py  calibrate.py  model.py            sonic integration, drift, T-D model
  wavelet/     statistical.py  deterministic.py  parametric.py wavelet estimation
  synth/       convolve.py                                     synthetic seismogram
  extract/     trace.py                                        seismic at/along the well path
  tie/         align.py  dtw.py  iterate.py                    bulk shift, warp, outer loop
  qc/          metrics.py  significance.py                     xcorr, PEP, NRMS, White test
  uq/          ensemble.py                                     perturbation ensemble, corridors
  copilot/     tools.py  agent.py  prompts.py                  Claude tool surface + loop
  viz/         panels.py                                       plot primitives (UI-agnostic)
  session.py                                                   the one mutable state object
app/
  streamlit_app.py                                             thin — calls swt only
tests/
  synthetic/   forward model fixtures with known ground truth
data/
  fetch.py     README.md                                       reachable sample data
docs/
  PLAN.md  THEORY.md                                           this file + the maths
```

`session.py` holds a single `TieSession` — logs, T-D model, wavelet, seismic trace,
warp, QC. It is the only mutable thing. Tools take a session and return a new state
plus a **compact** JSON summary. Big arrays never leave the process.

---

## 3. The deterministic chain (the fundamentals)

Each step below is a module, a test, and a UI panel. The order matters; failures
cascade downward, so the QC of each step gates the next.

### 3.1 Load and condition

- **LAS** via `lasio`. Detect units (µs/ft vs µs/m, g/cc vs kg/m³, ft vs m) rather than
  assume them; refuse to guess silently.
- **Nulls**: `-999.25` and friends → NaN, tracked as a mask, never interpolated blindly.
- **Deviation survey** → MD → TVD → TVDSS (minimum-curvature). Everything downstream
  works in TVDSS below the seismic reference datum (SRD). Vertical wells are the
  degenerate case, not the assumption.
- **The shallow gap.** Logs start hundreds of metres below SRD. The interval between
  SRD and log top has to be filled by checkshot, or by a replacement velocity. *This is
  where most bad ties are actually born* — a wrong shallow model shifts the whole
  synthetic. The UI must show this gap explicitly and force a choice.
- **Despike**: rolling-median + MAD threshold on DT and RHOB, with **cycle-skip
  detection** as a separate test (an abrupt ~×2 jump in DT persisting over several
  samples is a skip, not geology).
- **Bad-hole flagging**: caliper vs bit size; intervals where RHOB is unreliable get
  flagged and optionally replaced by Gardner (ρ = 0.23·Vp^0.25, Vp in ft/s).
- **Missing RHOB / DT** fallbacks: Gardner (density from velocity), Faust (velocity from
  resistivity + depth). Deterministic, clearly labelled as synthesised in every plot.

### 3.2 Depth → time

TWT to the base of each sample, integrating slowness downward:

```
TWT(z) = TWT(z₀) + 2 ∫ (1/V(z)) dz          V from DT
```

- **Checkshot / VSP calibration.** Compute drift `Δ(z) = TWT_sonic(z) − TWT_checkshot(z)`
  at each checkshot depth, fit a smooth, low-order drift curve (piecewise-linear with
  operator-chosen knees, or a monotone spline), and apply it as a correction to the
  sonic. Show the drift curve — it is a diagnostic in its own right: a step in drift
  means a log problem at that depth, and drift slope is a direct check on
  dispersion/invasion.
- **Hard invariant: T(z) must be strictly monotonic.** Enforce it after every operation
  and raise if violated. A non-monotonic T-D silently corrupts everything downstream and
  is the single most common source of nonsense ties.

### 3.3 Reflectivity

- `AI = ρ · Vp` (P-impedance).
- Normal-incidence RC: `r_i = (AI_{i+1} − AI_i) / (AI_{i+1} + AI_i)`.
- **Resample depth → uniform time correctly.** The RC series lives on irregular time
  samples; converting to the seismic rate (2 or 4 ms) by naive interpolation aliases thin
  beds and quietly changes the amplitude spectrum. Use time-domain binning/averaging of
  impedance *before* differencing, or an explicit anti-alias filter.
- Optional **Backus upscaling** to seismic scale, with the averaging length tied to the
  dominant wavelength (λ/3 rule of thumb), exposed as a slider — it visibly changes the
  synthetic and users should see that.

### 3.4 Wavelet estimation

Three methods, used at different stages:

1. **Parametric** (Ricker, Ormsby) — first look and sanity check.
2. **Statistical** — from the seismic autocorrelation in a window around the well:
   amplitude spectrum from the smoothed autocorrelation, phase assumed constant and
   scanned. No well needed, so it works before the T-D is trustworthy.
3. **Deterministic / least-squares (Roy White)** — Wiener filter matching RC → seismic,
   with taper, white-noise regularisation, and an explicit design window. This is the
   good one, but it *presupposes a decent T-D*, so it belongs in iteration 2+.

**Constant-phase scan**: rotate θ over [−180°, 180°], take the θ maximising correlation.
Report residual phase after tie — a persistent non-zero residual phase means the wavelet
or the polarity convention (SEG normal vs reverse) is wrong, and that should be said in
words, not buried in a number.

### 3.5 Synthetic

`s(t) = w(t) ⊛ r(t)`, resampled to exactly the seismic trace's sample rate and length.

### 3.6 Seismic extraction

- `segyio` reader; geometry from trace headers with a manual override, because **headers
  lie** — coordinate scalars (`SourceGroupScalar`) and CDP X/Y are wrong often enough
  that a manual geometry entry path is mandatory, not a nicety.
- Extract the trace at the well surface location; for a deviated well, extract *along the
  well path* (one trace per TVDSS sample, following the deviated x/y).
- Optional small aperture average (3×3 traces) to lift S/N, with the caveat that it also
  smooths — shown as an option, off by default.

---

## 4. The auto-tie (the "smart" engine)

Four stages, run as an outer loop until convergence (typically 2–3 passes).

**Stage A — bulk shift.** Cross-correlate synthetic against seismic over a generous lag
window; take the peak. Cheap, robust, does most of the work.

**Stage B — constrained DTW.** Dynamic time warping between synthetic and seismic, but
constrained so the warp stays geological:

- *monotonicity* — no time inversions, ever;
- *slope bounds* — warp gradient clamped to roughly [0.8, 1.25], i.e. a bounded
  stretch/squeeze, which is a direct statement about how much the interval velocity is
  allowed to differ from the log;
- *curvature penalty* — a regularisation term on the second derivative of the warp so it
  cannot wiggle to chase noise;
- *anchors* — formation tops or user picks enter as hard constraints on the warp path.

Run it coarse-to-fine: first on the **envelope** (analytic signal magnitude, which is
phase-insensitive so it will not be trapped by a wrong polarity or phase), then on the
trace itself for fine alignment.

**Stage C — guardrail and back-conversion.** The warp maps synthetic time to seismic
time; convert it into an updated T-D curve and compute the *implied interval velocity
change versus the sonic*. **Reject or flag any interval where that exceeds a threshold
(default ~15%).** This is the difference between a smart tie and a well-correlated lie.
The updated T-D must also stay monotonic (§3.2).

**Stage D — re-estimate and iterate.** With the better T-D, re-run the deterministic
wavelet extraction, rebuild the synthetic, and go back to A. Stop on convergence of the
correlation coefficient, or after N passes.

---

## 5. QC — always visible, never optional

| Metric | What it catches |
|---|---|
| Cross-correlation coefficient (in window) | overall tie quality |
| **White's significance test** | whether that correlation *means anything* given the bandwidth and window length |
| PEP / predictability | proportion of seismic energy the synthetic explains |
| NRMS | amplitude and phase mismatch together |
| Residual phase after tie | wrong wavelet phase or reversed polarity |
| Dominant frequency, bandwidth | whether the wavelet is plausible for the data |
| **Implied velocity change vs depth** | how much the tie has distorted the well — the honesty metric |
| Drift residual at checkshots | log problems, bad calibration knots |

The White significance test is the one that most tools skip and the one that keeps the
software honest: a correlation of 0.8 over a short window in a narrow band can be
statistically meaningless, and the report must say so rather than print a green number.

---

## 6. Uncertainty quantification

A tie is not one answer. Build an **ensemble** by perturbing the choices an interpreter
makes:

- wavelet design window position and length, and the white-noise regularisation term;
- log-conditioning choices — despike threshold, Backus length, bad-hole treatment;
- checkshot drift knot placement;
- bulk-shift starting lag (to expose cycle-skip ambiguity — ties that jump a loop).

Run N ties (embarrassingly parallel), then report:

- a **T-D corridor** (P10/P50/P90) rather than a single curve;
- per-horizon time uncertainty in ms — which is what the interpreter actually needs
  when they carry the tie into a depth conversion;
- the distribution of correlation and of wavelet phase;
- **multimodality detection** — if the ensemble splits into two clusters a loop apart,
  the tie is ambiguous and the software must say so instead of averaging them.

Full Bayesian inversion (MCMC/variational over wavelet coefficients and warp
parameters) is a later upgrade; the ensemble gets ~80% of the value for ~20% of the work.

---

## 7. Data

**Constraint found in this environment:** the egress proxy blocks most open-data hosts —
`terranubis.com` (F3 Demo), `data.equinor.com` (Volve), `zenodo.org`, `wiki.seg.org`,
`dataunderground.org` all fail to connect. `pypi.org` and `raw.githubusercontent.com`
work; `codeload.github.com` (tarball download) is blocked.

So the data strategy is three-tier:

1. **Synthetic forward model — the test oracle, and the primary development target.**
   Generate a blocky earth model → known Vp, ρ → known RC → convolve with a *known*
   wavelet → add controlled noise and a *known* time-depth distortion. The tie must
   recover the T-D within a tolerance and the wavelet phase within a few degrees. Ground
   truth is the only way to know the engine is right, and it is the only data that never
   goes stale or offline.
2. **Reachable real samples** for format/robustness smoke tests, fetched from
   `raw.githubusercontent.com` (verified reachable): `equinor/segyio` test SEG-Y files
   and a real LAS from the `welly` test assets.
3. **User-supplied local files** — `data/README.md` documents dropping F3 / Volve /
   Poseidon into `data/` manually. `data/fetch.py` attempts the download and degrades to
   clear instructions rather than a stack trace when the host is blocked.

---

## 8. The Claude copilot

**Model**: `claude-opus-5` with adaptive thinking (`thinking: {"type": "adaptive"}`),
streaming, via the Anthropic Python SDK's tool runner
(`client.beta.messages.tool_runner`). Streaming matters here because a tie conversation
involves several tool round-trips and the user should watch it happen, not stare at a
spinner.

**Tool surface** — the engine's public verbs, one tool each:

```
load_well, condition_logs, build_time_depth, calibrate_checkshots,
extract_seismic_trace, extract_wavelet, make_synthetic,
auto_tie, compute_qc, run_uncertainty, get_state, describe_interval
```

Two design rules that decide whether this works at all:

1. **No arrays across the boundary.** Tools return statistics, flags, and handles —
   `{"correlation": 0.62, "phase_deg": -34, "worst_interval_m": [2080, 2140],
   "implied_dv_pct": 21}` — never a 40 000-sample curve. Arrays stay in the session
   store. This is what keeps the context budget and the latency sane over a long tie.
2. **Mutating tools are gated.** `auto_tie` and anything that rewrites the T-D returns a
   *proposal* the UI renders for approval. The model cannot silently apply a 30% squeeze.

**What the copilot is actually for** — not chat, but diagnosis:

> "Correlation is 0.42. The drift curve steps by 8 ms at 2100 m and DT roughly doubles
> over 2080–2140 m, which reads as a cycle skip rather than geology. Want me to despike
> that interval and rebuild the T-D?"

That is a real interpreter workflow, and it is the thing a numeric dashboard cannot do.
Secondary jobs: explaining what a QC number means in context, and writing the final tie
report (method, parameters, metrics, caveats) in prose.

**Prompt/context design**: system prompt and tool list are stable and cached (prompt
caching, cache breakpoint after the tool list); the volatile session summary is appended
after the breakpoint so the cache actually hits.

---

## 9. Milestones

| # | Milestone | Done when | State |
|---|---|---|---|
| **M0** | Scaffolding | package, `pyproject.toml`, pytest, `data/fetch.py`, synthetic generator | **done** |
| **M1** | Deterministic tie, no AI | load → condition → T-D → RC → wavelet → synthetic → manual shift → QC, **validated against synthetic ground truth** | **done** |
| **M2** | Streamlit UI | log panel, T-D + drift, wavelet, synthetic-vs-seismic, QC dashboard | **done** |
| **M3** | Auto-tie | constrained DTW + phase scan + outer loop + velocity guardrail | **done** |
| **M4** | UQ | ensemble, T-D corridor, per-horizon ±ms, multimodality flag | **done** |
| **M5** | Copilot | tool surface, tool runner, streaming chat panel, generated tie report | next |

M1 is the milestone that matters. Everything after it is leverage on top of a correct
engine; if M1 is wrong, M3–M5 produce confident nonsense.

### What M1 actually established

Ten random earth models recover the imposed static to within one seismic sample and
the time-depth to under 3 ms rms, across wavelet phases of 0, ±30, ±60, ±90 and 180
degrees, statics of 0 to ±20 ms, and signal-to-noise from noise-free down to 2.

More usefully, the ground-truth tests found three defects that a correlation-only
test could not have — every one of which produced a *high correlation on a wrong
answer*, which is the exact failure this project is organised against:

1. **The phase scan returned a mismatched phase and synthetic.** `phases[0]` is
   −180°, but the best-so-far was seeded with the *unrotated* wavelet, so whenever
   −180° won, the reported phase and the returned synthetic disagreed. The next
   cross-correlation then compared a trace against its own inverse and picked a side
   lobe half a period away. Result: correlation 0.995, time-depth 16 ms out.
2. **Trace-domain coarse alignment cannot survive residual phase.** It locks onto the
   wrong half-cycle and converts a phase error into a time error that no later phase
   scan can undo. Coarse alignment now runs on the analytic-signal envelope, which is
   phase-blind; the sharper trace estimator runs afterwards, once phase is known.
3. **The least-squares wavelet absorbed residual statics into its own energy offset.**
   A 128 ms wavelet can move its energy several milliseconds to fit a misaligned
   synthetic. The loop then converges, the next shift search finds nothing — the error
   is inside the wavelet — and the time-depth is quietly wrong. Deterministic wavelets
   are now recentred, with the offset pushed back into the T-D where it stays visible.

Two smaller ones came out of the invariant tests: the despike threshold's robust scale
collapses to zero on a smooth log (the local median tracks it exactly, so the MAD is
zero and nothing is ever flagged), and despiking without a width guard eats bed
boundaries and cycle skips.

The lesson for M3–M5 is the one the plan already asserted, now with evidence: **the
guardrails are the product**. Three of these five defects would have shipped happily
under a correlation-threshold acceptance test.

---

## 10. Testing

- **Ground-truth recovery** (the headline test): forward-model → tie → assert recovered
  T-D within tolerance and wavelet phase within a few degrees.
- **Property tests**: T-D strictly monotonic after every operation; DTW warp respects its
  slope bounds; energy/scaling preserved through resampling.
- **Regression on QC**: a fixed synthetic case has fixed expected metrics.
- **Format robustness**: odd LAS units, null conventions, missing curves, IBM-float
  SEG-Y, lying header scalars, deviated wells.
- **Adversarial cases**: a cycle-skipped log, a 180°-phase-reversed seismic volume, a
  well with no checkshot. Each should produce a clear diagnosis, not a plausible-looking
  wrong answer.

---

## 11. Stack

`numpy`, `scipy`, `lasio`, `segyio`, `pandas`, `matplotlib`/`plotly`, `streamlit`,
`anthropic`, `pydantic`, `pytest`. Optionally `bruges` for reference wavelet/Backus
implementations to check ours against, and `numba` if the DTW needs speed.

DTW is implemented in-house rather than taken from a library — the constraints (slope
bounds, curvature penalty, hard anchors, envelope-then-trace staging) are the whole
point, and generic DTW packages do not expose them.

---

## 12. Known risks

| Risk | Mitigation |
|---|---|
| Unit chaos (µs/ft vs µs/m, ft vs m, g/cc vs kg/m³) | detect and assert, never assume; unit tests per convention |
| SEG-Y headers with wrong coordinate scalars | manual geometry override path, mandatory |
| The shallow no-log gap above the log top | explicit UI step; replacement velocity or checkshot required |
| Polarity/phase convention (SEG normal vs reverse) | residual-phase QC surfaced in words |
| DTW over-fitting into a great-looking, non-physical tie | slope bounds + curvature penalty + implied-velocity guardrail |
| Deviated wells | TVDSS throughout; extraction along the well path from day one |
| Copilot context blow-up | no arrays across the tool boundary; compact JSON summaries |
| Open datasets unreachable from this environment | synthetic oracle is the primary target; real data is a smoke test |


### What M3 established

The warp solver recovers a known shift field to under 2 ms rms (the sample interval is
2 ms) for constant, ramped and sinusoidal warps, never exceeds its strain limit —
including at the trace ends, where a control-point scheme leaks and nobody looks — and
survives a 180° polarity flip via the envelope pass.

The interesting result is not the solver, though. It is that **three separate tests are
needed before a warp may be applied**, and each was forced by a measured failure:

1. **The velocity claim must be plausible.** The guardrail converts the warp into the
   interval-velocity change it asserts against the sonic. This was built and tested
   first, against hand-constructed warps with analytically known velocity change,
   before the solver existed.

2. **The warp must have *reached* that claim, not been clipped to it.** A warp pinned at
   its strain limit passes the guardrail *by construction* — it was clipped to a passing
   value. In one forward-modelled case the tie was 114 ms wrong, the warp sat at its
   limit over 69% of the log, the guardrail reported an admissible 14.3%, and the
   correlation improved. Every check said yes. Saturation is the only test that catches
   this, and the threshold (0.5) is measured: over 24 cases, every improving warp
   saturated at ≤0.42 and the sole degrading one at 0.59.

3. **The warp must buy enough correlation to justify a degree of freedom per sample.**
   The original threshold of 0.005 was far too permissive. On already-good ties the warp
   would gain 0.006–0.036 correlation while adding 1–8 ms of time-depth error — and
   every damaging case had a **cycle skip** in the log. The warp was stretching to align
   a spurious event that the skip had put into the synthetic. Helpful warps gained
   0.10–0.25; the gap is threefold and clean, so the threshold is 0.05. The lesson
   outlives the number: *a warp that gains only a little is usually chasing a defect in
   the log*, and the fix is to repair the log.

With all three in place the measured behaviour is: **0 of 34 cases degraded**, across both
a well-calibrated regime (where the warp correctly declines to act at all) and a
localised-sonic-anomaly regime (where it improves 11 of 14). That is a stronger
guarantee than "improves the correlation", and it is the one worth having.

Two of the three tests would be absent from a system designed around a correlation
threshold, and both of them exist to catch warps that *improve the correlation*.


### What M4 established

The ensemble re-runs the whole pipeline over sampled interpreter choices and reports a
corridor, a per-horizon tolerance in milliseconds, and a multimodality flag.

The result that matters is **coverage**, because it is the only thing that makes an
uncertainty estimate worth reporting. Measured against ground truth across twelve
forward-modelled cases, the P10–P90 corridor covers **0.815** (nominal 0.80), and the
across-case coverage at mid-depth is 11/12. The corridor is calibrated, not decorative.

Three findings worth carrying forward:

1. **A sign error made the ensemble permanently "ambiguous".** Members start from
   time-depth models deliberately offset by up to a wavelet period, to probe whether two
   alignments a cycle apart are both defensible. The tie reports its shift relative to
   that offset model, so the comparable quantity is the *sum*, not the difference. With
   the sign wrong, every member looked like it had landed on a different cycle: a
   reported 132 ms shift spread alongside a 2.6 ms corridor, two numbers that cannot
   both be true. The disagreement between them is what exposed it.

2. **A fraction alone cannot detect multimodality on a small ensemble.** At eight
   members one stray draw is 12.5% and trips any sensible fraction threshold. Both a
   fraction and an absolute minimum of two members are required.

3. **The ensemble measures precision, not accuracy, and this is not fixable within the
   method.** In one case of twelve every member agreed to within 3 ms and all were wrong
   by 1.2 ms — a shared bias larger than the corridor half-width, so coverage collapsed
   to 0.05 while the corridor looked its most confident. An ensemble over processing
   choices is structurally blind to error that every member shares. The reported corridor
   therefore carries a **resolution floor** of half a seismic sample, acknowledging error
   sources the ensemble never varies (the noise realisation, the wavelet's own estimation
   error, the sample interval), and the raw ensemble spread is reported *separately* so
   the two are never conflated. The floor blunts the failure; nothing within the method
   removes it, and the docs say so rather than implying otherwise.
