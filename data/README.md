# Data

## The short version

Development and testing run against **synthetic data with a known answer**
(`swt.forward`). Real data is a format smoke test, not the validation target.

That is a deliberate choice, not a workaround. On any real dataset the true
time-depth curve is unknown, so a tie can only be judged by a correlation
coefficient — which is the number a subtly broken pipeline will happily
maximise. The forward model knows the answer, so the test suite can ask whether
the tie *landed in the right place*, not merely whether it correlates.

## Fetching

```bash
python data/fetch.py            # download what the network allows
python data/fetch.py --check    # report reachability, download nothing
```

## Reachable from this environment

Small files on `raw.githubusercontent.com`, used to exercise the readers against
real headers and real quirks:

| File | What it is |
|---|---|
| `samples/f3.sgy` | F3 subset, 414 traces, 4 ms, coordinate scalar −10 |
| `samples/P-129.LAS` | Real LAS with a full curve suite |

## Blocked from this environment

The egress proxy blocks most open-data hosts — `terranubis.com` (F3 Demo),
`data.equinor.com` (Volve), `zenodo.org`, `wiki.seg.org`, `dataunderground.org`.
`codeload.github.com` (GitHub tarballs) is also blocked, though
`raw.githubusercontent.com` is not.

To use these datasets, download them on a machine with open network access and
drop them in place:

```
data/
  f3/        F3 Demo     — .sgy volume + .las wells   terranubis.com/datainfo/F3-Demo-2020
  volve/     Volve       — seismic, logs, checkshots  data.equinor.com
  poseidon/  Poseidon    — 3D with deviated wells     wiki.seg.org/wiki/Open_data
```

Nothing in `swt` needs these to run. When they arrive, `swt.io.las`,
`swt.io.segy` and `swt.io.checkshot` load them directly — start with
`swt.io.segy.survey_info()`, which reports the geometry and its warnings before
anything is extracted.

## A note on real SEG-Y

Always run `survey_info()` on a new file before extracting anything. Trace
headers are wrong often enough — bad coordinate scalars, swapped X/Y,
coordinates in arc-seconds, missing geometry — that a reader trusting them will
quietly extract from the wrong place. A tie built on the wrong trace looks
mediocre rather than broken, which is the worst possible failure mode.
