# Getting started

A complete walkthrough, assuming nothing installed. Steps 0–5 take about fifteen
minutes and need only an internet connection. Do those before downloading any
seismic data — if something is wrong with the environment, better to find out
first.

---

## Step 0 — Check you have Python and git

Open a terminal:

| | |
|---|---|
| **Mac** | `Cmd + Space`, type `Terminal`, press Enter |
| **Windows** | Start button, type `PowerShell`, press Enter |
| **Linux** | `Ctrl + Alt + T` |

Type these one at a time:

```bash
python3 --version
git --version
```

You should see something like `Python 3.11.5` and `git version 2.39.3`.

- Python must be **3.10 or newer**. Missing or older → [python.org/downloads](https://www.python.org/downloads/)
- git missing? Mac offers to install it for you; otherwise [git-scm.com](https://git-scm.com/downloads)

> On Windows, if `python3` is not found, try `python`. Use whichever works for
> every later step.

---

## Step 1 — Download the code

```bash
cd ~
git clone https://github.com/kamranabbas20/SWT.git
cd SWT
git checkout claude/ai-seismic-well-tie-wmezk7
```

**The `git checkout` line is essential** — the code lives on that branch, not on
`main`. Without it you get an almost-empty folder.

Check it worked:

```bash
ls
```

You should see `README.md  app  data  docs  examples  pyproject.toml  swt  tests`.

---

## Step 2 — Install the dependencies

```bash
pip install -e ".[dev,app]"
```

**The quotes matter.** Without them, zsh on macOS treats the square brackets as
wildcards and errors. If `pip` is not found, try `pip3` or `python3 -m pip`.

This downloads numpy, scipy, lasio, segyio, streamlit and matplotlib. A minute or
two.

---

## Step 3 — Prove it works

```bash
python3 examples/demo_tie.py
```

This builds a synthetic earth, hands the pipeline only what an interpreter would
actually have, and grades the result against the answer it secretly knows. It
should end with:

```
Graded against the truth
------------------------
  time-depth error 0.50 ms rms
  shift error      0.00 ms
  phase error      5.0 deg

  PASS
```

Then the test suite — about three minutes:

```bash
python3 -m pytest -q
```

Expect `235 passed`. If either of these fails, stop here; the environment is wrong
and nothing later will make sense.

---

## Step 4 — Open the interface

```bash
streamlit run app/streamlit_app.py
```

This **does not return your prompt** — it prints a link like
`http://localhost:8501` and keeps running. That is normal. Open the link in a
browser.

From here everything happens **in the browser**, not the terminal. To stop the app
later, click the terminal window and press `Ctrl + C`.

---

## Step 5 — Run a synthetic tie

In the browser:

1. Sidebar → **Build case**
2. Sidebar → scroll down → **Run pipeline**

Across the top you should see correlation ≈ 0.925 and **error vs truth ≈ 0.50 ms**.

Then look through the tabs, in this order:

| Tab | What to look for |
|---|---|
| **Tie** | Synthetic (red) beside seismic (blue). Events should line up. |
| **Time-depth** | Read the drift panel. A smooth ramp is normal; a *step* means a log problem at that depth. |
| **QC** | Read the significance verdict, not just the correlation. |
| **Truth** | How far the answer actually is from the truth. |

**Then do this deliberately.** In the sidebar under *5. Auto-tie (warp)*, tick
**stretch and squeeze** and run again. A **Warp** tab appears — and the warp is
**rejected**, with a reason. Seeing it refuse is more informative than seeing it
succeed: that refusal is the entire thesis of the project.

To make it *accept* a warp, give it something real to fix: set **checkshot
spacing** to 600 m and **localised sonic anomaly** to 0.12, rebuild, and run again.

---

## Step 6 — The copilot (optional, needs an API key)

Stop the app (`Ctrl + C`), then set your key and restart:

**Mac / Linux**
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
streamlit run app/streamlit_app.py
```

**Windows PowerShell**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
streamlit run app/streamlit_app.py
```

Keys come from [console.anthropic.com](https://console.anthropic.com).

The variable applies only to **that terminal window**, and Streamlit will not pick
it up if it was already running — set it first, then start the app.

In the browser: Build case → Run pipeline → **Copilot** tab. Ask something like
*"How good is this tie, and what would you check?"*

Two things worth verifying:

1. **That it works at all.** This path has never made a live API call.
2. **That the gate holds.** With the toggle *off*, ask it to *"re-run the tie with
   a 40 ms wavelet"*. It must refuse and explain, not comply.

---

## Step 7 — Real data

```bash
mkdir -p data/f3
```

Download from [terranubis.com/datainfo/F3-Demo-2020](https://terranubis.com/datainfo/F3-Demo-2020)
into that folder. You need three things:

| File | What it must contain |
|---|---|
| `.las` | A well log with **both** a sonic curve (`DT`/`DTCO`) and density (`RHOB`) |
| `.sgy` | Seismic covering the well — **take the smallest crop offered** |
| `.txt`/`.csv` | A checkshot: two columns, depth and time |

A deviation survey (MD, inclination, azimuth) is also supported but F3's wells are
close enough to vertical to skip.

`.gitignore` already excludes `data/f3/`, so nothing large lands in the repository.

---

## Step 8 — Your first real tie

All in the browser. Sidebar → **Data source** → *Load files*.

**8a. Inspect the SEG-Y first.** Paste the path, click **Inspect SEG-Y**, read the
warnings. If it reports missing geometry or suspicious coordinates, fix that before
going on — a tie built on the wrong trace looks *mediocre* rather than broken,
which is the worst way to fail.

**8b. Load the LAS, and expect a unit error.** SWT refuses to guess units, and F3's
sonic is probably in µs/ft. An error here is the software working as designed. Add
the mnemonic to `CURVE_ALIASES` in `swt/io/las.py` or pass an explicit override.

**8c. Set the shallow model honestly.** The log starts hundreds of metres below the
seismic datum. With a checkshot, choose *Fix TWT at log top*. Without one, pick a
replacement velocity and know you are guessing — this is where most bad ties begin.

**8d. Run, then read the drift curve before the correlation.** A smooth ramp is
ordinary dispersion. A step means a log problem at that depth; check the **Logs**
tab for a cycle skip flagged there.

**8e. Only then look at the correlation — with its significance.** On real data,
0.5–0.8 is normal and honest.

---

## Troubleshooting

**`command not found: python3`** — Try `python`. If neither works, Python is not
installed or not on your PATH.

**`error: externally-managed-environment`** (Linux, newer Macs) — Your system
Python is protected. Make a virtual environment first:
```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,app]"
```
You will need the `activate` line in each new terminal.

**`zsh: no matches found: [dev,app]`** — You dropped the quotes. Use
`pip install -e ".[dev,app]"`.

**`ModuleNotFoundError: No module named 'swt'`** — You are not in the `SWT`
directory, or step 2 did not finish. `cd` to the folder containing
`pyproject.toml` and re-run the install.

**The app shows an old version after you edit a file** — Streamlit reloads most
edits automatically, but changes to imported modules sometimes need a restart:
`Ctrl + C`, then run it again.

**Tests fail on a fresh clone** — Send the output. That is a real bug and worth
knowing about.
