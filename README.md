# Spike Detective

Detects anomalous spikes in a daily metric (e.g. signups, registrations, event counts) and suggests which marketing touchpoint likely caused each one.

## The problem

Marketing attribution tools (multi-touch attribution platforms, CDPs with built-in modeling, etc.) are built for teams with the budget and data volume to justify them. A small team running a newsletter, a few social posts, and the occasional press mention doesn't have that infrastructure — but they still want to know "did that LinkedIn post actually do anything?"

This project is a lightweight stand-in: point it at a CSV of daily counts and a CSV of dated marketing touchpoints, and it flags the days that look statistically unusual and proposes the touchpoint most likely to have caused each one.

## Approach

1. **Changepoint detection** (`ruptures`, Pelt search with an RBF cost model) finds structural shifts in the series — points where the underlying distribution changes, not just single noisy days.
2. **Trailing rolling z-score** (`spike_detector.py`) flags individual days whose value is a statistical outlier relative to the preceding 7-day baseline. This is the primary spike trigger; the changepoint signal is attached as corroborating evidence (`is_changepoint`), since a single day's z-score can be noisy on its own.
3. **Proximity-based attribution** (`attribution.py`) then looks backward from each flagged spike within a configurable window (default 2 days) and ranks candidate touchpoints by recency — a touchpoint the day before a spike outranks one from a week earlier. Spikes with no touchpoint in the window are explicitly marked unexplained rather than silently dropped.

## Honest limitation

This is **correlation, not proof**. A touchpoint landing shortly before a spike is a plausible suspect, not a confirmed cause — the pipeline has no way to rule out coincidence, seasonality, or an unrelated event that happened to land in the same window. Treat every attribution as a lead to investigate, not a verdict. The "unexplained" spikes are just as important to look at as the explained ones: they're the ones where something happened that your tracked touchpoints don't account for.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

## Run

**Generate example data** (a synthetic timeseries with known ground-truth spikes, used to sanity-check the detector):

```bash
python generate_data.py
```

**Run the pipeline end-to-end** from the command line:

```bash
python pipeline.py                    # uses existing CSVs, generating them first if missing
python pipeline.py --regenerate       # force-regenerate the fixture CSVs before running
python pipeline.py --timeseries other.csv --touchpoints other_tp.csv
```

**Run the interactive UI:**

```bash
streamlit run app.py
```

Upload your own `date,value` timeseries CSV and `date,label` touchpoints CSV, or use the bundled example data. Adjust the z-score sensitivity and attribution window from the sidebar.

## Why this matters

Most "did our marketing work?" questions get answered by eyeballing a graph and pattern-matching against memory of what happened that week — which doesn't scale past a couple of data points and is easy to fool yourself with. A cheap, transparent statistical baseline forces the question into the open: is this actually unusual, and what's the nearest plausible cause? Even an imperfect, correlation-only answer beats no answer, as long as it's presented with the uncertainty intact instead of dressed up as certainty.
