"""
generate_data.py

Generates timeseries.csv and touchpoints.csv with engineered spikes for
testing a spike detector against known ground truth.

Ground truth:
  - 2 spikes each occur exactly 1 day after a marketing touchpoint (clear cause).
  - 1 spike has no touchpoint anywhere near it (unexplained).
  - The remaining 5 touchpoints do NOT produce a spike the following day,
    so a detector can't just flag "day after every touchpoint".
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

START = date(2026, 5, 16)
NUM_DAYS = 60

TOUCHPOINTS = [
    (date(2026, 5, 19), "LinkedIn post: launch announcement"),
    (date(2026, 5, 27), "Email newsletter: feature update"),
    (date(2026, 6, 3),  "Twitter/X thread: customer story"),
    (date(2026, 6, 9),  "Press release: Series A funding"),
    (date(2026, 6, 16), "Webinar: product demo"),
    (date(2026, 6, 30), "Reddit AMA: founder Q&A"),
    (date(2026, 7, 6),  "Podcast appearance: industry podcast"),
    (date(2026, 7, 12), "Instagram post: customer testimonial"),
]

# date -> (value, reason) for engineered spikes
SPIKES = {
    date(2026, 5, 20): (24, "caused: 1 day after 'LinkedIn post: launch announcement' (2026-05-19)"),
    date(2026, 6, 10): (26, "caused: 1 day after 'Press release: Series A funding' (2026-06-09)"),
    date(2026, 6, 23): (22, "unexplained: nearest touchpoints are 2026-06-16 (7 days before) "
                            "and 2026-06-30 (7 days after) — no plausible cause"),
}


def _assert_no_unintended_adjacency() -> None:
    """Guard against accidentally placing a spike after a "control" touchpoint.

    Only the two touchpoints explicitly referenced in SPIKES' reasons are
    allowed to be immediately followed by a spike; every other touchpoint
    must NOT have a spike the next day, or the fixture no longer tests
    what it claims to.
    """
    spike_dates = set(SPIKES)
    for tp_date, _label in TOUCHPOINTS:
        next_day = tp_date + timedelta(days=1)
        if next_day not in spike_dates:
            continue
        if tp_date.isoformat() not in SPIKES[next_day][1]:
            raise AssertionError(f"Unintended spike adjacency for touchpoint on {tp_date}")


def _print_ground_truth_summary(rows: list[tuple[str, int]], dates: list[date]) -> None:
    spike_dates = set(SPIKES)
    print(f"Wrote timeseries.csv ({len(rows)} rows) and touchpoints.csv ({len(TOUCHPOINTS)} rows)\n")

    print("=" * 70)
    print("GROUND TRUTH SUMMARY (for verifying the spike detector later)")
    print("=" * 70)
    print(f"\nBaseline: random noise, 2-8 registrations/day, over {NUM_DAYS} days "
          f"({dates[0].isoformat()} to {dates[-1].isoformat()})\n")

    print(f"Touchpoints ({len(TOUCHPOINTS)} total):")
    for d, label in TOUCHPOINTS:
        flag = " <-- precedes a spike" if (d + timedelta(days=1)) in spike_dates else ""
        print(f"  {d.isoformat()}  {label}{flag}")

    print(f"\nEngineered spikes ({len(SPIKES)} total):")
    for d in sorted(SPIKES):
        value, reason = SPIKES[d]
        print(f"  {d.isoformat()}  value={value}  {reason}")

    print("\nExpected detector output:")
    print("  - 2026-05-20 -> spike, explained by touchpoint on 2026-05-19")
    print("  - 2026-06-10 -> spike, explained by touchpoint on 2026-06-09")
    print("  - 2026-06-23 -> spike, UNEXPLAINED (no nearby touchpoint)")
    print("  - All other days (including the mornings after the other 5 "
          "touchpoints) should NOT be flagged as spikes.")


def generate_data(
    output_dir: str | Path = ".",
    seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the engineered timeseries + touchpoints fixtures.

    Writes ``timeseries.csv`` and ``touchpoints.csv`` into `output_dir`
    (created if missing) and also returns them as DataFrames, so callers
    can use the data directly without a round trip through disk.

    Parameters
    ----------
    output_dir : str or Path, default "."
        Directory to write the two CSV files into.
    seed : int, default 42
        Random seed for the baseline noise, for reproducibility.
    verbose : bool, default True
        If True, print the ground-truth summary after generating.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(timeseries_df, touchpoints_df)``.
    """
    _assert_no_unintended_adjacency()

    rng = random.Random(seed)
    dates = [START + timedelta(days=i) for i in range(NUM_DAYS)]

    rows = []
    for d in dates:
        if d in SPIKES:
            value, _ = SPIKES[d]
        else:
            value = rng.randint(2, 8)
        rows.append((d.isoformat(), value))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "timeseries.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "value"])
        writer.writerows(rows)

    with open(out_dir / "touchpoints.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "label"])
        for d, label in TOUCHPOINTS:
            writer.writerow([d.isoformat(), label])

    if verbose:
        _print_ground_truth_summary(rows, dates)

    timeseries_df = pd.DataFrame(rows, columns=["date", "value"])
    touchpoints_df = pd.DataFrame(
        [(d.isoformat(), label) for d, label in TOUCHPOINTS], columns=["date", "label"]
    )
    return timeseries_df, touchpoints_df


if __name__ == "__main__":
    generate_data()
