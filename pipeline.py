"""
pipeline.py

End-to-end spike-detection pipeline: generates (or loads) the
timeseries/touchpoints data, detects spikes, attributes each spike to
its likely marketing touchpoint, and prints one consolidated report.

Usage
-----
    python pipeline.py                    # use existing CSVs, generating them first if missing
    python pipeline.py --regenerate       # force-regenerate the fixture CSVs before running
    python pipeline.py --timeseries other.csv --touchpoints other_tp.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from attribution import attribute_spikes
from generate_data import generate_data
from spike_detector import find_spikes


def run_pipeline(
    timeseries_path: str | Path = "timeseries.csv",
    touchpoints_path: str | Path = "touchpoints.csv",
    regenerate: bool = False,
    baseline_window: int = 7,
    z_threshold: float = 2.0,
    changepoint_pen: float = 3.0,
    attribution_window_days: int = 2,
    verbose: bool = True,
):
    """Run generate -> detect -> attribute and report the results.

    Parameters
    ----------
    timeseries_path, touchpoints_path : str or Path
        Where to read the input CSVs from (and, if regenerating, where
        to write them).
    regenerate : bool, default False
        If True, always regenerate the fixture CSVs first via
        ``generate_data.generate_data``. If False, the fixtures are
        only generated when one of the input files doesn't exist yet.
    baseline_window : int, default 7
        Trailing-day window for the spike detector's z-score baseline.
    z_threshold : float, default 2.0
        Minimum |z-score| for a day to count as a spike.
    changepoint_pen : float, default 3.0
        Penalty for the ruptures Pelt changepoint search.
    attribution_window_days : int, default 2
        How many days before a spike to search for a causing touchpoint.
    verbose : bool, default True
        If True, print the consolidated report to stdout.

    Returns
    -------
    tuple[pd.DataFrame, list[attribution.Attribution]]
        ``(spikes, attributions)`` for programmatic use.
    """
    ts_path = Path(timeseries_path)
    tp_path = Path(touchpoints_path)

    if regenerate or not ts_path.exists() or not tp_path.exists():
        if verbose:
            print(f"Generating fixture data -> {ts_path}, {tp_path}\n")
        generate_data(output_dir=ts_path.parent or ".", verbose=False)

    timeseries = pd.read_csv(ts_path)
    touchpoints = pd.read_csv(tp_path)

    spikes = find_spikes(
        timeseries,
        window=baseline_window,
        z_threshold=z_threshold,
        changepoint_pen=changepoint_pen,
    )
    attributions = attribute_spikes(spikes, touchpoints, window_days=attribution_window_days)

    if verbose:
        _print_report(attributions)

    return spikes, attributions


def _print_report(attributions) -> None:
    print("=" * 70)
    print("SPIKE ATTRIBUTION REPORT")
    print("=" * 70)

    if not attributions:
        print("\nNo spikes detected.")
        return

    for result in attributions:
        print(f"\n{result.spike_date.date()}  value={result.spike_value:g}")
        if result.is_unexplained:
            print(f"    {result.top_cause}")
        else:
            for rank, candidate in enumerate(result.candidates, start=1):
                marker = "*" if rank == 1 else " "
                print(
                    f"  {marker} #{rank} {candidate.label!r} "
                    f"— {candidate.days_before}d before (score={candidate.score})"
                )

    n_unexplained = sum(1 for r in attributions if r.is_unexplained)
    print(f"\n{len(attributions)} spike(s) total, {n_unexplained} unexplained.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end spike detection + attribution pipeline."
    )
    parser.add_argument("--timeseries", default="timeseries.csv", help="Path to timeseries CSV.")
    parser.add_argument("--touchpoints", default="touchpoints.csv", help="Path to touchpoints CSV.")
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Regenerate the fixture CSVs before running, even if they already exist.",
    )
    parser.add_argument("--window", type=int, default=7, help="Baseline window in days (default: 7).")
    parser.add_argument("--z-threshold", type=float, default=2.0, help="Z-score spike threshold (default: 2.0).")
    parser.add_argument(
        "--changepoint-pen", type=float, default=3.0,
        help="Ruptures Pelt changepoint penalty (default: 3.0).",
    )
    parser.add_argument(
        "--attribution-window", type=int, default=2,
        help="Days before a spike to search for a causing touchpoint (default: 2).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        timeseries_path=args.timeseries,
        touchpoints_path=args.touchpoints,
        regenerate=args.regenerate,
        baseline_window=args.window,
        z_threshold=args.z_threshold,
        changepoint_pen=args.changepoint_pen,
        attribution_window_days=args.attribution_window,
    )
