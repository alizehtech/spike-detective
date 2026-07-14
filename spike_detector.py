"""
spike_detector.py

Detects anomalous spikes in a daily event-count time series by combining
two complementary signals:

1. **Changepoint detection** (``ruptures``, Pelt search with an RBF cost
   function) to find structural shifts in the series' distribution.
2. **Rolling 7-day baseline z-scores** to flag individual days whose
   value is a statistical outlier relative to its local neighborhood.

The primary trigger for a "spike" is the z-score outlier test; the
changepoint signal is attached to each flagged day as corroborating
evidence (``is_changepoint``), since a single day's z-score can be noisy
on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import ruptures as rpt
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "spike_detector requires the 'ruptures' package. Install it with "
        "`pip install ruptures`."
    ) from exc


DEFAULT_BASELINE_WINDOW = 7
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_CHANGEPOINT_PENALTY = 3.0
# ruptures' Pelt search needs a reasonable number of points to be meaningful;
# below this we skip changepoint detection rather than error out.
MIN_POINTS_FOR_CHANGEPOINTS = 10

SPIKE_COLUMNS = [
    "date",
    "value",
    "baseline_mean",
    "baseline_std",
    "z_score",
    "magnitude",
    "is_changepoint",
]


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the input DataFrame.

    Ensures the required ``date``/``value`` columns exist, coerces
    ``date`` to datetime and ``value`` to numeric, sorts chronologically,
    and returns a fresh, reindexed copy (the original is left untouched).
    """
    required = {"date", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required column(s): {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="raise")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def detect_changepoints(
    values: np.ndarray,
    pen: float = DEFAULT_CHANGEPOINT_PENALTY,
    min_size: int = 2,
) -> list[int]:
    """Detect changepoints in a 1-D signal using ruptures' Pelt/RBF model.

    Parameters
    ----------
    values : np.ndarray
        The signal to search for changepoints.
    pen : float
        Penalty passed to Pelt's ``predict()``; higher values yield fewer,
        more conservative changepoints.
    min_size : int
        Minimum number of samples between changepoints, forwarded to
        ``rpt.Pelt``.

    Returns
    -------
    list[int]
        Sorted 0-based indices marking the start of each new regime
        (i.e. the first day after a detected shift). The trailing
        endpoint that ruptures always appends is excluded.

    Notes
    -----
    Returns an empty list (rather than raising) when there are too few
    points for a meaningful search, or if ruptures fails internally on a
    degenerate/short signal.
    """
    n = len(values)
    if n < max(MIN_POINTS_FOR_CHANGEPOINTS, min_size * 2):
        return []

    signal = np.asarray(values, dtype=float).reshape(-1, 1)
    try:
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
        breakpoints = algo.predict(pen=pen)
    except Exception:
        return []

    # ruptures always includes len(signal) as the final breakpoint; drop it.
    return sorted(bp for bp in breakpoints if bp < n)


def compute_rolling_baseline(
    df: pd.DataFrame,
    window: int = DEFAULT_BASELINE_WINDOW,
) -> pd.DataFrame:
    """Compute a trailing rolling mean/std baseline and z-score per day.

    For each day, the baseline is the mean and standard deviation of the
    preceding ``window`` days, not including the day itself. Days with
    fewer than 2 prior observations (e.g. the start of a short series)
    get ``NaN`` baseline stats and z-score instead of raising.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date`` and ``value`` columns, already sorted
        chronologically (see :func:`_validate_input`).
    window : int
        Number of preceding days to use as the baseline window.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with added ``baseline_mean``, ``baseline_std``,
        and ``z_score`` columns.
    """
    out = df.copy()
    trailing = out["value"].shift(1).rolling(window=window, min_periods=2)
    out["baseline_mean"] = trailing.mean()
    out["baseline_std"] = trailing.std(ddof=0)
    # A perfectly flat baseline (std == 0) would divide-by-zero; treat it
    # as "z-score undefined" rather than +/-inf.
    safe_std = out["baseline_std"].replace(0, np.nan)
    out["z_score"] = (out["value"] - out["baseline_mean"]) / safe_std
    return out


def find_spikes(
    df: pd.DataFrame,
    window: int = DEFAULT_BASELINE_WINDOW,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    changepoint_pen: float = DEFAULT_CHANGEPOINT_PENALTY,
) -> pd.DataFrame:
    """Flag spike days in a daily event-count time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with ``date`` and ``value`` columns. Row order does
        not matter; the data is sorted chronologically internally.
    window : int, default 7
        Number of preceding days used to compute each day's baseline.
    z_threshold : float, default 2.0
        Minimum absolute z-score for a day to be flagged as a spike.
    changepoint_pen : float, default 3.0
        Penalty for the Pelt changepoint search (higher = fewer
        changepoints detected).

    Returns
    -------
    pd.DataFrame
        One row per flagged spike day, sorted chronologically, with
        columns:

        - ``date``
        - ``value``
        - ``baseline_mean`` -- mean of the preceding ``window`` days
        - ``baseline_std`` -- std dev of the preceding ``window`` days
        - ``z_score`` -- (value - baseline_mean) / baseline_std
        - ``magnitude`` -- value - baseline_mean
        - ``is_changepoint`` -- True if a ruptures-detected changepoint
          also starts on this date (corroborating evidence)

        An empty (but correctly-columned) DataFrame is returned if no
        spikes are found or the input is too short/empty to analyze.

    Notes
    -----
    Handles short time series gracefully rather than raising:

    - An empty or single-row input returns an empty result.
    - Days without enough preceding history for a baseline (fewer than
      2 prior points) simply can't be flagged (NaN z-score).
    - Changepoint detection is skipped below
      :data:`MIN_POINTS_FOR_CHANGEPOINTS` rows; the z-score signal still
      runs normally.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=SPIKE_COLUMNS)

    data = _validate_input(df)

    if len(data) < 2:
        return pd.DataFrame(columns=SPIKE_COLUMNS)

    baselined = compute_rolling_baseline(data, window=window)

    changepoint_idx = set(
        detect_changepoints(baselined["value"].to_numpy(), pen=changepoint_pen)
    )

    is_outlier = baselined["z_score"].abs() > z_threshold
    spikes = baselined.loc[is_outlier].copy()
    spikes["magnitude"] = spikes["value"] - spikes["baseline_mean"]
    spikes["is_changepoint"] = spikes.index.isin(changepoint_idx)

    return spikes[SPIKE_COLUMNS].reset_index(drop=True)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "timeseries.csv"
    ts = pd.read_csv(path)
    result = find_spikes(ts)
    if result.empty:
        print(f"No spikes detected in {path}.")
    else:
        print(result.to_string(index=False))
