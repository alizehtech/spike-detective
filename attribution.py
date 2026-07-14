"""
attribution.py

Attributes each detected spike (as returned by ``spike_detector.find_spikes``)
to the marketing touchpoint(s) most likely to have caused it.

For every spike, candidate touchpoints are any touchpoint dated on or
before the spike, within a configurable lookback window. Candidates are
ranked by recency: a touchpoint the day before a spike is a stronger
suspect than one from a week earlier. Spikes with no candidate
touchpoint in the window are explicitly marked as unexplained rather
than silently dropped, so gaps in tracked marketing activity stay
visible instead of being hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

DEFAULT_WINDOW_DAYS = 2
UNEXPLAINED_LABEL = "Unexplained — no tracked touchpoint in window."


@dataclass
class Candidate:
    """A single touchpoint proposed as a possible cause of a spike."""

    touchpoint_date: pd.Timestamp
    label: str
    days_before: int
    score: float


@dataclass
class Attribution:
    """Ranked attribution result for a single spike."""

    spike_date: pd.Timestamp
    spike_value: float
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def top_cause(self) -> str:
        """Best-guess cause label, or the unexplained marker if none."""
        if not self.candidates:
            return UNEXPLAINED_LABEL
        return self.candidates[0].label

    @property
    def is_unexplained(self) -> bool:
        """True if no touchpoint was found in the lookback window."""
        return not self.candidates


def _validate_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a spikes DataFrame, coercing ``date`` to datetime."""
    required = {"date", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"spikes DataFrame is missing required column(s): {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def _validate_touchpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a touchpoints DataFrame, coercing ``date`` to datetime."""
    required = {"date", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"touchpoints DataFrame is missing required column(s): {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def _recency_score(days_before: int) -> float:
    """Score a candidate touchpoint by recency.

    Uses simple inverse decay so closer touchpoints always outrank
    farther ones, independent of the configured window size: a
    same-day touchpoint scores 1.0, one day before scores 0.5, two
    days before scores ~0.333, and so on.
    """
    return round(1.0 / (1.0 + days_before), 4)


def attribute_spikes(
    spikes: pd.DataFrame,
    touchpoints: pd.DataFrame,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> list[Attribution]:
    """Attribute each spike to candidate touchpoint(s) that preceded it.

    Parameters
    ----------
    spikes : pd.DataFrame
        Flagged spikes, e.g. the output of ``spike_detector.find_spikes``.
        Must contain ``date`` and ``value`` columns.
    touchpoints : pd.DataFrame
        Marketing touchpoints with ``date`` and ``label`` columns.
    window_days : int, default 2
        How many days before a spike to search for candidate
        touchpoints. A touchpoint on the spike date itself (0 days
        before) also counts as a candidate.

    Returns
    -------
    list[Attribution]
        One :class:`Attribution` per input spike, in chronological
        order, each holding its candidates ranked best-first by
        recency score. Spikes with no candidates report
        ``top_cause == UNEXPLAINED_LABEL`` and ``is_unexplained is True``.

    Notes
    -----
    Handles empty or missing input gracefully: an empty/None ``spikes``
    returns an empty list; an empty/None ``touchpoints`` makes every
    spike unexplained rather than raising.
    """
    if spikes is None or len(spikes) == 0:
        return []

    spikes_df = _validate_spikes(spikes)

    if touchpoints is None or len(touchpoints) == 0:
        tp_df = pd.DataFrame(columns=["date", "label"])
    else:
        tp_df = _validate_touchpoints(touchpoints)

    results = []
    for _, spike_row in spikes_df.iterrows():
        spike_date = spike_row["date"]
        window_start = spike_date - pd.Timedelta(days=window_days)

        in_window = tp_df[(tp_df["date"] >= window_start) & (tp_df["date"] <= spike_date)]

        candidates = [
            Candidate(
                touchpoint_date=tp_row["date"],
                label=tp_row["label"],
                days_before=(spike_date - tp_row["date"]).days,
                score=_recency_score((spike_date - tp_row["date"]).days),
            )
            for _, tp_row in in_window.iterrows()
        ]
        candidates.sort(key=lambda c: (-c.score, c.days_before))

        results.append(
            Attribution(
                spike_date=spike_date,
                spike_value=spike_row["value"],
                candidates=candidates,
            )
        )

    return results


def attribution_summary(results: list[Attribution]) -> pd.DataFrame:
    """Flatten attribution results into a long-format DataFrame.

    One row per (spike, candidate) pair, ordered by rank within each
    spike (rank 1 = top cause). Unexplained spikes get a single row
    with ``rank=0``, ``cause`` set to :data:`UNEXPLAINED_LABEL`, and
    ``NaN``/``None`` touchpoint fields.

    Parameters
    ----------
    results : list[Attribution]
        Output of :func:`attribute_spikes`.

    Returns
    -------
    pd.DataFrame
        Columns: ``spike_date``, ``spike_value``, ``rank``, ``cause``,
        ``touchpoint_date``, ``days_before``, ``score``.
    """
    columns = [
        "spike_date", "spike_value", "rank", "cause",
        "touchpoint_date", "days_before", "score",
    ]

    rows = []
    for result in results:
        if result.is_unexplained:
            rows.append({
                "spike_date": result.spike_date,
                "spike_value": result.spike_value,
                "rank": 0,
                "cause": UNEXPLAINED_LABEL,
                "touchpoint_date": pd.NaT,
                "days_before": None,
                "score": None,
            })
        else:
            for rank, c in enumerate(result.candidates, start=1):
                rows.append({
                    "spike_date": result.spike_date,
                    "spike_value": result.spike_value,
                    "rank": rank,
                    "cause": c.label,
                    "touchpoint_date": c.touchpoint_date,
                    "days_before": c.days_before,
                    "score": c.score,
                })

    return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    import sys

    from spike_detector import find_spikes

    ts_path = sys.argv[1] if len(sys.argv) > 1 else "timeseries.csv"
    tp_path = sys.argv[2] if len(sys.argv) > 2 else "touchpoints.csv"

    ts = pd.read_csv(ts_path)
    tp = pd.read_csv(tp_path)

    spikes = find_spikes(ts)
    results = attribute_spikes(spikes, tp)

    for r in results:
        print(f"{r.spike_date.date()} (value={r.spike_value}): {r.top_cause}")
        for c in r.candidates:
            print(
                f"    - {c.label!r} on {c.touchpoint_date.date()} "
                f"({c.days_before}d before, score={c.score})"
            )
