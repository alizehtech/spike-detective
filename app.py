"""
app.py

Streamlit UI for the spike-detective pipeline: upload a timeseries CSV
and a touchpoints CSV (or use the bundled example), run spike detection
(spike_detector.py) and cause attribution (attribution.py), and inspect
the results as a chart plus per-spike cards.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from attribution import attribute_spikes
from spike_detector import find_spikes

DEFAULT_TIMESERIES_PATH = Path(__file__).parent / "timeseries.csv"
DEFAULT_TOUCHPOINTS_PATH = Path(__file__).parent / "touchpoints.csv"

UNEXPLAINED_COLOR = "#E45756"
EXPLAINED_COLOR = "#F2B701"
LINE_COLOR = "#4C78A8"

st.set_page_config(page_title="Spike Detective", page_icon="📈", layout="wide")


def load_csv(uploaded_file, default_path: Path, label: str) -> pd.DataFrame:
    """Load a CSV from an uploaded file, falling back to a bundled default.

    Stops the Streamlit app with an error if neither an upload nor a
    default file is available.
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if default_path.exists():
        st.sidebar.caption(f"No {label} uploaded — using bundled example `{default_path.name}`.")
        return pd.read_csv(default_path)
    st.error(f"No {label} uploaded and no default `{default_path.name}` found.")
    st.stop()


def build_chart(timeseries: pd.DataFrame, spikes: pd.DataFrame, unexplained_dates: set) -> go.Figure:
    """Build a Plotly line chart of the timeseries with spikes marked as points."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeseries["date"], y=timeseries["value"],
        mode="lines", name="Registrations", line=dict(color=LINE_COLOR),
    ))

    if not spikes.empty:
        colors = [UNEXPLAINED_COLOR if d in unexplained_dates else EXPLAINED_COLOR for d in spikes["date"]]
        labels = ["Unexplained" if d in unexplained_dates else "Explained" for d in spikes["date"]]
        fig.add_trace(go.Scatter(
            x=spikes["date"], y=spikes["value"],
            mode="markers", name="Spikes",
            marker=dict(size=13, color=colors, line=dict(width=1, color="white")),
            text=labels,
            hovertemplate="%{x|%Y-%m-%d}<br>value=%{y}<br>%{text}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Registrations",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig


def render_spike_cards(attributions, spike_context: dict) -> None:
    """Render one card per spike: date, magnitude, and top cause or unexplained flag."""
    if not attributions:
        st.info("No spikes detected in this timeseries.")
        return

    cols_per_row = 3
    for row_start in range(0, len(attributions), cols_per_row):
        row = attributions[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, result in zip(cols, row):
            with col, st.container(border=True):
                st.markdown(f"**{result.spike_date.date()}**")
                context = spike_context.get(result.spike_date, {})
                magnitude = context.get("magnitude")
                baseline_mean = context.get("baseline_mean")
                if magnitude is not None:
                    st.metric("Magnitude vs baseline", f"{magnitude:+.1f}")
                    st.caption(f"value {result.spike_value:g} vs baseline {baseline_mean:.1f}")
                if result.is_unexplained:
                    st.error(result.top_cause)
                else:
                    top = result.candidates[0]
                    st.success(f"Likely cause: {top.label}")
                    st.caption(f"{top.days_before}d before spike · score {top.score}")


def main() -> None:
    st.title("📈 Spike Detective")
    st.caption("Detect anomalous registration spikes and attribute them to marketing touchpoints.")

    with st.sidebar:
        st.header("Data")
        ts_file = st.file_uploader("Timeseries CSV (date, value)", type="csv")
        tp_file = st.file_uploader("Touchpoints CSV (date, label)", type="csv")

        st.header("Detection settings")
        z_threshold = st.slider(
            "Z-score sensitivity", min_value=1.0, max_value=4.0, value=2.0, step=0.1,
            help="Minimum |z-score| against the trailing 7-day baseline for a day to count "
                 "as a spike. Lower = more sensitive (flags more days).",
        )
        attribution_window = st.slider(
            "Attribution window (days)", min_value=1, max_value=5, value=2, step=1,
            help="How many days before a spike to search for a candidate causing touchpoint.",
        )

    timeseries = load_csv(ts_file, DEFAULT_TIMESERIES_PATH, "timeseries CSV")
    touchpoints = load_csv(tp_file, DEFAULT_TOUCHPOINTS_PATH, "touchpoints CSV")

    try:
        spikes = find_spikes(timeseries, z_threshold=z_threshold)
        attributions = attribute_spikes(spikes, touchpoints, window_days=attribution_window)
    except ValueError as exc:
        st.error(f"Could not process the uploaded data: {exc}")
        st.stop()

    timeseries_sorted = timeseries.assign(date=pd.to_datetime(timeseries["date"])).sort_values("date")
    unexplained_dates = {result.spike_date for result in attributions if result.is_unexplained}
    spike_context = spikes.set_index("date")[["magnitude", "baseline_mean"]].to_dict("index")

    n_unexplained = len(unexplained_dates)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Days analyzed", len(timeseries_sorted))
    metric_cols[1].metric("Spikes detected", len(attributions))
    metric_cols[2].metric("Unexplained spikes", n_unexplained)

    st.plotly_chart(
        build_chart(timeseries_sorted, spikes, unexplained_dates),
        width="stretch",
    )

    st.subheader("Spike details")
    render_spike_cards(attributions, spike_context)


if __name__ == "__main__":
    main()
