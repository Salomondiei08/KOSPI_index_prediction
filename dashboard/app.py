from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.csv"
FORECAST_PATH = REPORTS_DIR / "forecast_dec_2025.csv"

CUSTOM_CSS = """
<style>
.hero {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    border-radius: 18px;
    padding: 24px 32px;
    color: #f8fafc;
    margin-bottom: 1.5rem;
}
.hero h1 {
    margin-bottom: 0.2rem;
}
.hero p {
    opacity: 0.85;
}
.stat-card {
    background: #0f172a;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
    color: #e2e8f0;
    width: 100%;
}
.stat-card h4 {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
    color: #94a3b8;
}
.stat-card p {
    font-size: 1.6rem;
    margin: 0;
    font-weight: 600;
}
.stat-card small {
    color: #38bdf8;
}
.section-title {
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}
</style>
"""


@st.cache_data(show_spinner=False)
def load_metrics() -> Optional[Dict[str, Dict[str, float]]]:
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_predictions() -> Optional[pd.DataFrame]:
    if not PREDICTIONS_PATH.exists():
        return None
    df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    return df


@st.cache_data(show_spinner=False)
def load_forecast() -> Optional[pd.DataFrame]:
    if not FORECAST_PATH.exists():
        return None
    df = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    return df


def main() -> None:
    st.set_page_config(page_title="KOSPI Forecast Dashboard", layout="wide", page_icon="📈")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if st.sidebar.button("Reload Results"):
        load_metrics.clear()
        load_predictions.clear()
        load_forecast.clear()
        st.experimental_rerun()

    metrics_payload = load_metrics()
    predictions_df = load_predictions()
    forecast_df = load_forecast()

    if not metrics_payload or predictions_df is None:
        st.warning(
            "No evaluation artifacts detected. Run `python main.py` or the training pipeline "
            "to populate `reports/`."
        )
        return

    best_model, _ = min(
        metrics_payload.items(), key=lambda item: item[1]["RMSE"]
    )

    st.markdown(
        """
        <div class="hero">
            <h1>KOSPI Forecast Command Center</h1>
            <p>Monitor training quality, drill into residuals, and preview the December 2025 outlook.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        model_choice = st.selectbox("Focus model", list(metrics_payload.keys()))
        st.markdown("Forecast horizon: **Dec 1–5, 2025**")
        st.info(
            "Upload custom CSV files from backtests or replace `reports/predictions.csv` "
            "to update the visuals."
        )

    current_metrics = metrics_payload[model_choice]
    stat_cols = st.columns(4)
    for col, (label, value) in zip(
        stat_cols,
        [
            ("RMSE", f"{current_metrics['RMSE']:.2f}"),
            ("MAE", f"{current_metrics['MAE']:.2f}"),
            ("Directional Accuracy", f"{current_metrics['DirectionalAccuracy']:.2%}"),
            ("Best Performer", best_model.upper()),
        ],
    ):
        col.markdown(
            f"""
            <div class="stat-card">
                <h4>{label}</h4>
                <p>{value}</p>
                <small>Live validation</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    filtered = predictions_df[predictions_df["model"] == model_choice].copy()
    filtered.sort_values("date", inplace=True)
    filtered["residual"] = filtered["predicted"] - filtered["actual"]

    perf_tab, residual_tab, forecast_tab = st.tabs(
        ["Performance", "Error Analysis", "Dec 2025 Outlook"]
    )

    with perf_tab:
        fig_line = px.line(
            filtered,
            x="date",
            y=["actual", "predicted"],
            labels={"value": "KOSPI", "variable": "Series", "date": "Date"},
            title=f"{model_choice.upper()} · Actual vs Predicted",
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.download_button(
            "Download chart (HTML)",
            data=fig_line.to_html().encode("utf-8"),
            file_name=f"{model_choice}_chart.html",
            mime="text/html",
        )
        st.markdown("#### Raw predictions (latest 200 rows)")
        st.dataframe(filtered.tail(200), use_container_width=True)

    with residual_tab:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_residual = px.histogram(
                filtered,
                x="residual",
                nbins=30,
                title="Residual distribution",
                labels={"residual": "Prediction error"},
            )
            st.plotly_chart(fig_residual, use_container_width=True)
        with col_b:
            scatter = px.scatter(
                filtered,
                x="actual",
                y="predicted",
                trendline="ols",
                title="Actual vs Predicted Scatter",
            )
            st.plotly_chart(scatter, use_container_width=True)

        st.markdown("#### Upload custom prediction CSV")
        uploaded = st.file_uploader("Upload CSV with columns: date, actual, predicted", type="csv")
        if uploaded:
            custom_df = pd.read_csv(uploaded, parse_dates=["date"])
            st.dataframe(custom_df.head(100), use_container_width=True)

    with forecast_tab:
        if forecast_df is None:
            st.info("Run `python main.py` to generate the December 2025 forecast.")
        else:
            forecast_choice = st.selectbox(
                "Forecast model",
                sorted(forecast_df["model"].unique()),
                key="forecast_model",
            )
            forecast_subset = (
                forecast_df[forecast_df["model"] == forecast_choice]
                .sort_values("date")
                .copy()
            )
            fig_forecast = px.line(
                forecast_subset,
                x="date",
                y="predicted_close",
                markers=True,
                labels={"predicted_close": "Predicted close"},
                title=f"{forecast_choice.upper()} · Dec 2025 projection",
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            st.dataframe(forecast_subset, use_container_width=True)
            st.download_button(
                "Download Dec 2025 forecast",
                data=forecast_subset.to_csv(index=False).encode("utf-8"),
                file_name=f"{forecast_choice}_forecast_dec2025.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
