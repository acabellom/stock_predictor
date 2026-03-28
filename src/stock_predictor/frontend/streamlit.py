import os
import requests
import mlflow
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from stock_predictor.utils import create_s3_client, get_latest_data_s3

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
API_URL = os.getenv("API_URL", "http://localhost:8000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .metric-card {
        background: #0a0a0a;
        border: 1px solid #222;
        border-radius: 4px;
        padding: 1.2rem 1.5rem;
    }
    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #555;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px;
        font-weight: 500;
        color: #f0f0f0;
    }
    .metric-good { color: #00ff88; }
    .metric-bad  { color: #ff4455; }
    h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 20px;
        font-weight: 500;
        letter-spacing: 0.05em;
        color: #f0f0f0;
    }
    .stSelectbox label, .stSlider label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #555;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_historical(ticker: str, n_candles: int = 100) -> pd.DataFrame:
    s3 = create_s3_client()
    df = get_latest_data_s3(s3, f"{ticker.lower()}-features")
    df = df[df.index.hour.isin(range(9, 20))]
    return df.tail(n_candles)


def fetch_predictions(ticker: str, n_candles: int) -> list:
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"ticker": ticker, "n_candles": n_candles},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["predictions"]
    except Exception as e:
        st.error(f"API error: {e}")
        return []


def load_mlflow_metrics(experiment_name: str = "stock-prediction") -> dict:
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            return {}
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=10,
        )
        if not runs:
            return {}
        metrics = {}
        for run in runs:
            m = run.data.metrics
            model = run.data.params.get("model", "unknown")
            if model not in metrics:
                metrics[model] = m
        return metrics
    except Exception:
        return {}


st.markdown("# stock_predictor / dashboard")
st.markdown("---")

col_left, col_right = st.columns([1, 3])

with col_left:
    ticker = st.selectbox("ticker", ["AAPL", "TSLA"], index=0)
    n_hist = st.slider("historical candles", 50, 300, 100, step=10)
    n_pred = st.slider("candles to predict", 1, 10, 3)
    run_btn = st.button("run prediction →")

    st.markdown("---")
    st.markdown("#### flows")

    if st.button("run features flow →"):
        with st.spinner("running features flow..."):
            resp = requests.post(f"{API_URL}/trigger/features?ticker={ticker}")
            if resp.status_code == 200:
                st.success("features flow completed")
            else:
                st.error(f"error: {resp.json()['detail']}")

    if st.button("run training flow →"):
        with st.spinner("running training flow..."):
            resp = requests.post(f"{API_URL}/trigger/train?ticker={ticker}")
            if resp.status_code == 200:
                st.success("training flow completed")
            else:
                st.error(f"error: {resp.json()['detail']}")

with col_right:
    if run_btn:
        with st.spinner("loading data..."):
            hist = load_historical(ticker, n_hist)
            preds = fetch_predictions(ticker, n_pred)

        if hist.empty:
            st.error("No historical data found.")
        else:
            hist_prices = hist["c"]
            hist_times = hist.index.tolist()

            pred_times = [p["timestamp"] for p in preds]
            pred_returns = [p["predicted_return"] for p in preds]

            last_price = float(hist_prices.iloc[-1])
            pred_prices = []
            current = last_price
            for r in pred_returns:
                current = current * (1 + r)
                pred_prices.append(round(current, 6))

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=hist_times,
                    y=hist_prices,
                    mode="lines",
                    name="historical",
                    line=dict(color="#888888", width=1.5),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[hist_times[-1]] + pred_times,
                    y=[last_price] + pred_prices,
                    mode="lines+markers",
                    name="predicted",
                    line=dict(color="#00ff88", width=2, dash="dot"),
                    marker=dict(color="#00ff88", size=6),
                )
            )

            fig.update_layout(
                paper_bgcolor="#0a0a0a",
                plot_bgcolor="#0a0a0a",
                font=dict(family="IBM Plex Mono", color="#555", size=11),
                xaxis=dict(
                    gridcolor="#1a1a1a", showline=False, tickfont=dict(color="#444")
                ),
                yaxis=dict(
                    gridcolor="#1a1a1a", showline=False, tickfont=dict(color="#444")
                ),
                legend=dict(
                    bgcolor="#0a0a0a", bordercolor="#222", font=dict(color="#888")
                ),
                margin=dict(l=0, r=0, t=20, b=0),
                height=380,
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### predictions")
            pred_df = pd.DataFrame(
                [
                    {
                        "timestamp": pd.Timestamp(p["timestamp"]).strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        "predicted_return": f"{p['predicted_return']:.6f}",
                        "predicted_direction": p["predicted_direction"],
                    }
                    for p in preds
                ]
            )
            st.dataframe(
                pred_df.style.map(
                    lambda v: "color: #00ff88" if v == "up" else "color: #ff4455",
                    subset=["predicted_direction"],
                ),
                use_container_width=True,
                hide_index=True,
            )

st.markdown("---")
st.markdown("#### model metrics")

metrics = load_mlflow_metrics()

if metrics:
    cols = st.columns(len(metrics) * 3)
    i = 0
    for model_name, m in metrics.items():
        mae = m.get("avg_mae", m.get("mae", None))
        rmse = m.get("avg_rmse", m.get("rmse", None))
        dir_acc = m.get("avg_directional_accuracy", m.get("directional_accuracy", None))

        with cols[i]:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">{model_name} / mae</div>
                <div class="metric-value">{round(mae, 5) if mae else "—"}</div>
            </div>""",
                unsafe_allow_html=True,
            )
        i += 1

        with cols[i]:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">{model_name} / rmse</div>
                <div class="metric-value">{round(rmse, 5) if rmse else "—"}</div>
            </div>""",
                unsafe_allow_html=True,
            )
        i += 1

        with cols[i]:
            color_class = "metric-good" if dir_acc and dir_acc > 0.55 else "metric-bad"
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">{model_name} / dir. accuracy</div>
                <div class="metric-value {color_class}">{round(dir_acc, 4) if dir_acc else "—"}</div>
            </div>""",
                unsafe_allow_html=True,
            )
        i += 1
else:
    st.info("No MLflow runs found. Run train_flow first.")
