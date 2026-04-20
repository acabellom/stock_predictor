# 📈 Stock Predictor

> Can combining price action features with NLP-based news sentiment beat a random baseline at predicting intraday stock direction? This project explores that question with a production-grade ML stack — from raw data ingestion to a live inference API.

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-model-9B59B6)](https://lightgbm.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Prefect](https://img.shields.io/badge/Prefect-orchestration-2D6DF6?logo=prefect&logoColor=white)](https://www.prefect.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

---

## 📊 Results

| Model | Target | MAE | RMSE | Directional Accuracy |
|---|---|---|---|---|
| Ridge (baseline) | Close price | 0.2524 | 0.4520 | 47.2% |
| LightGBM (final) | Return | 0.0011 | 0.0019 | **61.2%** |

MAE and RMSE are not directly comparable between models as they use different targets. The key metric is directional accuracy — whether the model correctly predicts whether the next candle goes up or down. A random baseline would score 50%.

---

## 🏗️ Architecture

```
Polygon API ──► data_gathering_flow (Prefect)
                      │
                      ▼
                   MinIO (raw)
                      │
              feature_flow (Prefect)
                      │
                      ▼
                MinIO (processed)
                      │
               training_flow (Prefect)
                      │
                      ▼
              MLflow Model Registry
                      │
               FastAPI /predict
                      │
                Streamlit dashboard
```

---

## 🛠️ Stack

| Layer | Technology |
|---|---|
| Data storage | MinIO (S3-compatible) |
| Orchestration | Prefect |
| Experiment tracking | MLflow |
| Model | LightGBM |
| Serving | FastAPI |
| Dashboard | Streamlit |
| Containerisation | Docker Compose |

---

## ⚙️ Features

**Price features:** lagged returns (t-1, t-2, t-3, t-5, t-10), RSI(14), ATR(14), VWAP distance, realised volatility, relative volume

**Sentiment features:** FinBERT compound score (via Polygon news API), `news_changed` flag (binary — fires when a new headline replaces the previous one)

**Temporal features:** hour, minute, day of week, `is_short_session` flag for NYSE early closes

---

## 📁 Project Structure

```
stock_predictor/
├── src/stock_predictor/
│   ├── data_collector_prices.py   # Polygon OHLCV fetcher
│   ├── data_collector_news.py     # Polygon news + FinBERT sentiment
│   ├── features.py                # Feature engineering pipeline
│   ├── train.py                   # Training loop + MLflow logging
│   ├── tune.py                    # Optuna hyperparameter search
│   ├── predict.py                 # FastAPI inference server
│   ├── models/
│   │   ├── base.py                # Abstract BaseModel interface
│   │   ├── linear.py              # Ridge baseline
│   │   └── lgbm.py                # LightGBM model
│   ├── flows_prefect/
│   │   ├── data_gathering_flow.py # Raw data collection
│   │   ├── feature_flow.py        # Feature engineering
│   │   └── train_flow.py          # Training + MLflow registration
│   └── frontend/
│       └── streamlit.py           # Dashboard
├── tests/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   └── 03_experiments.ipynb
├── Dockerfile
├── Dockerfile.streamlit
└── docker-compose.yml
```

---

## 🚀 Quickstart

**Prerequisites:** Docker, Docker Compose, Polygon API key

### 1. Clone and configure

```bash
git clone https://github.com/acabellom/stock_predictor
cd stock_predictor
cp .env.example .env
# edit .env with your credentials
```

### 2. Start the stack

```bash
docker compose up -d
```

| Service | URL |
|---|---|
| MinIO console | http://localhost:9001 |
| Prefect UI | http://localhost:4200 |
| MLflow UI | http://localhost:5000 |
| API docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

### 3. Collect data

```bash
poetry run python src/stock_predictor/flows_prefect/data_gathering_flow.py
```

### 4. Build features

```bash
poetry run python src/stock_predictor/flows_prefect/feature_flow.py
```

### 5. Train

```bash
poetry run python src/stock_predictor/flows_prefect/train_flow.py
```

### 6. Register the model

Go to `http://localhost:5000` → Models → Register the best run → add alias `staging`

### 7. Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "n_candles": 3}'
```

---

## 🔑 Environment Variables

| Variable | Default |
|---|---|
| `ROOT_USER` | — MinIO root user |
| `ROOT_PASSWORD` | — MinIO root password |
| `POLYGON_API_KEY` | — Polygon.io API key |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` |
| `MODEL_URI` | `models:/lgbm@staging` |
| `PREFECT_API_URL` | `http://localhost:4200/api` |
| `MINIO_ENDPOINT` | `http://localhost:9000` |

---

## 🧪 Tests

```bash
poetry run pytest tests/ -v
```

---

## 🧠 Key Design Decisions

**Target as return instead of price** — predicting the raw close price causes the model to learn that "tomorrow's price ≈ today's price", producing low MAE but no directional signal. Using percentage return as the target forces the model to learn actual price movement patterns.

**TimeSeriesSplit cross-validation** — standard random splits cause data leakage in time series. All validation folds are strictly in the future relative to their training set.

**Baseline first** — Ridge regression was trained before LightGBM. If LightGBM had not clearly outperformed Ridge (61.2% vs 47.2% directional accuracy), it would have indicated a feature engineering problem rather than a model problem.

**`news_changed` flag** — EDA revealed that a single headline covers an average of multiple 10-minute candles. The raw sentiment score is therefore highly autocorrelated and carries little new information. The flag that fires when the headline changes is more informative than the score itself.

---

## ⚠️ Limitations

- Predictions beyond 1–2 candles ahead degrade quickly due to error accumulation in the iterative inference loop
- The model is trained on AAPL only — generalisation to other tickers is not validated
- 61.2% directional accuracy is measured before transaction costs; net profitability depends on spread and commission

---

## 📄 License

MIT
