import os
from fastapi import FastAPI, HTTPException
import mlflow.lightgbm
from pydantic import BaseModel
import pandas_market_calendars as mcal
import pytz
from datetime import datetime
import pandas as pd
from stock_predictor.utils import create_s3_client, get_latest_data_s3
from stock_predictor.train import FEATURE_COLS

os.environ["PREFECT_API_URL"] = os.getenv(
    "PREFECT_API_URL", "http://localhost:4200/api"
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/lgbm@staging")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.lightgbm.load_model(MODEL_URI)

app = FastAPI(title="Stock Price Predictor API", version="1.0")


class PredictionRequest(BaseModel):
    ticker: str
    n_candles: int = 10


class CandlePrediction(BaseModel):
    timestamp: str
    predicted_return: float
    predicted_direction: str


class PredictionResponse(BaseModel):
    ticker: str
    model_uri: str
    predictions: list[CandlePrediction]


def get_next_valid_timestamp(current: pd.Timestamp) -> pd.Timestamp:
    """
    Given a timestamp, return the next valid 10-minute candle timestamp.
    Skips non-trading hours, weekends and NYSE holidays.

    Args:
        current: Reference timestamp to advance from.
    Returns:
        Next valid candle timestamp within NYSE trading hours.
    """
    nyse = mcal.get_calendar("NYSE")
    next_ts = current + pd.Timedelta(minutes=10)

    schedule = nyse.schedule(
        start_date=next_ts.strftime("%Y-%m-%d"),
        end_date=(next_ts + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
    )

    for _, row in schedule.iterrows():
        market_open = pd.Timestamp(row["market_open"].to_pydatetime())
        market_close = pd.Timestamp(row["market_close"].to_pydatetime())

        if next_ts < market_open:
            return market_open

        if market_open <= next_ts <= market_close:
            return next_ts

    return pd.Timestamp(schedule.iloc[0]["market_open"])


def build_inference_features(ticker: str) -> pd.DataFrame:
    """Fetches the latest data for the given ticker and applies the same feature engineering steps as during training.
    Args:
        ticker (str): Stock ticker symbol
    Returns:
        pd.DataFrame: DataFrame with features ready for model inference.
    """
    s3_client = create_s3_client()
    df = get_latest_data_s3(s3_client, f"{ticker.lower()}-features")
    return df


@app.get("/health")
def health_check():
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict the return of the next n_candles for a given ticker.
    Each prediction is the expected return of one 10-minute candle.
    Predictions are generated iteratively — each predicted candle
    feeds into the features of the next one.

    Args:
        ticker: Ticker symbol (e.g. AAPL, TSLA).
        n_candles: Number of future candles to predict.
    Returns:
        List of predicted returns and directions per candle.
    """
    try:
        df = build_inference_features(request.ticker)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Could not load data for {request.ticker}: {str(e)}",
        )
    if len(df) < 10:
        raise HTTPException(
            status_code=400, detail="Not enough data to generate predictions."
        )
    predictions = []
    current_df = df.copy()
    now = pd.Timestamp(datetime.now(pytz.timezone("America/New_York")))
    next_timestamp = get_next_valid_timestamp(now)

    for _ in range(request.n_candles):
        last_row = current_df[FEATURE_COLS].iloc[-1:]
        predicted_return = model.predict(last_row)[0]
        direction = "up" if predicted_return > 0 else "down"
        predictions.append(
            CandlePrediction(
                timestamp=next_timestamp.isoformat(),
                predicted_return=predicted_return,
                predicted_direction=direction,
            )
        )
        next_row = current_df.iloc[[-1]].copy()
        next_row.index = [next_timestamp]
        next_row["c"] = current_df["c"].iloc[-1] * (1 + predicted_return)
        last_returns = current_df["c"].pct_change().iloc[-11:]
        next_row["c1"] = last_returns.iloc[-1]
        next_row["c2"] = last_returns.iloc[-2] if len(last_returns) > 1 else 0
        next_row["c3"] = last_returns.iloc[-3] if len(last_returns) > 2 else 0
        next_row["c5"] = last_returns.iloc[-5] if len(last_returns) > 4 else 0
        next_row["c10"] = last_returns.iloc[-10] if len(last_returns) > 9 else 0
        next_row["volatility"] = current_df["c"].pct_change().rolling(10).std().iloc[-1]
        next_row["vwap_dist"] = (next_row["c"] - next_row["vw"]) / next_row["vw"]
        next_row["rel_volume"] = (
            next_row["v"] / current_df["v"].rolling(20).mean().iloc[-1]
        )
        current_df = pd.concat([current_df, next_row])
        next_timestamp = get_next_valid_timestamp(next_timestamp)

    return PredictionResponse(
        ticker=request.ticker,
        model_uri=MODEL_URI,
        predictions=predictions,
    )


@app.post("/trigger/features")
def trigger_features(ticker: str = "AAPL"):
    """Trigger the features flow for a given ticker."""
    try:
        from stock_predictor.flows_prefect.feature_flow import feature_flow

        feature_flow(ticker=ticker)
        return {"status": "ok", "ticker": ticker, "flow": "features_flow"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trigger/train")
def trigger_train(ticker: str = "AAPL", tune_first: bool = False):
    """Trigger the training flow for a given ticker."""
    try:
        from stock_predictor.flows_prefect.train_flow import train_flow

        train_flow(ticker=ticker, tune_first=tune_first)
        return {
            "status": "ok",
            "ticker": ticker,
            "flow": "train_flow",
            "tune_first": tune_first,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("stock_predictor.predict:app", host="0.0.0.0", port=8000, reload=True)
