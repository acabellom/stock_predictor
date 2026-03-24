import os
from fastapi import FastAPI
import mlflow.lightgbm
from pydantic import BaseModel

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
    predicted_direction: int


class PedrictionResponse(BaseModel):
    ticker: str
    model_uri: str
    prediction: list[CandlePrediction]


@app.get("/health")
def health_check():
    return {"status": "ok", "model_uri": MODEL_URI}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("stock_predictor.predict:app", host="0.0.0.0", port=8000, reload=True)
