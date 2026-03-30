import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def make_feature_df(n: int = 50) -> pd.DataFrame:
    """
    Build a minimal DataFrame that mimics the processed feature data loaded
    from MinIO. Contains all columns that predict.py reads during inference,
    including FEATURE_COLS, lag columns and auxiliary price columns.
    Uses n=50 rows — enough for lag and rolling calculations but fast to build.
    """
    import numpy as np
    from stock_predictor.train import FEATURE_COLS

    rng = np.random.default_rng(42)
    data = {col: rng.standard_normal(n) for col in FEATURE_COLS}
    data["vw"] = rng.uniform(150, 200, n)
    data["c"] = rng.uniform(150, 200, n)
    data["v"] = rng.uniform(1000, 5000, n)
    return pd.DataFrame(
        data,
        index=pd.date_range("2024-01-01 09:30", periods=n, freq="10min"),
    )


@pytest.fixture
def client():
    """
    FastAPI TestClient fixture. Patches mlflow.lightgbm.load_model at import
    time so the app can be instantiated without a real MLflow server or model.
    The mock model always predicts 0.0001 (a small positive return).
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.0001]

    with patch("mlflow.lightgbm.load_model", return_value=mock_model):
        with patch("mlflow.set_tracking_uri"):
            from stock_predictor import predict
            import importlib

            importlib.reload(predict)
            from stock_predictor.predict import app

            yield TestClient(app)


# health_check


def test_health_check_returns_200(client):
    """
    GET /health must return HTTP 200. This endpoint is used by Docker and
    monitoring tools to verify the service is alive.
    """
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_check_returns_status_ok(client):
    """
    GET /health must return a JSON body with 'status': 'ok'.
    Any other value would indicate the service is degraded.
    """
    resp = client.get("/health")
    assert resp.json()["status"] == "ok"


def test_get_next_valid_timestamp_returns_timestamp():
    """
    get_next_valid_timestamp() must return a pd.Timestamp.
    Returning any other type would break the isoformat() call in /predict.
    """
    from stock_predictor.predict import get_next_valid_timestamp
    import pandas as pd

    ts = pd.Timestamp("2024-03-18 14:00:00", tz="UTC")
    result = get_next_valid_timestamp(ts)
    assert isinstance(result, pd.Timestamp)


def test_get_next_valid_timestamp_advances_time():
    """
    get_next_valid_timestamp() must always return a timestamp strictly
    after the input. Returning the same or earlier timestamp would cause
    an infinite loop in the /predict endpoint.
    """
    from stock_predictor.predict import get_next_valid_timestamp
    import pandas as pd

    ts = pd.Timestamp("2024-03-18 14:00:00", tz="UTC")
    result = get_next_valid_timestamp(ts)
    assert result > ts


def test_get_next_valid_timestamp_skips_weekend():
    """
    get_next_valid_timestamp() must skip Saturday and Sunday and return
    a timestamp on Monday or later. Weekend candles do not exist for NYSE.
    """
    from stock_predictor.predict import get_next_valid_timestamp
    import pandas as pd

    saturday = pd.Timestamp("2024-03-16 20:00:00", tz="UTC")
    result = get_next_valid_timestamp(saturday)
    assert result.dayofweek < 5


# build_inference_features


def test_build_inference_features_returns_dataframe():
    """
    build_inference_features() must return a pd.DataFrame.
    Any other return type would cause a TypeError in the /predict endpoint.
    """
    from stock_predictor.predict import build_inference_features

    mock_df = make_feature_df()

    with patch("stock_predictor.predict.create_s3_client"):
        with patch("stock_predictor.predict.get_latest_data_s3", return_value=mock_df):
            result = build_inference_features("AAPL")
    assert isinstance(result, pd.DataFrame)


def test_build_inference_features_not_empty():
    """
    build_inference_features() must return a non-empty DataFrame.
    An empty DataFrame would trigger the 400 error in /predict.
    """
    from stock_predictor.predict import build_inference_features

    mock_df = make_feature_df()

    with patch("stock_predictor.predict.create_s3_client"):
        with patch("stock_predictor.predict.get_latest_data_s3", return_value=mock_df):
            result = build_inference_features("AAPL")
    assert not result.empty
