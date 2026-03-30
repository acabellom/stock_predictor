import numpy as np
import pandas as pd
import pytest
from stock_predictor.models.linear import RidgeModel
from stock_predictor.models.lgbm import LGBMModel


def make_xy(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a minimal feature matrix and target series for testing.
    Uses random data — we are testing the interface, not model quality.
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "c": rng.standard_normal(n),
            "v": rng.standard_normal(n),
            "vw": rng.standard_normal(n),
            "n": rng.standard_normal(n),
            "c1": rng.standard_normal(n),
            "c2": rng.standard_normal(n),
            "c3": rng.standard_normal(n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="10min"),
    )
    y = pd.Series(rng.standard_normal(n), index=X.index)
    return X, y


# RidgeModel


def test_ridge_fit_predict_returns_series():
    """
    Check that the predict method returns a pandas Series after fitting.
    This ensures that the model is producing outputs in the expected format."""
    X, y = make_xy()
    model = RidgeModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, pd.Series)


def test_ridge_predict_same_length_as_input():
    """Check that the predict method returns a Series of the same length as the input features."""
    X, y = make_xy()
    model = RidgeModel()
    model.fit(X, y)
    assert len(model.predict(X)) == len(X)


def test_ridge_predict_preserves_index():
    """
    Check that the predict method returns a Series with the same index as the input features.
    This is important for downstream processing and for ensuring that predictions can be aligned with the original data
    """
    X, y = make_xy()
    model = RidgeModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert list(preds.index) == list(X.index)


def test_ridge_get_params_contains_model_name():
    """
    Check that the get_params method returns the model name correctly.
    This is important for MLflow logging and for ensuring that the model's configuration is transparent.
    """
    model = RidgeModel(alpha=2.0)
    params = model.get_params()
    assert params["model"] == "Ridge"


def test_ridge_get_params_contains_alpha():
    """
    Check that the get_params method returns the alpha hyperparameter correctly.
    This is important for MLflow logging and for ensuring that the model's configuration is transparent.
    """
    model = RidgeModel(alpha=2.0)
    params = model.get_params()
    assert params["alpha"] == 2.0


def test_ridge_predict_no_nulls():
    """
    Check that predictions do not contain null values after fitting.
    This is a basic sanity check to ensure the model is producing valid outputs.
    """
    X, y = make_xy()
    model = RidgeModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.isnull().sum() == 0


def test_ridge_different_alpha_gives_different_predictions():
    """
    Check that changing the alpha hyperparameter actually changes predictions.
    This is a sanity check to ensure that the model is using the alpha parameter.
    """
    X, y = make_xy()
    model1 = RidgeModel(alpha=0.01)
    model2 = RidgeModel(alpha=100.0)
    model1.fit(X, y)
    model2.fit(X, y)
    assert not model1.predict(X).equals(model2.predict(X))


# LGBMModel


def test_lgbm_fit_predict_returns_series():
    """
    Check that the predict method returns a pandas Series after fitting.
    This ensures that the model is producing outputs in the expected format."""
    X, y = make_xy()
    model = LGBMModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, pd.Series)


def test_lgbm_predict_same_length_as_input():
    """
    Check that the predict method returns a Series of the same length as the input features.
    This ensures that the model is producing outputs in the expected format."""
    X, y = make_xy()
    model = LGBMModel()
    model.fit(X, y)
    assert len(model.predict(X)) == len(X)


def test_lgbm_predict_preserves_index():
    """
    Check that the predict method returns a Series with the same index as the input features.
    This is important for downstream processing and for ensuring that predictions can be aligned with the original data
    """
    X, y = make_xy()
    model = LGBMModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert list(preds.index) == list(X.index)


def test_lgbm_get_params_contains_model_name():
    """
    Check that the get_params method returns the model name correctly.
    This is important for MLflow logging and for ensuring that the model's configuration is transparent."""
    model = LGBMModel(n_estimators=100)
    params = model.get_params()
    assert params["model"] == "lgbm"


def test_lgbm_get_params_contains_n_estimators():
    """
    Check that the get_params method returns the n_estimators hyperparameter correctly.
    This is important for MLflow logging and for ensuring that the model's configuration is transparent."""
    model = LGBMModel(n_estimators=100)
    params = model.get_params()
    assert params["n_estimators"] == 100


def test_lgbm_predict_no_nulls():
    """
    Check that predictions do not contain null values after fitting.
    This is a basic sanity check to ensure the model is producing valid outputs.
    """
    X, y = make_xy()
    model = LGBMModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.isnull().sum() == 0


def test_lgbm_predict_before_fit_raises():
    """
    Check that calling predict before fit raises an exception.
    This ensures that the model is properly enforcing the fit-predict workflow.
    """
    X, _ = make_xy()
    model = LGBMModel()
    with pytest.raises(Exception):
        model.predict(X)


# Interface compliance — both models must implement BaseModel


def test_ridge_implements_base_model_interface():
    model = RidgeModel()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "get_params")
    assert hasattr(model, "log_model")


def test_lgbm_implements_base_model_interface():
    model = LGBMModel()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "get_params")
    assert hasattr(model, "log_model")
