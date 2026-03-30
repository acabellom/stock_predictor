import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from stock_predictor.tune import tune
from stock_predictor.models.linear import RidgeModel
from stock_predictor.train import FEATURE_COLS, TARGET_COL


def make_df(n: int = 200) -> pd.DataFrame:
    """
    Build a minimal DataFrame that mimics the processed MinIO feature data.
    Uses 200 rows so TimeSeriesSplit with 3 folds has meaningful train/val splits.
    All FEATURE_COLS are present plus TARGET_COL.
    """
    rng = np.random.default_rng(42)
    data = {col: rng.standard_normal(n) for col in FEATURE_COLS}
    data[TARGET_COL] = rng.standard_normal(n)
    return pd.DataFrame(
        data,
        index=pd.date_range("2024-01-01 09:30", periods=n, freq="10min"),
    )


def ridge_param_space() -> dict:
    """
    Minimal param space for Ridge — used in tests to avoid the slow LGBM fitting.
    Uses a single hyperparameter (alpha) to keep each trial fast.
    """
    return {
        "alpha": lambda t: t.suggest_float("alpha", 0.01, 10.0),
    }


def test_tune_returns_dict_with_expected_keys():
    """
    tune() must return a dict with 'best_params' and 'best_directional_accuracy'.
    These are the two values consumed by train_flow to run the final training step.
    """
    df = make_df()
    with patch("stock_predictor.tune.mlflow"):
        result = tune(
            model_class=RidgeModel,
            param_space=ridge_param_space(),
            df=df,
            n_trials=3,
            n_splits=2,
        )
    assert "best_params" in result
    assert "best_directional_accuracy" in result


def test_tune_best_params_contains_param_space_keys():
    """
    The keys in best_params must match the keys defined in param_space.
    If a key is missing, the model instantiation in train_flow will fail.
    """
    df = make_df()
    with patch("stock_predictor.tune.mlflow"):
        result = tune(
            model_class=RidgeModel,
            param_space=ridge_param_space(),
            df=df,
            n_trials=3,
            n_splits=2,
        )
    assert "alpha" in result["best_params"]


def test_tune_best_directional_accuracy_between_0_and_1():
    """
    The best directional accuracy found by Optuna must be a valid proportion in [0, 1].
    Values outside this range indicate an error in the objective function.
    """
    df = make_df()
    with patch("stock_predictor.tune.mlflow"):
        result = tune(
            model_class=RidgeModel,
            param_space=ridge_param_space(),
            df=df,
            n_trials=3,
            n_splits=2,
        )
    assert 0.0 <= result["best_directional_accuracy"] <= 1.0


def test_tune_runs_correct_number_of_trials():
    """
    Optuna must run exactly n_trials trials. Fewer trials means the search
    was interrupted; more trials means unexpected behaviour in the study.
    """
    df = make_df()
    trial_count = []

    def counting_param_space():
        def suggest(trial):
            trial_count.append(1)
            return trial.suggest_float("alpha", 0.01, 10.0)

        return {"alpha": suggest}

    with patch("stock_predictor.tune.mlflow"):
        tune(
            model_class=RidgeModel,
            param_space=counting_param_space(),
            df=df,
            n_trials=5,
            n_splits=2,
        )
    assert len(trial_count) == 5


def test_tune_fails_if_feature_col_missing():
    """
    If any column in FEATURE_COLS is missing from the DataFrame,
    tune() must raise a KeyError before running any Optuna trial.
    This prevents wasting time on trials with corrupt data.
    """
    df = make_df().drop(columns=[FEATURE_COLS[0]])
    with patch("stock_predictor.tune.mlflow"):
        with pytest.raises(KeyError):
            tune(
                model_class=RidgeModel,
                param_space=ridge_param_space(),
                df=df,
                n_trials=2,
                n_splits=2,
            )


def test_tune_fails_if_target_col_missing():
    """
    If TARGET_COL is missing from the DataFrame, tune() must raise a KeyError.
    Without a target there is nothing to optimise.
    """
    df = make_df().drop(columns=[TARGET_COL])
    with patch("stock_predictor.tune.mlflow"):
        with pytest.raises(KeyError):
            tune(
                model_class=RidgeModel,
                param_space=ridge_param_space(),
                df=df,
                n_trials=2,
                n_splits=2,
            )


def test_tune_logs_best_params_to_mlflow():
    """
    tune() must call mlflow.log_params() with the best params found by Optuna.
    Without this call, the best configuration would not be visible in the MLflow UI.
    """
    df = make_df()
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)

    with patch("stock_predictor.tune.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value = mock_run
        tune(
            model_class=RidgeModel,
            param_space=ridge_param_space(),
            df=df,
            n_trials=3,
            n_splits=2,
        )
        mock_mlflow.log_params.assert_called()


def test_tune_logs_best_directional_accuracy_to_mlflow():
    """
    tune() must call mlflow.log_metric() with the best directional accuracy.
    This is the primary metric used to compare tuning runs in the MLflow UI.
    """
    df = make_df()
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)

    with patch("stock_predictor.tune.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value = mock_run
        tune(
            model_class=RidgeModel,
            param_space=ridge_param_space(),
            df=df,
            n_trials=3,
            n_splits=2,
        )
        mock_mlflow.log_metric.assert_called()


def test_tune_with_single_trial_returns_valid_result():
    """
    tune() must work correctly even with a single trial (n_trials=1).
    This is the minimum meaningful configuration and must not raise any errors.
    """
    df = make_df()
    with patch("stock_predictor.tune.mlflow"):
        result = tune(
            model_class=RidgeModel,
            param_space=ridge_param_space(),
            df=df,
            n_trials=1,
            n_splits=2,
        )
    assert "best_params" in result
    assert "best_directional_accuracy" in result
