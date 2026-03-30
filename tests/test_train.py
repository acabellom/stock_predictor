import numpy as np
import pandas as pd
from stock_predictor.train import evaluate, FEATURE_COLS, TARGET_COL


def make_df(n: int = 200) -> pd.DataFrame:
    """
    Build a minimal DataFrame that mimics the processed MinIO feature data.
    Uses enough rows (200) so that TimeSeriesSplit with 3 folds has meaningful splits.
    All FEATURE_COLS are present plus the TARGET_COL.
    """
    rng = np.random.default_rng(42)
    data = {col: rng.standard_normal(n) for col in FEATURE_COLS}
    data[TARGET_COL] = rng.standard_normal(n)
    return pd.DataFrame(
        data,
        index=pd.date_range("2024-01-01 09:30", periods=n, freq="10min"),
    )


# evaluate


def test_evaluate_returns_dict_with_expected_keys():
    """
    evaluate() must return a dictionary containing mae, rmse and directional_accuracy.
    These are the three metrics logged to MLflow on every training run.
    """
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = pd.Series([1.1, 1.9, 3.1, 3.9, 5.1])
    result = evaluate(y_true, y_pred)
    assert "mae" in result
    assert "rmse" in result
    assert "directional_accuracy" in result


def test_evaluate_mae_is_non_negative():
    """
    MAE must always be >= 0. A negative MAE would indicate a sign error
    in the absolute difference calculation.
    """
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([2.0, 1.0, 4.0])
    result = evaluate(y_true, y_pred)
    assert result["mae"] >= 0


def test_evaluate_rmse_is_non_negative():
    """
    RMSE must always be >= 0 since it is the square root of a mean squared value.
    """
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([2.0, 1.0, 4.0])
    result = evaluate(y_true, y_pred)
    assert result["rmse"] >= 0


def test_evaluate_perfect_predictions_give_zero_mae():
    """
    When predictions exactly match ground truth, MAE and RMSE must be 0.
    This validates the basic correctness of the error formula.
    """
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = evaluate(y, y)
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0


def test_evaluate_directional_accuracy_between_0_and_1():
    """
    Directional accuracy is a proportion and must always be in [0, 1].
    Values outside this range would indicate a logic error in the calculation.
    """
    y_true = pd.Series([1.0, 2.0, 1.5, 3.0, 2.5])
    y_pred = pd.Series([1.1, 1.9, 1.6, 2.8, 2.6])
    result = evaluate(y_true, y_pred)
    assert 0.0 <= result["directional_accuracy"] <= 1.0


def test_evaluate_rmse_geq_mae():
    """
    RMSE is always >= MAE because squaring errors penalises larger mistakes more.
    If RMSE < MAE the error computation is wrong.
    """
    y_true = pd.Series([1.0, 2.0, 3.0, 10.0])
    y_pred = pd.Series([1.5, 2.5, 3.5, 5.0])
    result = evaluate(y_true, y_pred)
    assert result["rmse"] >= result["mae"]


def test_evaluate_values_are_rounded():
    """
    All returned metrics must be Python floats rounded to a fixed number
    of decimal places, not raw numpy floats with many trailing digits.
    This ensures clean MLflow logging.
    """
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.1, 2.2, 3.3])
    result = evaluate(y_true, y_pred)
    for v in result.values():
        assert isinstance(v, float)
