import pandas as pd
import mlflow
from stock_predictor.models.base import BaseModel
from sklearn.model_selection import TimeSeriesSplit
import os


FEATURE_COLS = [
    "v",
    "vw",
    "c",
    "n",
    "sentiment",
    "c1",
    "c2",
    "c3",
    "c5",
    "c10",
    "change_new_flag",
    "volatility",
    "vwap_dist",
    "rel_volume",
    "hour",
    "minute",
    "day_of_week",
]
TARGET_COL = "target"


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Compute regression metrics for model evaluation.

    Args:
        y_true: Ground truth target values.
        y_pred: Model predictions.
    Returns:
        Dictionary with MAE, RMSE and directional accuracy.
    """
    mae = (y_true - y_pred).abs().mean()
    rmse = ((y_true - y_pred) ** 2).mean() ** 0.5
    directional_acc = ((y_true.diff() > 0) == (y_pred.diff() > 0)).mean()

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "directional_accuracy": round(directional_acc, 4),
    }


def train_model(
    model: BaseModel,
    df: pd.DataFrame,
    n_splits: int = 5,
    experiment_name: str = "stock-prediction",
) -> dict:
    """
    Train the given model using time series cross-validation and log the results to MLflow.

    Args:
        model (BaseModel): The model to be trained
        df (pd.DataFrame): The input DataFrame containing features and target variable
        n_splits (int): The number of splits for time series cross-validation
        experiment_name (str): The name of the MLflow experiment for logging
    Returns:
        dict: A dictionary containing the average performance metrics across all splits
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment_name)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_per_fold = []

    with mlflow.start_run():
        mlflow.log_params(model.get_params())
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("training_data_size", len(df))

        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            fold_metrics = evaluate(y_test, predictions)
            metrics_per_fold.append(fold_metrics)

            mlflow.log_metrics(
                {
                    f"{metric}_fold_{fold}": value
                    for metric, value in fold_metrics.items()
                }
            )

        avg_metrics = {
            k: round(sum(m[k] for m in metrics_per_fold) / n_splits, 6)
            for k in metrics_per_fold[0]
        }
        mlflow.log_metrics(
            {f"avg_{metric}": value for metric, value in avg_metrics.items()}
        )
        model.log_model()

    return avg_metrics


if __name__ == "__main__":
    from stock_predictor.utils import get_latest_data_s3, create_s3_client
    from stock_predictor.models.lgbm import LGBMModel

    s3_client = create_s3_client()
    df = get_latest_data_s3(s3_client, "aapl-features")

    X = df[
        [
            "v",
            "vw",
            "c",
            "n",
            "sentiment",
            "c1",
            "c2",
            "c3",
            "c5",
            "c10",
            "change_new_flag",
            "volatility",
            "vwap_dist",
            "rel_volume",
            "hour",
            "minute",
            "day_of_week",
        ]
    ]

    tscv = TimeSeriesSplit(n_splits=3)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold}: train={len(train_idx)} val={len(val_idx)}")
        print(f"  train: {X.index[train_idx[0]]} → {X.index[train_idx[-1]]}")
        print(f"  val:   {X.index[val_idx[0]]} → {X.index[val_idx[-1]]}")

    model = LGBMModel(n_estimators=100, max_depth=3, num_leaves=15, learning_rate=0.01)
    metrics = train_model(
        model=model,
        df=df,
        n_splits=3,
    )

    print(metrics)
