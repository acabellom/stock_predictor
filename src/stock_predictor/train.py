import pandas as pd
import mlflow
from stock_predictor.models.base import BaseModel
from sklearn.model_selection import TimeSeriesSplit


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
]
TARGET_COL = "target"


def evaluate():
    pass


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

            fold_metrics = evaluate(predictions, y_test)
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
