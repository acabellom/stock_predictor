import pandas as pd
import mlflow
import optuna
import os
from sklearn.model_selection import TimeSeriesSplit

from stock_predictor.models.base import BaseModel
from stock_predictor.train import FEATURE_COLS, TARGET_COL, evaluate


optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune(
    model_class: type[BaseModel],
    param_space: dict,
    df: pd.DataFrame,
    n_trials: int = 50,
    n_splits: int = 3,
    experiment_name: str = "stock-prediction-tuning",
) -> dict:
    """
    Tune hyperparameters for any BaseModel using Optuna and log results to MLflow.
    Each trial is logged as a nested MLflow run inside a parent tuning run.

    Args:
        model_class: Class of the model to tune (e.g. LGBMModel).
        param_space: Dictionary mapping param names to Optuna suggest functions.
                     Example: {"alpha": lambda t: t.suggest_float("alpha", 0.01, 10.0)}
        df: Full feature DataFrame loaded from MinIO.
        n_trials: Number of Optuna trials.
        n_splits: Number of TimeSeriesSplit folds per trial.
        experiment_name: MLflow experiment name.
    Returns:
        Dictionary with best params and best directional accuracy found.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment_name)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    with mlflow.start_run(run_name=f"tune_{model_class.__name__}"):

        def objective(trial: optuna.Trial) -> float:
            params = {k: v(trial) for k, v in param_space.items()}
            model = model_class(**params)

            fold_metrics = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                fold_metrics.append(evaluate(y_val, predictions))

            avg_dir_acc = (
                sum(m["directional_accuracy"] for m in fold_metrics) / n_splits
            )

            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("directional_accuracy", avg_dir_acc)

            return avg_dir_acc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_score = study.best_value

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_directional_accuracy", best_score)

    return {"best_params": best_params, "best_directional_accuracy": best_score}


if __name__ == "__main__":
    from stock_predictor.utils import get_latest_data_s3, create_s3_client
    from stock_predictor.models.lgbm import LGBMModel

    s3_client = create_s3_client()
    df = get_latest_data_s3(s3_client, "aapl-features")

    lgbm_param_space = {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 1000),
        "learning_rate": lambda t: t.suggest_float(
            "learning_rate", 1e-3, 0.3, log=True
        ),
        "max_depth": lambda t: t.suggest_int("max_depth", 3, 10),
        "num_leaves": lambda t: t.suggest_int("num_leaves", 15, 128),
        "min_child_samples": lambda t: t.suggest_int("min_child_samples", 10, 100),
        "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    result = tune(
        model_class=LGBMModel,
        param_space=lgbm_param_space,
        df=df,
        n_trials=50,
    )

    print(f"Best directional accuracy: {result['best_directional_accuracy']:.4f}")
    print(f"Best params: {result['best_params']}")
