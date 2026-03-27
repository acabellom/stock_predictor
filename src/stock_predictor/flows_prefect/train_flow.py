from prefect import task
from stock_predictor.utils import create_s3_client, load_processed_data
from stock_predictor.train import train_model
from stock_predictor.models.lgbm import LGBMModel
from stock_predictor.tune import tune
import pandas as pd


@task
def load_processed_data_s3(ticker: str):
    """Load the latest processed data for a given ticker from S3."""
    s3_client = create_s3_client()
    return load_processed_data(s3_client, f"{ticker}-features")


@task
def tune_model(data: pd.DataFrame):
    """Tune the model hyperparameters using Optuna."""
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
        df=data,
        n_trials=50,
    )
    return result


@task
def train_flow(df: pd.DataFrame, model_params: dict):
    model = LGBMModel(
        n_estimators=model_params["n_estimators"],
        learning_rate=model_params["learning_rate"],
        max_depth=model_params["max_depth"],
        num_leaves=model_params["num_leaves"],
        min_child_samples=model_params["min_child_samples"],
        subsample=model_params["subsample"],
        colsample_bytree=model_params["colsample_bytree"],
    )
    train_model(
        model=model,
        df=df,
        n_splits=3,
    )
