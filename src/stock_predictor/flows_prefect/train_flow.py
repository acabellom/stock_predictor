from prefect import flow, task
from stock_predictor.utils import create_s3_client, get_latest_data_s3
from stock_predictor.train import train_model
from stock_predictor.models.lgbm import LGBMModel
from stock_predictor.tune import tune
import pandas as pd


@task
def load_processed_data_s3(ticker: str):
    """Load the latest processed data for a given ticker from S3."""
    s3_client = create_s3_client()
    return get_latest_data_s3(s3_client, f"{ticker.lower()}-features")


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
def train_model_task(df: pd.DataFrame, model_params: dict):
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


@flow
def train_flow(tickers: list = ["AAPL", "TSLA"], tune_first: bool = True):
    dfs = []
    for ticker in tickers:
        df = load_processed_data_s3(ticker)
        df["ticker"] = ticker
        dfs.append(df)
    df_all = pd.concat(dfs).sort_index()
    if tune_first:
        result = tune_model(df_all)
        params = result["best_params"]
    else:
        params = {
            "n_estimators": 559,
            "learning_rate": 0.0017,
            "max_depth": 3,
            "num_leaves": 34,
            "min_child_samples": 43,
            "subsample": 0.983,
            "colsample_bytree": 0.826,
        }
    train_model_task(df_all, params)


if __name__ == "__main__":
    train_flow(["AAPL", "TSLA"], tune_first=True)
