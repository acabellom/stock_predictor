import pandas as pd
import mlflow.lightgbm
from lightgbm import LGBMRegressor

from stock_predictor.models.base import BaseModel


class LGBMModel(BaseModel):
    """
    LightGBM regression model for stock price prediction.

    Unlike Ridge, LightGBM can capture non-linear relationships
    between features and the target price. No scaling needed
    as tree-based models are invariant to feature magnitude.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the LightGBM model on the given data.

        Args:
            X: Feature matrix.
            y: Target series (next candle close price).
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions for the given feature matrix.

        Args:
            X: Feature matrix.
        Returns:
            Series of predicted close prices.
        """
        return pd.Series(self.model.predict(X), index=X.index)

    def get_params(self) -> dict:
        """
        Return model parameters for MLflow logging.

        Returns:
            Dictionary with model name and all hyperparameters.
        """
        return {
            "model": "lgbm",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
        }

    def log_model(self) -> None:
        """
        Log the trained LightGBM model as an MLflow artifact.
        """
        mlflow.lightgbm.log_model(self.model, name="lgbm_model")
