import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn

from stock_predictor.models.base import BaseModel


class RidgeModel(BaseModel):
    """
    Ridge regression model for stock price prediction.
    Args:
        alpha (float): Regularization strength. Default is 1.0.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Ridge regression model on the given data.
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target variable
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions using the trained Ridge regression model.
        Args:
            X (pd.DataFrame): Features for prediction
        Returns:
            pd.Series: Predicted values"""
        X_scaled = self.scaler.transform(X)
        return pd.Series(self.model.predict(X_scaled), index=X.index)

    def get_params(self) -> dict:
        """Return the parameters for MLflow logging.
        Returns:
            dict: Dictionary of model parameters
        """
        return {"model": "Ridge", "alpha": self.alpha}

    def log_model(self) -> None:
        """Log the model parameters and performance metrics to MLflow."""
        mlflow.sklearn.log_model(self.model, "ridge_model")
