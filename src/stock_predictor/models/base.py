from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions on the given data."""
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return the parameters of the model."""
        pass

    @abstractmethod
    def log_model(self) -> None:
        """Log the model parameters and performance metrics."""
        pass
