import pandas as pd
from stock_predictor.utils import get_latest_data_s3, create_s3_client


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not useful for the model.

    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        pd.DataFrame: DataFrame with useless columns dropped
    """
    columns_to_drop = [
        "h",
        "l",
        "o",
        "average_price",
        "published_utc",
        "headline",
        "positive",
        "negative",
        "neutral",
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    return df


def add_lag_data(df: pd.DataFrame, lag_list: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """
    Add lagged features to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        lag_list (list): List of lagged periods to add
    Returns:
        pd.DataFrame: DataFrame with lagged features added
    """
    returns = df["c"].pct_change()
    for lag in lag_list:
        df[f"c{lag}"] = returns.shift(lag)
    return df


def add_change_new_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a flag to indicate if the latest new has changed compared to the previous period.

    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        pd.DataFrame: DataFrame with the change flag added
    """
    df["change_new_flag"] = (df["headline"] != df["headline"].shift(1)).astype(int)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a target variable to the DataFrame indicating the next period close price.

    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        pd.DataFrame: DataFrame with the target variable added
    """
    df["target"] = df["c"].shift(-1) / df["c"] - 1
    return df


def fill_missing_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing news headlines with the last available headline.

    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        pd.DataFrame: DataFrame with missing news filled
    """
    df["sentiment"] = df["sentiment"].fillna(0.0)
    return df


def drop_na_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NA values from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        pd.DataFrame: DataFrame with NA values dropped
    """
    df = df.dropna()
    return df


def add_new_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new columns to the DataFrame based on the existing features.

    This function can be used to create interaction terms, polynomial features, or any other derived features that may help the model capture complex relationships between the features and the target price. For example, we could add a feature that represents the ratio of volume to price, or a feature that captures the volatility of the stock over a certain period. The specific new columns to add would depend on domain knowledge and experimentation with the data.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with new columns added
    """
    df["volatility"] = df["c"].pct_change().rolling(10).std()

    df["vwap_dist"] = (df["c"] - df["vw"]) / df["vw"]

    df["rel_volume"] = df["v"] / df["v"].rolling(20).mean()
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["day_of_week"] = df.index.dayofweek
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI) to the DataFrame.
    RSI measures momentum — values above 70 indicate overbought,
    below 30 indicate oversold conditions.
    """
    delta = df["c"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Average True Range (ATR) to the DataFrame.
    ATR measures volatility — high ATR means large price swings expected.
    Since we dropped h and l, we approximate true range using close only.
    """
    tr = df["c"].diff().abs()
    df["atr"] = tr.rolling(period).mean()
    return df


if __name__ == "__main__":
    s3_client = create_s3_client()
    df = get_latest_data_s3(s3_client, "AAPL")
    df = add_change_new_flag(df)
    df = drop_useless_columns(df)
    df = add_lag_data(df)
    df = add_target(df)
    df = add_new_columns(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = fill_missing_news(df)
    df = drop_na_values(df)
    df.to_csv("./data/aapl_features_test.csv", index=True, quoting=1)
