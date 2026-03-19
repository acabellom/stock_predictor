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
    for lag in lag_list:
        df[f"c{lag}"] = df["c"].shift(lag)
    return df


if __name__ == "__main__":
    s3_client = create_s3_client()
    df = get_latest_data_s3(s3_client, "AAPL")
    df = drop_useless_columns(df)
    df = add_lag_data(df)
    df.to_csv("./data/aapl_features_test.csv", index=True, quoting=1)
