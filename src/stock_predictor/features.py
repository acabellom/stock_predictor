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


if __name__ == "__main__":
    s3_client = create_s3_client()
    df = get_latest_data_s3(s3_client, "AAPL")
    df = drop_useless_columns(df)
    df.to_csv("./data/aapl_features_test.csv", index=True, quoting=1)
