from prefect import flow, task
from stock_predictor.utils import (
    create_s3_client,
    get_latest_data_s3,
    create_bucket_if_not_exists,
    upload_to_s3,
)
from stock_predictor.features import (
    drop_useless_columns,
    add_lag_data,
    add_change_new_flag,
    add_target,
    fill_missing_news,
    drop_na_values,
)


@task
def load_raw_data(ticker: str):
    """Load the latest raw data for a given ticker from S3."""
    s3_client = create_s3_client()
    return get_latest_data_s3(s3_client, ticker)


@task
def build_features(df):
    """Build the features for the given DataFrame."""
    df = add_change_new_flag(df)
    df = drop_useless_columns(df)
    df = add_lag_data(df)
    df = add_target(df)
    df = fill_missing_news(df)
    df = drop_na_values(df)
    return df


@task
def save_features(df, ticker: str):
    """Save the features DataFrame to S3."""
    s3_client = create_s3_client()
    create_bucket_if_not_exists(s3_client, f"{ticker.lower()}-features")
    upload_to_s3(s3_client, f"{ticker.lower()}-features", f"{ticker}_features.csv", df)


@flow
def feature_flow(ticker: str = "AAPL"):
    df = load_raw_data(ticker)
    df = build_features(df)
    save_features(df, ticker)


if __name__ == "__main__":
    feature_flow("AAPL")
