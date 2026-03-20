import pandas as pd
import numpy as np
from stock_predictor.features import (
    drop_useless_columns,
    add_lag_data,
)


def make_df(n: int = 15) -> pd.DataFrame:
    """
    Build a minimal DataFrame that mimics the raw MinIO data structure.
    Uses n rows to ensure all lags (up to 10) can be computed without
    all rows becoming NaN.
    """
    return pd.DataFrame(
        {
            "c": [round(100 + i * 0.1, 2) for i in range(n)],
            "v": [1000 + i * 10 for i in range(n)],
            "vw": [round(100 + i * 0.1, 4) for i in range(n)],
            "n": [50 + i for i in range(n)],
            "h": [round(100 + i * 0.2, 2) for i in range(n)],
            "l": [round(100 + i * 0.05, 2) for i in range(n)],
            "o": [round(100 + i * 0.1, 2) for i in range(n)],
            "average_price": [round(100 + i * 0.1, 2) for i in range(n)],
            "published_utc": ["2024-01-01"] * n,
            "headline": (["Headline A"] * 5 + ["Headline B"] * 5 + ["Headline A"] * 5),
            "positive": [0.5] * n,
            "negative": [0.1] * n,
            "neutral": [0.4] * n,
            "sentiment": [0.4] * 5 + [np.nan] * 5 + [0.2] * 5,
        },
        index=pd.date_range("2024-01-01 09:00", periods=n, freq="10min"),
    )


def test_drop_useless_columns_drops_target_columns():
    df = make_df()
    result = drop_useless_columns(df)
    dropped = [
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
    for col in dropped:
        assert col not in result.columns


def test_drop_useless_columns_keeps_useful_columns():
    df = make_df()
    result = drop_useless_columns(df)
    for col in ["c", "v", "vw", "n", "sentiment"]:
        assert col in result.columns


def test_drop_useless_columns_does_not_raise_if_column_missing():
    df = make_df().drop(columns=["h", "l"])
    result = drop_useless_columns(df)
    assert "o" not in result.columns


def test_drop_useless_columns_returns_dataframe():
    df = make_df()
    assert isinstance(drop_useless_columns(df), pd.DataFrame)


def test_add_lag_data_adds_default_lag_columns():
    df = make_df()
    result = add_lag_data(df)
    for lag in [1, 2, 3, 5, 10]:
        assert f"c{lag}" in result.columns


def test_add_lag_data_lag1_is_previous_close():
    df = make_df()
    result = add_lag_data(df)
    assert result["c1"].iloc[1] == df["c"].iloc[0]


def test_add_lag_data_lag1_first_row_is_nan():
    df = make_df()
    result = add_lag_data(df)
    assert pd.isna(result["c1"].iloc[0])


def test_add_lag_data_custom_lag_list():
    df = make_df()
    result = add_lag_data(df, lag_list=[2, 4])
    assert "c2" in result.columns
    assert "c4" in result.columns
    assert "c1" not in result.columns


def test_add_lag_data_does_not_modify_original_close():
    df = make_df()
    original_close = df["c"].copy()
    add_lag_data(df)
    pd.testing.assert_series_equal(df["c"], original_close)
