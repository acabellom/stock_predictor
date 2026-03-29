import pandas as pd
import numpy as np
from stock_predictor.features import (
    drop_useless_columns,
    add_lag_data,
    add_change_new_flag,
    add_target,
    fill_missing_news,
    drop_na_values,
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
    assert result["c1"].iloc[2] == df["c"].pct_change().iloc[1]


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


def test_add_change_new_flag_adds_column():
    df = make_df()
    result = add_change_new_flag(df)
    assert "change_new_flag" in result.columns


def test_add_change_new_flag_is_1_when_headline_changes():
    df = make_df()
    result = add_change_new_flag(df)
    assert result["change_new_flag"].iloc[5] == 1


def test_add_change_new_flag_is_0_when_headline_stays_same():
    df = make_df()
    result = add_change_new_flag(df)
    assert result["change_new_flag"].iloc[2] == 0


def test_add_change_new_flag_is_integer():
    df = make_df()
    result = add_change_new_flag(df)
    assert result["change_new_flag"].dtype in [int, np.int32, np.int64]


def test_add_change_new_flag_first_row_is_1():
    df = make_df()
    result = add_change_new_flag(df)
    assert result["change_new_flag"].iloc[0] == 1


def test_add_target_adds_column():
    df = make_df()
    result = add_target(df)
    assert "target" in result.columns


def test_add_target_is_next_close():
    df = make_df()
    result = add_target(df)
    expected = df["c"].iloc[1] / df["c"].iloc[0] - 1
    assert result["target"].iloc[0] == expected


def test_add_target_last_row_is_nan():
    df = make_df()
    result = add_target(df)
    assert pd.isna(result["target"].iloc[-1])


def test_add_target_is_float():
    df = make_df()
    result = add_target(df)
    assert result["target"].dtype == float


def test_fill_missing_news_no_nulls_after_fill():
    df = make_df()
    result = fill_missing_news(df)
    assert result["sentiment"].isnull().sum() == 0


def test_fill_missing_news_nulls_filled_with_zero():
    df = make_df()
    result = fill_missing_news(df)
    assert (result["sentiment"].iloc[5:10] == 0.0).all()


def test_fill_missing_news_non_null_values_unchanged():
    df = make_df()
    result = fill_missing_news(df)
    assert result["sentiment"].iloc[0] == 0.4


def test_drop_na_values_no_nulls_after_drop():
    df = make_df()
    df = add_lag_data(df)
    df = add_target(df)
    result = drop_na_values(df)
    assert result.isnull().sum().sum() == 0


def test_drop_na_values_removes_first_rows_from_lags():
    df = make_df()
    df = add_lag_data(df, lag_list=[1, 2, 3, 5, 10])
    result = drop_na_values(df)
    assert result.index[0] == df.index[11]


def test_drop_na_values_removes_last_row_from_target():
    df = make_df()
    df = add_target(df)
    result = drop_na_values(df)
    assert result.index[-1] != df.index[-1]


def test_drop_na_values_returns_dataframe():
    df = make_df()
    assert isinstance(drop_na_values(df), pd.DataFrame)
