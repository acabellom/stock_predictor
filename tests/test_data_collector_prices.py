import pytest
from unittest.mock import patch, Mock
from stock_predictor.data_collector_prices import (
    fetch_stock_data_month,
    fetch_last_2_years,
    fetch_last_month_,
    process_stock_data,
    merge_raw_data,
)
import requests
from datetime import datetime
import pandas as pd


@patch("stock_predictor.data_collector_prices.requests.get")
def test_fetch_stock_data_month_success(mock_get):
    """
    Test fetching stock data for a specific month successfully.

    :param mock_get: Mocked requests.get method.
    """

    fake_response = {
        "ticker": "AAPL",
        "resultsCount": 2,
        "results": [
            {"t": 1704067200000, "c": 190.5},
            {"t": 1704070800000, "c": 191.0},
        ],
    }

    mock_get.return_value = Mock(
        status_code=200,
        json=Mock(return_value=fake_response),
        raise_for_status=Mock(),
    )

    result = fetch_stock_data_month("AAPL", 2024, 1)

    assert result == fake_response
    mock_get.assert_called_once()

    called_url = mock_get.call_args[0][0]
    assert "AAPL" in called_url
    assert "2024-01-01" in called_url
    assert "2024-01-31" in called_url


@patch("stock_predictor.data_collector_prices.requests.get")
def test_fetch_stock_data_month_http_error(mock_get):
    """
    Test fetching stock data for a specific month with an HTTP error.

    :param mock_get: Mocked requests.get method.
    """
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("API error")
    mock_get.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        fetch_stock_data_month("AAPL", 2024, 1)


@patch("stock_predictor.data_collector_prices.requests.get")
def test_fetch_stock_data_month_february(mock_get):
    """
    Test fetching stock data for February, ensuring correct date handling.
    :param mock_get: Mocked requests.get method.
    """
    mock_get.return_value = Mock(
        status_code=200,
        json=Mock(return_value={}),
        raise_for_status=Mock(),
    )

    fetch_stock_data_month("AAPL", 2023, 2)

    called_url = mock_get.call_args[0][0]
    assert "2023-02-01" in called_url
    assert "2023-02-28" in called_url


@patch("stock_predictor.data_collector_prices.requests.get")
def test_fetch_stock_data_month_leap_year(mock_get):
    """
    Test fetching stock data for February in a leap year.
    :param mock_get: Mocked requests.get method.
    """
    mock_get.return_value = Mock(
        status_code=200,
        json=Mock(return_value={}),
        raise_for_status=Mock(),
    )

    fetch_stock_data_month("AAPL", 2024, 2)

    called_url = mock_get.call_args[0][0]
    assert "2024-02-29" in called_url


@patch("stock_predictor.data_collector_prices.sleep")
@patch("stock_predictor.data_collector_prices.fetch_last_month_")
@patch("stock_predictor.data_collector_prices.fetch_stock_data_month")
@patch("stock_predictor.data_collector_prices.datetime")
def test_fetch_last_2_years_ok(
    mock_datetime, mock_fetch_month, mock_fetch_last_month, mock_sleep
):
    """
    Test fetching historical stock data for the last 2 years successfully.
    :param mock_datetime: Mocked datetime module.
    :param mock_fetch_month: Mocked fetch_stock_data_month function.
    :param mock_fetch_last_month: Mocked fetch_last_month_ function.
    :param mock_sleep: Mocked sleep function.
    """

    mock_datetime.today.return_value = datetime(2024, 1, 15)
    mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

    mock_fetch_month.return_value = {"results": [{"price": 100}, {"price": 101}]}

    mock_fetch_last_month.return_value = {"results": [{"price": 200}]}

    data = fetch_last_2_years("AAPL")

    assert isinstance(data, list)
    assert len(data) > 0
    assert {"price": 100} in data
    assert {"price": 200} in data

    assert mock_fetch_month.called
    assert mock_fetch_last_month.called

    mock_sleep.assert_called()


@patch("stock_predictor.data_collector_prices.sleep")
@patch("stock_predictor.data_collector_prices.fetch_last_month_")
@patch("stock_predictor.data_collector_prices.fetch_stock_data_month")
def test_fetch_last_2_years_handles_exception(
    mock_fetch_month, mock_fetch_last_month, mock_sleep
):
    """
    Test that fetch_last_2_years handles exceptions during monthly data fetching.
     :param mock_fetch_month: Mocked fetch_stock_data_month function.
     :param mock_fetch_last_month: Mocked fetch_last_month_ function.
     :param mock_sleep: Mocked sleep function.
    """
    mock_fetch_month.side_effect = Exception("API down")
    mock_fetch_last_month.return_value = {"results": []}

    data = fetch_last_2_years("AAPL")

    assert data == []


@patch("stock_predictor.data_collector_prices.requests.get")
def test_fetch_last_month_success(mock_get):
    """
    Test fetching stock data for the last month successfully.

    :param mock_get: Mocked requests.get method.
    """
    fake_response = {
        "ticker": "AAPL",
        "resultsCount": 2,
        "results": [
            {"t": 1704067200000, "c": 190.5},
            {"t": 1704070800000, "c": 191.0},
        ],
    }

    mock_get.return_value = Mock(
        status_code=200,
        json=Mock(return_value=fake_response),
        raise_for_status=Mock(),
    )

    result = fetch_last_month_("AAPL")

    assert result == fake_response
    mock_get.assert_called_once()

    called_url = mock_get.call_args[0][0]
    assert "AAPL" in called_url
    today = datetime.today()
    start_month = today.replace(day=1).strftime("%Y-%m-%d")
    end_day = today.strftime("%Y-%m-%d")
    assert start_month in called_url
    assert end_day in called_url


@patch("stock_predictor.data_collector_prices.requests.get")
def test_fetch_last_month_special_case(mock_get):
    """
    Test fetching stock data for the last month when the first day of the month is a weekend and today is 2nd or 1st.

    :param mock_get: Mocked requests.get method.
    """

    result1 = fetch_last_month_("AAPL", today=datetime(2026, 3, 1))
    result2 = fetch_last_month_("AAPL", today=datetime(2025, 11, 2))
    assert result1 == -1
    assert result2 == -1


@patch("stock_predictor.data_collector_prices.requests.get")
def test_fetch_last_month_other_ticker(mock_get):
    """
    Test that the URL is correctly built for a different ticker symbol.

    :param mock_get: Mocked requests.get method.
    """
    mock_get.return_value = Mock(
        status_code=200,
        json=Mock(return_value={"ok": True}),
        raise_for_status=Mock(),
    )

    fetch_last_month_("TSLA", today=datetime(2024, 1, 15))

    called_url = mock_get.call_args[0][0]
    assert "/TSLA/" in called_url


@patch("stock_predictor.data_collector_prices.requests.get")
def test_fetch_last_month_returns_json(mock_get):
    """
    Test that the function returns the JSON payload from the API.

    :param mock_get: Mocked requests.get method.
    """
    data = {"results": [{"price": 100}, {"price": 101}]}

    mock_get.return_value = Mock(
        status_code=200,
        json=Mock(return_value=data),
        raise_for_status=Mock(),
    )

    result = fetch_last_month_("AAPL", today=datetime(2024, 1, 15))

    assert result == data


def test_process_stock_data_basic():
    """
    Test processing of basic stock data into a DataFrame.
    """
    raw_data = [
        {"t": 1704067200000, "h": 192.0, "l": 189.0, "c": 190.5},
        {"t": 1704070800000, "h": 193.0, "l": 191.0, "c": 192.0},
    ]

    df = process_stock_data(raw_data)

    assert isinstance(df, pd.DataFrame)
    assert pd.api.types.is_datetime64_any_dtype(df.index)

    expected_avg = [(192.0 + 189.0) / 2, (193.0 + 191.0) / 2]
    assert df["average_price"].tolist() == expected_avg

    for col in ["h", "l", "c", "average_price"]:
        assert col in df.columns


def test_process_stock_data_empty():
    """
    Test processing when input data is empty.
    """
    raw_data = []
    df = (
        process_stock_data(raw_data)
        if raw_data
        else pd.DataFrame(
            columns=["t", "h", "l", "c", "average_price"],
            index=pd.DatetimeIndex([], name="t"),
        )
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_process_stock_data_single_row():
    """
    Test processing when input data has a single record.
    """
    raw_data = [
        {"t": 1704067200000, "h": 200.0, "l": 198.0, "c": 199.0},
    ]

    df = process_stock_data(raw_data)

    assert len(df) == 1
    assert df["average_price"].iloc[0] == (200.0 + 198.0) / 2


@patch("stock_predictor.data_collector_prices.pd.read_csv")
def test_merge_raw_data_multiple_raw_files(mock_read_csv):
    """
    Test merging multiple raw_ files from S3 correctly.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "raw_old1.csv"},
                    {"Key": "raw_old2.csv"},
                ]
            }

        def get_object(self, Bucket, Key):
            return {"Body": "fake_body"}

    new_data = pd.DataFrame(
        {"value": [1]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    new_data.index.name = "t"

    old_df1 = pd.DataFrame(
        {"value": [2]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    old_df1.index.name = "t"

    old_df2 = pd.DataFrame(
        {"value": [3]},
        index=pd.to_datetime(["2024-01-03"]),
    )
    old_df2.index.name = "t"

    mock_read_csv.side_effect = [old_df1, old_df2]

    merged_df, last_key = merge_raw_data("AAPL", MockS3(), new_data)

    assert len(merged_df) == 3
    assert merged_df.index.is_monotonic_increasing
    assert last_key == "raw_old2.csv"


@patch("stock_predictor.data_collector_prices.pd.read_csv")
def test_merge_raw_data_ignores_non_raw_files(mock_read_csv):
    """
    Test that non raw_ files are ignored.
    :param mock_read_csv: Mocked pandas.read_csv function.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "processed.csv"},
                    {"Key": "raw_valid.csv"},
                ]
            }

        def get_object(self, Bucket, Key):
            return {"Body": "fake_body"}

    new_data = pd.DataFrame(
        {"value": [1]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    new_data.index.name = "t"

    old_df = pd.DataFrame(
        {"value": [2]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    old_df.index.name = "t"

    mock_read_csv.return_value = old_df

    merged_df, last_key = merge_raw_data("AAPL", MockS3(), new_data)

    assert len(merged_df) == 2
    assert last_key == "raw_valid.csv"
    mock_read_csv.assert_called_once()


@patch("stock_predictor.data_collector_prices.pd.read_csv")
def test_merge_raw_data_removes_duplicates(mock_read_csv):
    """
    Test that duplicate timestamps are removed keeping the first occurrence.
    :param mock_read_csv: Mocked pandas.read_csv function.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {"Contents": [{"Key": "raw_dup.csv"}]}

        def get_object(self, Bucket, Key):
            return {"Body": "fake_body"}

    new_data = pd.DataFrame(
        {"value": [1]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    new_data.index.name = "t"

    duplicate_df = pd.DataFrame(
        {"value": [999]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    duplicate_df.index.name = "t"

    mock_read_csv.return_value = duplicate_df

    merged_df, _ = merge_raw_data("AAPL", MockS3(), new_data)

    assert merged_df.loc["2024-01-01"]["value"] == 1
    assert len(merged_df) == 1


def test_merge_raw_data_no_raw_files():
    """
    Test that if there are no raw_ files, it returns the original dataframe.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {"Contents": [{"Key": "processed.csv"}]}

    new_data = pd.DataFrame(
        {"value": [1]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    new_data.index.name = "t"

    merged_df, last_key = merge_raw_data("AAPL", MockS3(), new_data)

    assert merged_df.equals(new_data)
    assert last_key == "processed.csv"


def test_merge_raw_data_empty_bucket():
    """
    Test behavior when bucket is empty (detects potential bug).
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {}

    new_data = pd.DataFrame(
        {"value": [1]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    new_data.index.name = "t"

    _, old_data = merge_raw_data("AAPL", MockS3(), new_data)
    assert old_data == ""
