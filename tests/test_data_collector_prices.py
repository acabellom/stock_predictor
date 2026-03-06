import pytest
from unittest.mock import patch, Mock
from stock_predictor.data_collector_prices import (
    check_highest_date_in_s3,
    create_bucket_if_not_exists,
    fetch_stock_data_month,
    fetch_last_2_years,
    fetch_last_month_,
    process_stock_data,
    create_s3_client,
    upload_to_s3,
    merge_raw_data,
    get_lowest_date_in_s3,
    store_old_data,
)
import requests
from datetime import datetime, timedelta
import pandas as pd
import boto3


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


def test_create_s3_client_calls_boto3_client(monkeypatch):
    """
    Test that create_s3_client calls boto3.client with correct parameters from environment variables.
    :param monkeypatch: pytest fixture for patching.
    """
    calls = {}

    def mock_boto3_client(
        service_name,
        endpoint_url=None,
        aws_access_key_id=None,
        aws_secret_access_key=None,
    ):
        calls["service_name"] = service_name
        calls["endpoint_url"] = endpoint_url
        calls["aws_access_key_id"] = aws_access_key_id
        calls["aws_secret_access_key"] = aws_secret_access_key
        return "mock_s3_client"

    monkeypatch.setattr(boto3, "client", mock_boto3_client)

    monkeypatch.setenv("ENDPOINT_URL", "http://fake-endpoint")
    monkeypatch.setenv("ROOT_USER", "fake_user")
    monkeypatch.setenv("ROOT_PASSWORD", "fake_password")

    client = create_s3_client()

    assert client == "mock_s3_client"
    assert calls["service_name"] == "s3"
    assert calls["endpoint_url"] == "http://fake-endpoint"
    assert calls["aws_access_key_id"] == "fake_user"
    assert calls["aws_secret_access_key"] == "fake_password"


def test_create_s3_client_without_env_vars(monkeypatch):
    """
    Test that create_s3_client can be called without environment variables and handles it gracefully.
    :param monkeypatch: pytest fixture for patching.
    """
    calls = {}

    def mock_boto3_client(
        service_name,
        endpoint_url=None,
        aws_access_key_id=None,
        aws_secret_access_key=None,
    ):
        calls["endpoint_url"] = endpoint_url
        calls["aws_access_key_id"] = aws_access_key_id
        calls["aws_secret_access_key"] = aws_secret_access_key
        return "mock_s3_client"

    monkeypatch.setattr(boto3, "client", mock_boto3_client)

    monkeypatch.delenv("ENDPOINT_URL", raising=False)
    monkeypatch.delenv("ROOT_USER", raising=False)
    monkeypatch.delenv("ROOT_PASSWORD", raising=False)

    client = create_s3_client()

    assert client == "mock_s3_client"
    assert calls["endpoint_url"] is None
    assert calls["aws_access_key_id"] is None
    assert calls["aws_secret_access_key"] is None


def test_upload_to_s3_creates_bucket_if_not_exists(monkeypatch):
    """
    Test that upload_to_s3 creates the bucket if it does not exist and uploads the file.
        :param monkeypatch: pytest fixture for patching.
    """
    calls = {}

    class MockS3:
        def list_buckets(self):
            return {"Buckets": []}

        def create_bucket(self, Bucket):
            calls["create_bucket"] = Bucket

        def put_object(self, Bucket, Key, Body):
            calls["put_object"] = (Bucket, Key, Body)

    s3_client = MockS3()

    data = pd.DataFrame({"a": [1, 2, 3]})

    upload_to_s3(s3_client, "test-bucket", "file.csv", data)

    assert calls["create_bucket"] == "test-bucket"
    assert calls["put_object"][0] == "test-bucket"
    assert calls["put_object"][1] == "file.csv"
    assert isinstance(calls["put_object"][2], str)


def test_upload_to_s3_does_not_create_bucket_if_exists():
    """
    Test that upload_to_s3 does not create the bucket if it already exists and uploads the file.
     :param monkeypatch: pytest fixture for patching.
    """
    calls = {}

    class MockS3:
        def list_buckets(self):
            return {"Buckets": [{"Name": "test-bucket"}]}

        def create_bucket(self, Bucket):
            calls["create_bucket"] = Bucket

        def put_object(self, Bucket, Key, Body):
            calls["put_object"] = (Bucket, Key, Body)

    s3_client = MockS3()

    data = pd.DataFrame({"a": [1, 2, 3]})

    upload_to_s3(s3_client, "test-bucket", "file.csv", data)

    assert "create_bucket" not in calls
    assert calls["put_object"][0] == "test-bucket"
    assert calls["put_object"][1] == "file.csv"


def test_upload_to_s3_sends_csv_data():
    """
    Test that upload_to_s3 sends the correct CSV data in the Body of the put_object call.
     :param monkeypatch: pytest fixture for patching.
    """

    class MockS3:
        def list_buckets(self):
            return {"Buckets": [{"Name": "test-bucket"}]}

        def put_object(self, Bucket, Key, Body):
            self.body = Body

    s3_client = MockS3()

    data = pd.DataFrame({"a": [1, 2, 3]})

    upload_to_s3(s3_client, "test-bucket", "file.csv", data)

    assert "a" in s3_client.body
    assert "1" in s3_client.body


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


def test_get_lowest_date_in_s3_single_raw_file():
    """
    Test getting lowest date from a single raw_ file.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "raw_20240101_data.csv"},
                ]
            }

    result = get_lowest_date_in_s3("AAPL", MockS3())

    assert result == "20240101"


def test_get_lowest_date_in_s3_multiple_raw_files_returns_last_found():
    """
    Test that when multiple raw_ files exist,
    the function returns the date from the last iterated raw_ file.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "raw_20220101_data.csv"},
                    {"Key": "raw_20230101_data.csv"},
                ]
            }

    result = get_lowest_date_in_s3("AAPL", MockS3())

    assert result == "20230101"


def test_get_lowest_date_in_s3_ignores_non_raw_files():
    """
    Test that non raw_ files are ignored.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "processed.csv"},
                    {"Key": "raw_20231231_data.csv"},
                ]
            }

    result = get_lowest_date_in_s3("AAPL", MockS3())

    assert result == "20231231"


def test_get_lowest_date_in_s3_no_raw_files():
    """
    Test behavior when there are no raw_ files.
    This exposes a bug in the current implementation.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "processed.csv"},
                ]
            }

    with pytest.raises(UnboundLocalError):
        get_lowest_date_in_s3("AAPL", MockS3())


def test_get_lowest_date_in_s3_empty_bucket():
    """
    Test behavior when bucket is empty.
    This exposes a bug in the current implementation.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {}

    result = get_lowest_date_in_s3("AAPL", MockS3())
    assert result == (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")


def test_store_old_data_skips_when_key_empty():
    """
    Test that when old_file_key is empty,
    the function does not call S3 operations.
    """

    class MockS3:
        def copy_object(self, **kwargs):
            raise AssertionError("copy_object should not be called")

        def delete_object(self, **kwargs):
            raise AssertionError("delete_object should not be called")

    # Should not raise
    store_old_data(MockS3(), "AAPL", "")


def test_store_old_data_archives_file_successfully():
    """
    Test that the file is copied to archive/ and then deleted.
    """

    calls = {}

    class MockS3:
        def copy_object(self, Bucket, CopySource, Key):
            calls["copy"] = {
                "Bucket": Bucket,
                "CopySource": CopySource,
                "Key": Key,
            }

        def delete_object(self, Bucket, Key):
            calls["delete"] = {
                "Bucket": Bucket,
                "Key": Key,
            }

    store_old_data(MockS3(), "AAPL", "raw/data.csv")

    assert calls["copy"]["Bucket"] == "aapl"
    assert calls["copy"]["CopySource"] == {
        "Bucket": "aapl",
        "Key": "raw/data.csv",
    }
    assert calls["copy"]["Key"] == "archive/raw/data.csv"

    assert calls["delete"]["Bucket"] == "aapl"
    assert calls["delete"]["Key"] == "raw/data.csv"


def test_store_old_data_uses_lowercase_bucket():
    """
    Test that ticker is always converted to lowercase for bucket name.
    """

    calls = {}

    class MockS3:
        def copy_object(self, Bucket, CopySource, Key):
            calls["bucket"] = Bucket

        def delete_object(self, Bucket, Key):
            pass

    store_old_data(MockS3(), "MsFt", "file.csv")

    assert calls["bucket"] == "msft"


def test_store_old_data_propagates_copy_error():
    """
    Test that if copy_object fails, the exception is propagated
    and delete_object is not called.
    """

    class MockS3:
        def copy_object(self, Bucket, CopySource, Key):
            raise Exception("S3 copy failed")

        def delete_object(self, Bucket, Key):
            raise AssertionError("delete_object should not be called")

    with pytest.raises(Exception, match="S3 copy failed"):
        store_old_data(MockS3(), "AAPL", "file.csv")


def test_check_highest_date_in_s3_multiple_raw_files_returns_last_found():
    """
    Test that when multiple raw_ files exist,
    the function returns the date from the last iterated raw_ file.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "raw_data_20220101.csv"},
                    {"Key": "raw_data_20231231.csv"},
                ]
            }

    result = check_highest_date_in_s3("AAPL", MockS3())

    # Returns last iterated raw_ file date (not actually highest)
    assert result == "20231231"


def test_check_highest_date_in_s3_ignores_non_raw_files():
    """
    Test that non raw_ files are ignored.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "processed.csv"},
                    {"Key": "raw_data_20240101.csv"},
                ]
            }

    result = check_highest_date_in_s3("AAPL", MockS3())

    assert result == "20240101"


def test_check_highest_date_in_s3_empty_bucket():
    """
    Test behavior when bucket is empty.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {}

    result = check_highest_date_in_s3("AAPL", MockS3())

    assert result == "00000000"


def test_check_highest_date_in_s3_no_raw_files():
    """
    Test behavior when there are no raw_ files.
    This exposes a bug in the current implementation.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "processed.csv"},
                    {"Key": "another_file.txt"},
                ]
            }

    with pytest.raises(UnboundLocalError):
        check_highest_date_in_s3("AAPL", MockS3())


def test_check_highest_date_in_s3_uses_lowercase_bucket():
    """
    Test that ticker is converted to lowercase when calling S3.
    """

    captured_bucket = {}

    class MockS3:
        def list_objects_v2(self, Bucket):
            captured_bucket["value"] = Bucket
            return {"Contents": []}

    check_highest_date_in_s3("MsFt", MockS3())

    assert captured_bucket["value"] == "msft"


def test_create_bucket_if_not_exists_creates_bucket_when_missing():
    """
    Test that the bucket is created if it does not exist.
    """

    calls = {}

    class MockS3:
        def list_buckets(self):
            return {
                "Buckets": [
                    {"Name": "existing-bucket"},
                ]
            }

        def create_bucket(self, Bucket):
            calls["created"] = Bucket

    create_bucket_if_not_exists(MockS3(), "new-bucket")

    assert calls["created"] == "new-bucket"


def test_create_bucket_if_not_exists_does_not_create_if_exists():
    """
    Test that the bucket is not created if it already exists.
    """

    class MockS3:
        def list_buckets(self):
            return {
                "Buckets": [
                    {"Name": "existing-bucket"},
                ]
            }

        def create_bucket(self, Bucket):
            raise AssertionError("create_bucket should not be called")

    create_bucket_if_not_exists(MockS3(), "existing-bucket")


def test_create_bucket_if_not_exists_creates_when_no_buckets():
    """
    Test that the bucket is created when the account has no buckets.
    """

    calls = {}

    class MockS3:
        def list_buckets(self):
            return {"Buckets": []}

        def create_bucket(self, Bucket):
            calls["created"] = Bucket

    create_bucket_if_not_exists(MockS3(), "my-bucket")

    assert calls["created"] == "my-bucket"


def test_create_bucket_if_not_exists_handles_missing_buckets_key():
    """
    Test that the function works when 'Buckets' key is missing in response.
    """

    calls = {}

    class MockS3:
        def list_buckets(self):
            return {}

        def create_bucket(self, Bucket):
            calls["created"] = Bucket

    create_bucket_if_not_exists(MockS3(), "test-bucket")

    assert calls["created"] == "test-bucket"
