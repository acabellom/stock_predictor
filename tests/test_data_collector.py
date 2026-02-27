import pytest
from unittest.mock import patch, Mock
from stock_predictor.data_collector import (
    fetch_stock_data_month,
    fetch_last_2_years,
    fetch_last_month_,
    process_stock_data,
    create_s3_client,
)
import requests
from datetime import datetime
import pandas as pd
import boto3


@patch("stock_predictor.data_collector.requests.get")
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


@patch("stock_predictor.data_collector.requests.get")
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


@patch("stock_predictor.data_collector.requests.get")
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


@patch("stock_predictor.data_collector.requests.get")
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


@patch("stock_predictor.data_collector.sleep")
@patch("stock_predictor.data_collector.fetch_last_month_")
@patch("stock_predictor.data_collector.fetch_stock_data_month")
@patch("stock_predictor.data_collector.datetime")
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


@patch("stock_predictor.data_collector.sleep")
@patch("stock_predictor.data_collector.fetch_last_month_")
@patch("stock_predictor.data_collector.fetch_stock_data_month")
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


@patch("stock_predictor.data_collector.requests.get")
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
    # check that start and end dates appear in URL (YYYY-MM-DD format)
    today = datetime.today()
    start_month = today.replace(day=1).strftime("%Y-%m-%d")
    end_day = today.strftime("%Y-%m-%d")
    assert start_month in called_url
    assert end_day in called_url


@patch("stock_predictor.data_collector.requests.get")
def test_fetch_last_month_http_error(mock_get):
    """
    Test fetching last month's stock data with an HTTP error.

    :param mock_get: Mocked requests.get method.
    """
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("API error")
    mock_get.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        fetch_last_month_("AAPL")


@patch("stock_predictor.data_collector.requests.get")
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

    fetch_last_month_("TSLA")

    called_url = mock_get.call_args[0][0]
    assert "/TSLA/" in called_url


@patch("stock_predictor.data_collector.requests.get")
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

    result = fetch_last_month_("AAPL")

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
    )  # <-- manejar caso de lista vacía

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

    # Eliminar variables de entorno por si existen
    monkeypatch.delenv("ENDPOINT_URL", raising=False)
    monkeypatch.delenv("ROOT_USER", raising=False)
    monkeypatch.delenv("ROOT_PASSWORD", raising=False)

    client = create_s3_client()

    assert client == "mock_s3_client"
    assert calls["endpoint_url"] is None
    assert calls["aws_access_key_id"] is None
    assert calls["aws_secret_access_key"] is None
