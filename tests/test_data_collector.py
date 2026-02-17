import pytest
from unittest.mock import patch, Mock
from stock_predictor.data_collector import (
    fetch_stock_data_month,
    fetch_last_2_years,
    fetch_last_month_,
)
import requests
from datetime import datetime


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
