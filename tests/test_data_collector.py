import pytest
from unittest.mock import patch, Mock
from stock_predictor.data_collector import fetch_stock_data_month
import requests


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
