import pytest
from unittest.mock import patch, Mock
from stock_predictor.data_collector_news import fetch_news_data
import requests
from datetime import datetime


@patch("stock_predictor.data_collector_news.requests.get")
def test_fetch_news_data_success_single_page(mock_get):
    """
    Test fetching news data successfully when the API returns a single page.
    :param mock_get: Mocked requests.get method.
    """

    fake_response = {
        "results": [
            {"title": "Apple releases new product"},
            {"title": "Apple stock rises"},
        ],
        "next_url": None,
    }

    mock_get.return_value = Mock(
        status_code=200,
        json=Mock(return_value=fake_response),
        raise_for_status=Mock(),
    )

    result = fetch_news_data("AAPL", datetime(2024, 1, 1))

    assert "results" in result
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Apple releases new product"

    mock_get.assert_called_once()

    called_url = mock_get.call_args[0][0]
    assert "AAPL" in called_url
    assert "2022-01-01" in called_url
    assert "2024-01-01" in called_url


@patch("stock_predictor.data_collector_news.requests.get")
def test_fetch_news_data_pagination(mock_get):
    """
    Test that pagination using next_url works correctly.
    :param mock_get: Mocked requests.get method.
    """

    first_response = {
        "results": [{"title": "News 1"}],
        "next_url": "http://next-page",
    }

    second_response = {
        "results": [{"title": "News 2"}],
        "next_url": None,
    }

    mock_get.side_effect = [
        Mock(
            status_code=200,
            json=Mock(return_value=first_response),
            raise_for_status=Mock(),
        ),
        Mock(
            status_code=200,
            json=Mock(return_value=second_response),
            raise_for_status=Mock(),
        ),
    ]

    result = fetch_news_data("AAPL", datetime(2024, 1, 1))

    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "News 1"
    assert result["results"][1]["title"] == "News 2"

    assert mock_get.call_count == 2


@patch("stock_predictor.data_collector_news.requests.get")
def test_fetch_news_data_http_error(mock_get):
    """
    Test that an HTTP error from the API raises an exception.
    :param mock_get: Mocked requests.get method.
    """

    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("API error")

    mock_get.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        fetch_news_data("AAPL", datetime(2024, 1, 1))


@patch("stock_predictor.data_collector_news.requests.get")
def test_fetch_news_data_empty_results(mock_get):
    """
    Test behavior when the API returns no results.
    :param mock_get: Mocked requests.get method.
    """

    fake_response = {"results": [], "next_url": None}

    mock_get.return_value = Mock(
        status_code=200,
        json=Mock(return_value=fake_response),
        raise_for_status=Mock(),
    )

    result = fetch_news_data("AAPL", datetime(2024, 1, 1))

    assert isinstance(result, dict)
    assert result["results"] == []


@patch("stock_predictor.data_collector_news.requests.get")
def test_fetch_news_data_multiple_pages(mock_get):
    """
    Test that multiple pagination pages are merged correctly.
    :param mock_get: Mocked requests.get method.
    """

    page1 = {"results": [{"title": "News 1"}], "next_url": "url2"}
    page2 = {"results": [{"title": "News 2"}], "next_url": "url3"}
    page3 = {"results": [{"title": "News 3"}], "next_url": None}

    mock_get.side_effect = [
        Mock(status_code=200, json=Mock(return_value=page1), raise_for_status=Mock()),
        Mock(status_code=200, json=Mock(return_value=page2), raise_for_status=Mock()),
        Mock(status_code=200, json=Mock(return_value=page3), raise_for_status=Mock()),
    ]

    result = fetch_news_data("AAPL", datetime(2024, 1, 1))

    assert len(result["results"]) == 3
    assert result["results"][0]["title"] == "News 1"
    assert result["results"][2]["title"] == "News 3"

    assert mock_get.call_count == 3
