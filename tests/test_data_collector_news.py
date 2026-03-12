import pytest
from unittest.mock import patch, Mock
from stock_predictor.data_collector_news import (
    fetch_news_data,
    extract_headlines,
    clean_data,
    get_dataframe,
    get_sentiment_analysis,
)
import requests
from datetime import datetime
import pandas as pd


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


def test_extract_headlines_basic():
    """
    Test extracting headlines when news data contains title, description and published_utc.
    """

    news_data = {
        "results": [
            {
                "title": "Apple launches new iPhone",
                "description": "The new model improves battery life.",
                "published_utc": "2024-01-01T10:00:00Z",
            }
        ]
    }

    result = extract_headlines(news_data)

    assert isinstance(result, list)
    assert len(result) == 1
    assert (
        result[0][0]
        == "Apple launches new iPhone: The new model improves battery life."
    )
    assert result[0][1] == "2024-01-01T10:00:00Z"


def test_extract_headlines_missing_description():
    """
    Test extracting headlines when description is missing.
    """

    news_data = {
        "results": [
            {
                "title": "Apple stock rises",
                "published_utc": "2024-01-02T12:00:00Z",
            }
        ]
    }

    result = extract_headlines(news_data)

    assert len(result) == 1
    assert result[0][0] == "Apple stock rises: "
    assert result[0][1] == "2024-01-02T12:00:00Z"


def test_extract_headlines_missing_published_date():
    """
    Test extracting headlines when published_utc is missing.
    """

    news_data = {
        "results": [
            {
                "title": "Apple earnings report",
                "description": "Revenue beats expectations",
            }
        ]
    }

    result = extract_headlines(news_data)

    assert len(result) == 1
    assert result[0][0] == "Apple earnings report: Revenue beats expectations"
    assert result[0][1] == ""


def test_extract_headlines_multiple_articles():
    """
    Test extracting multiple headlines correctly.
    """

    news_data = {
        "results": [
            {
                "title": "News 1",
                "description": "Desc 1",
                "published_utc": "2024-01-01",
            },
            {
                "title": "News 2",
                "description": "Desc 2",
                "published_utc": "2024-01-02",
            },
        ]
    }

    result = extract_headlines(news_data)

    assert len(result) == 2
    assert result[0][0] == "News 1: Desc 1"
    assert result[1][0] == "News 2: Desc 2"


def test_extract_headlines_empty_results():
    """
    Test behavior when results list is empty.
    """

    news_data = {"results": []}

    result = extract_headlines(news_data)

    assert result == []


def test_extract_headlines_no_results_key():
    """
    Test behavior when results key does not exist.
    """

    news_data = {}

    result = extract_headlines(news_data)

    assert result == []


def test_clean_data_basic():
    """
    Test cleaning headlines with newline characters.
    """

    headlines = [
        ("Apple releases new iPhone\nwith better battery", "2024-01-01"),
    ]

    result = clean_data(headlines)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0][0] == "Apple releases new iPhone with better battery"
    assert result[0][1] == "2024-01-01"


def test_clean_data_strips_spaces():
    """
    Test that leading and trailing spaces are removed.
    """

    headlines = [
        ("   Apple stock rises   ", "2024-01-02"),
    ]

    result = clean_data(headlines)

    assert result[0][0] == "Apple stock rises"


def test_clean_data_multiple_headlines():
    """
    Test cleaning multiple headlines correctly.
    """

    headlines = [
        ("News 1\n", "2024-01-01"),
        ("\nNews 2", "2024-01-02"),
    ]

    result = clean_data(headlines)

    assert len(result) == 2
    assert result[0][0] == "News 1"
    assert result[1][0] == "News 2"


def test_clean_data_preserves_dates():
    """
    Test that published dates remain unchanged.
    """

    headlines = [
        ("Headline text\n", "2024-01-01T10:00:00Z"),
    ]

    result = clean_data(headlines)

    assert result[0][1] == "2024-01-01T10:00:00Z"


def test_clean_data_empty_list():
    """
    Test behavior when input list is empty.
    """

    headlines = []

    result = clean_data(headlines)

    assert result == []


def test_clean_data_no_newlines():
    """
    Test behavior when headlines do not contain newline characters.
    """

    headlines = [
        ("Apple earnings beat expectations", "2024-01-03"),
    ]

    result = clean_data(headlines)

    assert result[0][0] == "Apple earnings beat expectations"


def test_get_dataframe_basic():
    """
    Test converting a list of headlines into a DataFrame.
    """

    headlines = [
        ("Apple launches new product", "2024-01-01"),
        ("Apple stock rises", "2024-01-02"),
    ]

    df = get_dataframe(headlines)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["headline", "published_utc"]

    assert df.iloc[0]["headline"] == "Apple launches new product"
    assert df.iloc[1]["published_utc"] == "2024-01-02"


def test_get_dataframe_empty_list():
    """
    Test behavior when input list is empty.
    """

    headlines = []

    df = get_dataframe(headlines)

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == ["headline", "published_utc"]


def test_get_dataframe_single_row():
    """
    Test DataFrame creation when only one headline is provided.
    """

    headlines = [
        ("Apple earnings beat expectations", "2024-01-03"),
    ]

    df = get_dataframe(headlines)

    assert len(df) == 1
    assert df.iloc[0]["headline"] == "Apple earnings beat expectations"
    assert df.iloc[0]["published_utc"] == "2024-01-03"


def test_get_dataframe_preserves_order():
    """
    Test that the order of headlines is preserved in the DataFrame.
    """

    headlines = [
        ("News 1", "2024-01-01"),
        ("News 2", "2024-01-02"),
        ("News 3", "2024-01-03"),
    ]

    df = get_dataframe(headlines)

    assert df["headline"].tolist() == ["News 1", "News 2", "News 3"]


def test_get_dataframe_column_types():
    """
    Test that DataFrame columns contain the expected values.
    """

    headlines = [
        ("Headline text", "2024-01-01T10:00:00Z"),
    ]

    df = get_dataframe(headlines)

    assert "headline" in df.columns
    assert "published_utc" in df.columns


@patch("stock_predictor.data_collector_news.tqdm", lambda x, **kwargs: x)
@patch("stock_predictor.data_collector_news.pipeline")
def test_get_sentiment_analysis_basic(mock_pipeline):
    """
    Test sentiment analysis producing correct positive/neutral/negative scores.
    """

    df = pd.DataFrame(
        {
            "headline": [
                "Apple stock rises",
                "Apple earnings disappoint investors",
            ]
        }
    )

    mock_model = Mock()

    mock_model.return_value = [
        [
            {"label": "positive", "score": 0.8},
            {"label": "neutral", "score": 0.1},
            {"label": "negative", "score": 0.1},
        ],
        [
            {"label": "positive", "score": 0.2},
            {"label": "neutral", "score": 0.3},
            {"label": "negative", "score": 0.5},
        ],
    ]

    mock_pipeline.return_value = mock_model

    result = get_sentiment_analysis(df)

    assert isinstance(result, pd.DataFrame)

    assert result.loc[0, "positive"] == 0.8
    assert result.loc[0, "negative"] == 0.1
    assert result.loc[0, "neutral"] == 0.1

    assert result.loc[1, "positive"] == 0.2
    assert result.loc[1, "negative"] == 0.5

    assert result.loc[0, "sentiment"] == pytest.approx(0.7)
    assert result.loc[1, "sentiment"] == pytest.approx(-0.3)


@patch("stock_predictor.data_collector_news.tqdm", lambda x, **kwargs: x)
@patch("stock_predictor.data_collector_news.pipeline")
def test_get_sentiment_analysis_handles_nan(mock_pipeline):
    """
    Test that NaN headlines are converted to empty strings.
    """

    df = pd.DataFrame({"headline": [None]})

    mock_model = Mock()

    mock_model.return_value = [
        [
            {"label": "positive", "score": 0.5},
            {"label": "neutral", "score": 0.4},
            {"label": "negative", "score": 0.1},
        ]
    ]

    mock_pipeline.return_value = mock_model

    result = get_sentiment_analysis(df)

    assert result["headline"].iloc[0] == ""
    assert result["positive"].iloc[0] == 0.5


@patch("stock_predictor.data_collector_news.tqdm", lambda x, **kwargs: x)
@patch("stock_predictor.data_collector_news.pipeline")
def test_get_sentiment_analysis_truncates_headline(mock_pipeline):
    """
    Test that headlines longer than 512 characters are truncated.
    """

    long_text = "A" * 600

    df = pd.DataFrame({"headline": [long_text]})

    mock_model = Mock()

    mock_model.return_value = [
        [
            {"label": "positive", "score": 0.3},
            {"label": "neutral", "score": 0.5},
            {"label": "negative", "score": 0.2},
        ]
    ]

    mock_pipeline.return_value = mock_model

    get_sentiment_analysis(df)

    called_batch = mock_model.call_args[0][0]

    assert len(called_batch[0]) == 512


@patch("stock_predictor.data_collector_news.tqdm", lambda x, **kwargs: x)
@patch("stock_predictor.data_collector_news.pipeline")
def test_get_sentiment_analysis_empty_dataframe(mock_pipeline):
    """
    Test behavior when dataframe is empty.
    """

    df = pd.DataFrame(columns=["headline"])

    mock_model = Mock()
    mock_model.return_value = []

    mock_pipeline.return_value = mock_model

    result = get_sentiment_analysis(df)

    assert result.empty
    assert "positive" in result.columns
    assert "negative" in result.columns
    assert "neutral" in result.columns
    assert "sentiment" in result.columns


@patch("stock_predictor.data_collector_news.tqdm", lambda x, **kwargs: x)
@patch("stock_predictor.data_collector_news.pipeline")
def test_get_sentiment_analysis_batching(mock_pipeline):
    """
    Test that batching works correctly with more than 32 headlines.
    """

    df = pd.DataFrame({"headline": [f"News {i}" for i in range(40)]})

    mock_model = Mock()

    mock_model.return_value = [
        [
            {"label": "positive", "score": 0.5},
            {"label": "neutral", "score": 0.3},
            {"label": "negative", "score": 0.2},
        ]
    ] * 32

    mock_pipeline.return_value = mock_model

    get_sentiment_analysis(df)

    assert mock_model.call_count == 2
