from dotenv import load_dotenv
import os
from datetime import datetime
import requests
import pandas as pd
from transformers import pipeline

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")


def fetch_news_data(ticker: str, date: datetime) -> dict:
    """
    Fetch news data for a specific ticker and date from the Polygon API.

    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL")
        date (datetime): Date for which to fetch news data
    Returns:
        dict: JSON response containing news data for the specified ticker and date
    """
    date_2_years_ago = date.replace(year=date.year - 2)
    date_str = date.strftime("%Y-%m-%d")
    date_str_2_years_ago = date_2_years_ago.strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&published_utc.gte={date_str_2_years_ago}T00:00:00Z&published_utc.lte={date_str}T23:59:59Z&apiKey={API_KEY}&limit=1000"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    all_results = data.get("results", [])
    next_url = data.get("next_url")
    while next_url:
        next_url = next_url + f"&apiKey={API_KEY}"
        next_response = requests.get(next_url)
        next_response.raise_for_status()
        all_results.extend(next_response.json().get("results", []))
        next_url = next_response.json().get("next_url")
    data["results"] = all_results
    return data


def extract_headlines(news_data: dict) -> list:
    """
    Extract headlines from the news data.

    Args:
        news_data (dict): JSON response containing news data

    Returns:
        list: List of headlines
    """
    return [
        (
            f"{article['title']}: {article.get('description', '')}",
            article.get("published_utc", ""),
        )
        for article in news_data.get("results", [])
    ]


def clean_data(headlines: list) -> list:
    """
    Clean the headlines by removing any unwanted characters or formatting.

    Args:
        headlines (list): List of tuples containing headlines and their published dates

    Returns:
        list: Cleaned list of headlines
    """
    cleaned_headlines = []
    for headline, published_utc in headlines:
        cleaned_headline = headline.replace("\n", " ").strip()
        cleaned_headlines.append((cleaned_headline, published_utc))
    return cleaned_headlines


def get_dataframe(headlines: list) -> pd.DataFrame:
    """
    Convert the list of headlines into a pandas DataFrame.

    Args:
        headlines (list): List of tuples containing headlines and their published dates

    Returns:
        pd.DataFrame: DataFrame with columns "headline" and "published_utc"
    """
    return pd.DataFrame(headlines, columns=["headline", "published_utc"])


def get_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform sentiment analysis on the headlines using a pre-trained model.

    Args:
        df (pd.DataFrame): DataFrame containing headlines and their published dates

    Returns:
        pd.DataFrame: DataFrame with an additional column for sentiment labels
    """
    texts = df["headline"].str[:512].tolist()
    sentiment_analyzer = pipeline(
        "sentiment-analysis", model="ProsusAI/finbert", return_all_scores=True
    )
    results = sentiment_analyzer(texts, batch_size=32)
    df["positive"] = [r[0]["score"] for r in results]
    df["neutral"] = [r[1]["score"] for r in results]
    df["negative"] = [r[2]["score"] for r in results]
    return df


if __name__ == "__main__":
    news_data = fetch_news_data("AAPL", datetime.now())
    df = extract_headlines(news_data)
    df = clean_data(df)
    df = get_dataframe(df)
    df = get_sentiment_analysis(df)

    df.to_csv("./data/aapl_news_test.csv", index=False, quoting=1)
