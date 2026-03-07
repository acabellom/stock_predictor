from dotenv import load_dotenv
import os
from datetime import datetime
import requests

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
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&published_utc.gte={date_str}T00:00:00Z&published_utc.lte={date_str}T23:59:59Z&apiKey={API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def extract_headlines(news_data: dict) -> list:
    """
    Extract headlines from the news data.

    Args:
        news_data (dict): JSON response containing news data

    Returns:
        list: List of headlines
    """
    return [
        f"{article['title']}: {article.get('description', '')}"
        for article in news_data.get("results", [])
    ]


if __name__ == "__main__":
    news_data = fetch_news_data("AAPL", datetime(2024, 6, 1))
    print(extract_headlines(news_data))
