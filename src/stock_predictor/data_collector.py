import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Fetch historical stock data from Polygon.io API.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        dict: JSON response containing stock data.
    """
    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/5/minute/{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    # Example usage
    data = fetch_stock_data("AAPL", "2023-12-28", "2025-12-27")
    with open("./data/aapl_5min.json", "w") as f:
        json.dump(data, f, indent=4)