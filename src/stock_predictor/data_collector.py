import os
import requests
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import calendar
from time import sleep
import boto3

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")

def fetch_stock_data_month(ticker: str, year: int, month: int) -> dict:
    """
    Fetch historical stock data from Massive API for a specific month.

    Args:
        ticker (str): Stock ticker symbol.
        year (int): Year (e.g., 2024)
        month (int): Month (1-12)

    Returns:
        dict: JSON response containing stock data for the month.
    """
    start_date = datetime(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end_date = datetime(year, month, last_day)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    url = (
        f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/10/minute/"
        f"{start_str}/{end_str}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
    )
    
    response = requests.get(url)
    response.raise_for_status()
    
    return response.json()

def fetch_last_2_years(ticker: str) -> list:
    """
    Fetch historical stock data for the last 2 years by aggregating monthly data.

    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL")

    Returns:
        list: List of all aggregated data for the last 2 years
    """
    today = datetime.today()
    start_date = today - timedelta(days=2*365)
    
    all_data = []
    
    year = start_date.year
    month = start_date.month
    
    while (year < today.year) or (year == today.year and month <= today.month):
        try:
            print(f"downloading {ticker} {year}-{month:02d}...")
            month_data = fetch_stock_data_month(ticker, year, month)
            sleep(12) 
            if 'results' in month_data:
                all_data.extend(month_data['results'])
            else:
                print(f"No data for {ticker} {year}-{month:02d}")
        except Exception as e:
            print(f"Error fetching data for {ticker} {year}-{month:02d}: {e}")
            
        month += 1
        if month > 12:
            month = 1
            year += 1
    
    return all_data

def process_stock_data(data: dict) -> pd.DataFrame:
    """
    Process raw stock data into a pandas DataFrame.

    Args:
        data (dict): Raw JSON data from the API.

    Returns:
        pd.DataFrame: Processed DataFrame with datetime index and average price.
    """
    df = pd.json_normalize(data)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("t", inplace=True)
    df["average_price"] = (df["h"] + df["l"]) / 2
    return df

def create_s3_client():
    """
    Create and return an S3 client using boto3.

    Returns:
        boto3.client: Configured S3 client.
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("ENDPOINT_URL"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD"),
    )
    return s3

if __name__ == "__main__":
    s3 = create_s3_client()

