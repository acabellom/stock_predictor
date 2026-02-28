import os
import requests
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import calendar
from time import sleep
import boto3
from stock_predictor.logger_config import logger

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
    start_date = today - timedelta(days=2 * 365)

    all_data = []

    year = start_date.year
    month = start_date.month

    while (year < today.year) or (year == today.year and month < today.month):
        try:
            logger.info(f"downloading {ticker} {year}-{month:02d}...")
            month_data = fetch_stock_data_month(ticker, year, month)
            sleep(12)
            if "results" in month_data:
                all_data.extend(month_data["results"])
            else:
                logger.warning(f"No data for {ticker} {year}-{month:02d}")
        except Exception as e:
            logger.error(f"Error fetching data for {ticker} {year}-{month:02d}: {e}")

        month += 1
        if month > 12:
            month = 1
            year += 1
    try:
        all_data.extend(fetch_last_month_(ticker)["results"])
    except Exception as e:
        logger.error(f"Error fetching data for last month {ticker}: {e}")

    return all_data


def fetch_last_month_(ticker: str) -> dict:
    """
    Fetch stock data for the last month up to today.
    Args:
        ticker (str): Stock ticker symbol.
    Returns:
        dict: JSON response containing stock data for the last month.
    """

    today = datetime.today()
    start_month = today.replace(day=1)
    start_str = start_month.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    url = (
        f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/10/minute/"
        f"{start_str}/{end_str}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
    )
    response = requests.get(url)
    response.raise_for_status()

    return response.json()


def process_stock_data(data: dict) -> pd.DataFrame:
    """
    Process raw stock data into a pandas DataFrame.

    Args:
        data (dict): Raw JSON data from the API.

    Returns:
        pd.DataFrame: Processed DataFrame with datetime index and average price.
    """
    logger.info("Processing stock data...")
    df = pd.json_normalize(data)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("t", inplace=True)
    df["average_price"] = (df["h"] + df["l"]) / 2
    logger.info("Stock data processed successfully.")
    return df


def create_s3_client():
    """
    Create and return an S3 client using boto3.

    Returns:
        boto3.client: Configured S3 client.
    """
    logger.info("Creating S3 client...")
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("ENDPOINT_URL"),
        aws_access_key_id=os.getenv("ROOT_USER"),
        aws_secret_access_key=os.getenv("ROOT_PASSWORD"),
    )
    return s3


def upload_to_s3(s3_client, bucket_name: str, file_name: str, data: pd.DataFrame):
    """
    Upload a DataFrame as a CSV file to an S3 bucket.

    Args:
        s3_client (boto3.client): S3 client.
        bucket_name (str): Name of the S3 bucket.
        file_name (str): Name of the file to be saved in S3.
        data (pd.DataFrame): DataFrame to upload.
    """
    logger.info(f"Checking if bucket {bucket_name}exists...")
    existing_buckets = [b["Name"] for b in s3_client.list_buckets().get("Buckets", [])]
    if bucket_name not in existing_buckets:
        logger.info(f"Creating bucket {bucket_name}...")
        s3_client.create_bucket(Bucket=bucket_name)
    csv_buffer = data.to_csv(index=True)
    s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer)
    logger.info(f"Uploaded {file_name} to bucket {bucket_name}.")


def merge_raw_data(
    ticker: str, s3_client, new_data: pd.DataFrame
) -> list[pd.DataFrame, str]:
    """
    Merge all raw CSV files from S3 for a given ticker into a single DataFrame.

    Args:
        ticker (str): Stock ticker symbol.
        s3_client (boto3.client): S3 client.

    Returns:
        pd.DataFrame: Merged DataFrame containing all raw data.
    """
    bucket_name = ticker.lower()
    merged_df = new_data.copy()
    logger.info(f"Merging old raw data files from bucket {bucket_name}...")
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    content = response.get("Contents", [])
    if content == []:
        logger.info("No old raw data files found in S3 bucket. Merging skipped.")
        return (merged_df, "")
    for obj in content:
        file_key = obj["Key"]
        if file_key.startswith("raw_"):
            obj_response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            df = pd.read_csv(obj_response["Body"], parse_dates=["t"], index_col="t")
            merged_df = pd.concat([merged_df, df])
    logger.info(f"Merging completed: actual data and {file_key}")
    merged_df = merged_df[~merged_df.index.duplicated(keep="first")]
    merged_df.sort_index(inplace=True)
    return (merged_df, file_key)


def get_lowest_date_in_s3(ticker: str, s3_client) -> str:
    """
    Get the lowest date from the raw data files stored in S3 for a given ticker.
    Args:
        ticker (str): Stock ticker symbol.
        s3_client (boto3.client): S3 client.
    Returns:
        str: The lowest date in 'YYYYMMDD' format.
    """

    bucket_name = ticker.lower()
    logger.info(f"Getting lowest date from S3 bucket {bucket_name}...")
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    content = response.get("Contents", [])
    if content == []:
        logger.info(
            "No files found in S3 bucket. Returning today 2 year ago as lowest date."
        )
        return (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
    for obj in content:
        file_key = obj["Key"]
        if file_key.startswith("raw_"):
            parts = file_key.split("_")
    logger.info(f"Lowest date found: {parts[1]}")
    return parts[1]


def store_old_data(s3_client, ticker: str, old_file_key: str):
    """
    Store the old raw data file in an 'archive' folder within the S3 bucket.

    Args:
        s3_client (boto3.client): S3 client.
        ticker (str): Stock ticker symbol.
        old_file_key (str): The key of the old raw data file to be archived.
    """
    if old_file_key == "":
        logger.info("No old raw data file to archive. Skipping archiving step.")
        return
    logger.info(f"Archiving old data file {old_file_key} in bucket {ticker.lower()}...")
    bucket_name = f"{ticker.lower()}"
    archive_key = f"archive/{old_file_key}"

    s3_client.copy_object(
        Bucket=bucket_name,
        CopySource={"Bucket": bucket_name, "Key": old_file_key},
        Key=archive_key,
    )
    s3_client.delete_object(Bucket=bucket_name, Key=old_file_key)
    logger.info(f"Archived {old_file_key} to {archive_key}.")


def check_highest_date_in_s3(ticker: str, s3_client) -> str:
    """
    Get the highest date from the raw data files stored in S3 for a given ticker.
    Args:
        ticker (str): Stock ticker symbol.
        s3_client (boto3.client): S3 client.
    Returns:
        str: The highest date in 'YYYYMMDD' format.
    """

    bucket_name = ticker.lower()
    logger.info(f"Getting highest date from S3 bucket {bucket_name}...")
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    content = response.get("Contents", [])
    if content == []:
        logger.info(
            "No files found in S3 bucket. Returning '00000000' as highest date."
        )
        return "00000000"
    for obj in response.get("Contents", []):
        file_key = obj["Key"]
        if file_key.startswith("raw_"):
            parts = file_key.split("_")
    logger.info(f"Highest date found: {parts[2].split('.')[0]}")
    return parts[2].split(".")[0]


def create_bucket_if_not_exists(s3_client, bucket_name: str):
    """
    Create an S3 bucket if it does not already exist.

    Args:
        s3_client (boto3.client): S3 client.
        bucket_name (str): Name of the S3 bucket to create.
    """
    logger.info(f"Checking if bucket {bucket_name} exists...")
    existing_buckets = [b["Name"] for b in s3_client.list_buckets().get("Buckets", [])]
    if bucket_name not in existing_buckets:
        logger.info(f"Creating bucket {bucket_name}...")
        s3_client.create_bucket(Bucket=bucket_name)
        logger.info(f"Bucket {bucket_name} created successfully.")
    else:
        logger.info(f"Bucket {bucket_name} already exists.")


def main():
    ticker = "TSLA"
    logger.info(f"Starting data collection for {ticker}...")
    s3_client = create_s3_client()
    create_bucket_if_not_exists(s3_client, ticker.lower())
    if check_highest_date_in_s3(ticker, s3_client) < datetime.now().strftime("%Y%m%d"):
        raw_data = fetch_last_2_years(ticker)
        logger.info(f"Fetched {len(raw_data)} data points for {ticker}.")
        logger.info("Processing stock data...")
        processed_data = process_stock_data(raw_data)

        final_data, old_file_key = merge_raw_data(ticker, s3_client, processed_data)
        final_data.to_csv(f"data/{ticker}_historical_data.csv")
        logger.info(f"Processed and saved historical data for {ticker}.")
        upload_to_s3(
            s3_client,
            f"{ticker.lower()}",
            f"raw_{get_lowest_date_in_s3(ticker, s3_client)}_{datetime.now().strftime('%Y%m%d')}.csv",
            final_data,
        )
        store_old_data(s3_client, ticker, old_file_key)
    else:
        logger.info(
            f"Data for {ticker} is already up to date in S3. No new data fetched."
        )


if __name__ == "__main__":
    main()
