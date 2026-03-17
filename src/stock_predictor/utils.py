import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import boto3
from stock_predictor.logger_config import logger

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")


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


def get_latest_data_s3(s3_client, bucket_name: str):
    """
    Get the latest raw data file from an S3 bucket and return it as a DataFrame.

    Args:
        s3_client (boto3.client): S3 client.
        bucket_name (str): Name of the S3 bucket.
    Returns:
        pd.DataFrame: DataFrame containing the latest raw data from S3.
    """
    logger.info(f"Getting latest data from S3 bucket {bucket_name}...")
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    content = response.get("Contents", [])
    if content == []:
        logger.info("No files found in S3 bucket. Returning empty DataFrame.")
        return pd.DataFrame()
    latest_file = [obj["Key"] for obj in content if obj["Key"].startswith("raw")]
    obj = s3_client.get_object(Bucket=bucket_name, Key=latest_file[0])
    df = pd.read_csv(obj["Body"])
    logger.info(f"Latest data file {latest_file} loaded successfully.")
    return df
