from prefect import flow, task
from stock_predictor.data_collector_prices import (
    fetch_last_2_years,
    process_stock_data,
    merge_raw_data,
)
from stock_predictor.logger_config import logger
from stock_predictor.utils import (
    create_bucket_if_not_exists,
    check_highest_date_in_s3,
    store_old_data,
    create_s3_client,
    upload_to_s3,
    get_lowest_date_in_s3,
)
from datetime import datetime


@task
def create_s3_client_task():
    return create_s3_client()


@task
def create_bucket_task(s3_client, ticker):
    create_bucket_if_not_exists(s3_client, ticker.lower())


@task
def check_data_up_to_date_task(ticker, s3_client):
    return check_highest_date_in_s3(ticker, s3_client) < datetime.now().strftime(
        "%Y%m%d"
    )


@task
def fetch_and_process_data_task(ticker):
    raw_data = fetch_last_2_years(ticker)
    logger.info(f"Fetched {len(raw_data)} data points for {ticker}.")
    logger.info("Processing stock data...")
    return process_stock_data(raw_data)


@task
def merge_raw_data_task(ticker, s3_client, processed_data):
    return merge_raw_data(ticker, s3_client, processed_data)


@task
def upload_to_s3_task(s3_client, ticker, final_data):
    upload_to_s3(
        s3_client,
        f"{ticker.lower()}",
        f"raw_{get_lowest_date_in_s3(ticker, s3_client)}_{datetime.now().strftime('%Y%m%d')}.csv",
        final_data,
    )


@task
def store_old_data_task(s3_client, ticker, old_file_key):
    store_old_data(s3_client, ticker, old_file_key)


@flow(name="Data Gathering Flow")
def data_gathering_flow(ticker: str):
    """
    Prefect flow to gather stock price and news data for a given ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL")
    """

    logger.info(f"Starting data collection for {ticker}...")
    s3_client = create_s3_client_task()
    create_bucket_task(s3_client, ticker)
    if check_data_up_to_date_task(ticker, s3_client):
        processed_data = fetch_and_process_data_task(ticker)

        final_data, old_file_key = merge_raw_data_task(
            ticker, s3_client, processed_data
        )
        logger.info(f"Processed and saved historical data for {ticker}.")
        upload_to_s3_task(s3_client, ticker, final_data)
        store_old_data_task(s3_client, ticker, old_file_key)
    else:
        logger.info(
            f"Data for {ticker} is already up to date in S3. No new data fetched."
        )


if __name__ == "__main__":
    data_gathering_flow("TSLA")
