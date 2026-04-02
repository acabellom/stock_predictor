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
from stock_predictor.data_collector_news import (
    fetch_news_data,
    extract_headlines,
    clean_data,
    get_dataframe,
    get_sentiment_analysis,
    merge_prices_news,
    download_model_locally,
)
from datetime import datetime


@task
def fetch_news_data_task(ticker, date):
    return fetch_news_data(ticker, date)


@task
def extract_headlines_task(news_data):
    return extract_headlines(news_data)


@task
def clean_data_task(df):
    return clean_data(df)


@task
def get_dataframe_task(df):
    return get_dataframe(df)


@task
def get_sentiment_analysis_task(df):
    return get_sentiment_analysis(df)


@task
def merge_prices_news_task(df, df_prices):
    return merge_prices_news(df, df_prices)


@task
def download_model_locally_task():
    download_model_locally()


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
def data_gathering_flow(tickers: list = ["AAPL", "TSLA"]):
    """
    Prefect flow to gather stock price and news data for a given ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL")
    """
    download_model_locally_task.submit()
    for ticker in tickers:
        logger.info(f"Starting data collection for {ticker}...")
        s3_client = create_s3_client_task()
        create_bucket_task(s3_client, f"{ticker.lower()}")
        if check_data_up_to_date_task(f"{ticker.lower()}", s3_client):
            processed_data_price = fetch_and_process_data_task.submit(ticker)
            processed_data_news = fetch_news_data_task.submit(ticker, datetime.now())
            df_headlines = extract_headlines_task(processed_data_news)
            df_cleaned = clean_data_task(df_headlines)
            df_dataframe = get_dataframe_task(df_cleaned)
            df_sentiment = get_sentiment_analysis_task(df_dataframe)
            df_merged = merge_prices_news_task(
                df_sentiment, processed_data_price.result()
            )
            final_data, old_file_key = merge_raw_data_task(
                f"{ticker.lower()}", s3_client, df_merged
            )
            logger.info(f"Processed and saved historical data for {ticker}.")
            upload_to_s3_task(s3_client, f"{ticker.lower()}", final_data)
            store_old_data_task(s3_client, f"{ticker.lower()}", old_file_key)
        else:
            logger.info(
                f"Data for {ticker} is already up to date in S3. No new data fetched."
            )


TOP50_TICKERS = [
    # Big Tech (muy importantes)
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "META",
    "AMZN",
    # Semiconductores (alta señal)
    "AMD",
    "AVGO",
    "QCOM",
    "INTC",
    # Finanzas (estabilidad + contexto macro)
    "JPM",
    "BAC",
    "WFC",
    "GS-bucket",
    "MS-bucket",
    # Energía (movimientos fuertes)
    "XOM",
    "CVX",
    "COP",
    "SLB",
    # Consumo (defensivos + tendencia)
    "WMT",
    "COST",
    "PG-bucket",
    "KO-bucket",
    "PEP",
    "MCD",
    "NKE",
    "SBUX",
    # Salud (menos ruido extremo)
    "JNJ",
    "PFE",
    "UNH",
    "ABBV",
    "MRK",
    # Industriales (macro-driven)
    "CAT",
    "BA-bucket",
    "GE-bucket",
    "HON",
    "UPS",
    # Tech / software adicional
    "CRM",
    "ORCL",
    "ADBE",
    "CSCO",
    "NOW",
    # Otros con buena señal
    "TSLA",
    "NFLX",
    "DIS",
    "PYPL",
    "SPGI",
    "V-bucket",
    "MA-bucket",
]

if __name__ == "__main__":
    data_gathering_flow(TOP50_TICKERS)
