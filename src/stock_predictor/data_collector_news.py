from dotenv import load_dotenv
import os
from datetime import datetime
import requests
import pandas as pd
from transformers import pipeline
from stock_predictor.logger_config import logger
from tqdm import tqdm

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


def get_sentiment_analysis(
    df: pd.DataFrame, path_to_model: str = "./.models/finbert"
) -> pd.DataFrame:
    """
    Perform sentiment analysis on the headlines using a pre-trained model.

    Args:
        df (pd.DataFrame): DataFrame containing headlines and their published dates
        path_to_model (str): Path to the locally saved model

    Returns:
        pd.DataFrame: DataFrame with an additional column for sentiment labels
    """
    df["headline"] = df["headline"].fillna("").astype(str)
    texts = df["headline"].str[:512].tolist()
    sentiment_analyzer = pipeline(
        "sentiment-analysis", model=path_to_model, tokenizer=path_to_model, top_k=None
    )
    batch_size = 32
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Procesando en batch"):
        batch = texts[i : i + batch_size]
        batch_results = sentiment_analyzer(
            batch, batch_size=batch_size, return_all_scores=True
        )
        results.extend(batch_results)
    df["positive"] = 0.0
    df["neutral"] = 0.0
    df["negative"] = 0.0

    for idx, r in enumerate(results):
        for item in r:
            label = item["label"].lower()
            df.at[idx, label] = item["score"]
    df["sentiment"] = df["positive"] - df["negative"]
    return df


def download_model_locally(
    model_name: str = "ProsusAI/finbert", local_path: str = "./.models/finbert"
) -> None:
    """
    Download a pre-trained model from Hugging Face and save it locally.

    Args:
        model_name (str): The name of the model to download (e.g., "ProsusAI/finbert")
        local_path (str): The local directory path where the model should be saved
    """
    if os.path.exists(local_path):
        logger.info(f"Model already exists at {local_path}. Skipping download.")
        return
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)


def merge_prices_news(df_news: pd.DataFrame, df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the news DataFrame with the stock prices DataFrame based on the published date.

    Args:
        df_news (pd.DataFrame): DataFrame containing news headlines and their published dates
        df_prices (pd.DataFrame): DataFrame containing stock prices with a datetime index

    Returns:
        pd.DataFrame: Merged DataFrame containing both news and stock price information
    """
    df_news["published_utc"] = pd.to_datetime(df_news["published_utc"]).dt.tz_localize(
        None
    )
    df_news = df_news[~df_news["published_utc"].isna()]
    df_news = df_news.groupby("published_utc", as_index=False).agg(
        {
            "headline": lambda x: " | ".join(x),
            "positive": "mean",
            "neutral": "mean",
            "negative": "mean",
            "sentiment": "mean",
        }
    )
    if "t" not in df_prices.columns:
        df_prices = df_prices.reset_index()
    df_prices["t"] = pd.to_datetime(df_prices["t"])
    df_prices = df_prices.drop_duplicates(subset=["t"], keep="last")
    merged_df = pd.merge_asof(
        df_prices.sort_values("t"),
        df_news.sort_values("published_utc"),
        left_on="t",
        right_on="published_utc",
        direction="backward",
    )
    return merged_df


if __name__ == "__main__":
    download_model_locally()
    news_data = fetch_news_data("AAPL", datetime.now())
    df = extract_headlines(news_data)
    df = clean_data(df)
    df = get_dataframe(df)
    df = get_sentiment_analysis(df)
    df.to_csv("./data/aapl_news_test.csv", index=False, quoting=1)
    df = merge_prices_news(
        df, pd.read_csv("./data/AAPL_historical_data.csv", parse_dates=["t"])
    )
    df.to_csv("./data/aapl_news_test.csv", index=False, quoting=1)
