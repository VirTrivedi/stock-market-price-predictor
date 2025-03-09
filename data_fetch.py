import yfinance as yf
import pandas as pd
from textblob import TextBlob
import requests
import os
import numpy as np

def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

def get_earnings_data(ticker):
    """Fetch earnings data for a given stock ticker."""
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.financials  # 'financials' returns the income statement

        # Check for Net Income or EPS in the income statement
        eps = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else np.nan
        return {'EPS': eps}

    except Exception as e:
        print(f"Error fetching earnings data for {ticker}: {e}")
        return None

def get_news_sentiment(ticker):
    """Fetch and analyze the sentiment of news articles related to the stock ticker."""
    api_key = os.getenv("NEWS_API_KEY")
    news_url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    try:
        response = requests.get(news_url)
        articles = response.json().get('articles', [])
        if not articles:
            print(f"No news articles found for {ticker}.")
            return np.nan
        
        # Combine the titles and descriptions of all articles
        news_text = " ".join([article['title'] + " " + article['description'] if article['title'] else "" for article in articles])
        
        if not news_text:
            print(f"No valid news text found for {ticker}.")
            return np.nan  # Return NaN if no valid text is found

        # Perform sentiment analysis on the combined news text
        sentiment = TextBlob(news_text).sentiment.polarity  # Sentiment polarity: -1 (negative) to 1 (positive)
        
        return sentiment
    except Exception as e:
        print(f"Error fetching news sentiment for {ticker}: {e}")
        return None

def get_interest_rate():
    """Fetch current interest rates (e.g., Federal Reserve rates)."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'api_key': os.getenv("FRED_API_KEY"),
        'series_id': 'FEDFUNDS',  # FRED code for the Federal Funds Rate (interest rate)
        'frequency': 'm',  # Monthly data
        'file_type': 'json'
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        interest_rate = data['observations'][-1]['value']  # Get the most recent value
        if interest_rate is None:
            print("Could not fetch interest rate.")
            return None
        return float(interest_rate)
    except Exception as e:
        print(f"Error fetching interest rate: {e}")
        return None

def get_eps(ticker):
    """Fetch Earnings Per Share (EPS) for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        eps = stock.info.get('trailingEps', None)
        return eps
    except Exception as e:
        print(f"Error fetching EPS for {ticker}: {e}")
        return None

if __name__ == "__main__":
    df = get_stock_data("AAPL")
    print(df.head())