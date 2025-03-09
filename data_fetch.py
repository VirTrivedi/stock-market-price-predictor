import yfinance as yf
import pandas as pd

def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

if __name__ == "__main__":
    df = get_stock_data("AAPL")  # Example: Apple stock data
    print(df.head())