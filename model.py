import os
import pandas as pd
import numpy as np
import ta  # Technical indicators library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from data_fetch import get_stock_data, get_earnings_data, get_news_sentiment, get_interest_rate, get_eps

# Ensure models folder exists
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_model_path(ticker, future_steps):
    """Generate file path for the model based on prediction interval."""
    return os.path.join(MODEL_DIR, f"{ticker}_{future_steps}step_model.pkl")

def add_technical_indicators(df):
    """Add technical indicators to the dataframe."""
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])

    # Fill NaN values created by indicators
    df.bfill(inplace=True)

    return df

def add_fundamental_indicators(df, ticker):
    """Add fundamental indicators to the dataframe."""
    
    # Check if df is empty before proceeding
    if df.empty:
        print(f"Warning: No stock data found for {ticker}. Returning empty DataFrame.")
        return df
    
    # Example fundamental data: Earnings Per Share (EPS)
    eps = get_eps(ticker)  # Implement get_eps to fetch the EPS from Yahoo Finance or another API
    
    # Check if EPS data is fetched successfully
    if eps is None:
        print(f"Warning: Could not fetch EPS data for {ticker}. Setting to NaN.")
        eps = np.nan  # Set to NaN if EPS data is missing
    else:
        df['EPS'] = eps
    
    # Fetch the earnings data using get_earnings_data function
    earnings_data = get_earnings_data(ticker)
    if earnings_data and 'EPS' in earnings_data:
        latest_earnings = earnings_data['EPS'] if isinstance(earnings_data['EPS'], (float, int)) else np.nan
        df['Earnings'] = latest_earnings
    else:
        print(f"Warning: Could not fetch earnings data for {ticker}. Setting to NaN.")
        latest_earnings = np.nan  # Set to NaN if no data is available or 'EPS' is not in the data
        df['Earnings'] = np.nan  # If earnings data is missing, set to NaN

    # Fetch the news sentiment data using get_news_sentiment function
    news_sentiment = get_news_sentiment(ticker)
    if news_sentiment is None:
        print(f"Warning: Could not fetch news sentiment for {ticker}. Setting to NaN.")
        df['News_Sentiment'] = np.nan  # If news sentiment is missing, set to NaN
    else:
        df['News_Sentiment'] = news_sentiment
    
    # Fetch the interest rate data using get_interest_rate function
    interest_rate = get_interest_rate()
    if interest_rate is None:
        print(f"Warning: Could not fetch interest rate data. Setting to NaN.")
        df['Interest_Rate'] = np.nan  # If interest rate data is missing, set to NaN
    else:
        df['Interest_Rate'] = interest_rate

    return df

def train_model(ticker, future_steps=1):
    """Train a Linear Regression model to predict stock prices at different time intervals."""
    df = get_stock_data(ticker)
    df = add_technical_indicators(df)
    df = add_fundamental_indicators(df, ticker)

    # Create the target variable (predict `future_steps` ahead)
    df[f'Close_{future_steps}steps'] = df['Close'].shift(-future_steps)
    df.dropna(inplace=True)

    # Selecting Features
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'EMA_10', 'RSI', 'MACD', 'MACD_Signal', 'EPS', 'News_Sentiment', 'Interest_Rate']
    
    X = df[FEATURES]
    y = df[f'Close_{future_steps}steps']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    model_path = get_model_path(ticker, future_steps)
    joblib.dump((model, FEATURES), model_path)  # Save model with features list
    print(f"Model saved: {model_path}")

def get_feature_importance(model, FEATURES):
    """Get feature importance from the trained model."""
    importance = model.coef_  # Get coefficients (importance of features)
    feature_importance = dict(zip(FEATURES, importance))
    return feature_importance

def generate_explanation(feature_importance, current_price, predicted_price_change):
    """Generate a natural, conversational explanation of the prediction using feature importance."""
    explanation = []

    for feature, importance in feature_importance.items():
        if importance > 0:
            explanation.append(f"An increase in {feature} generally causes the price to go up. (Coefficient: {importance:.4f})")
        else:
            explanation.append(f"An increase in {feature} generally causes the price to go down. (Coefficient: {importance:.4f})")
    
    explanation_str = " ".join(explanation)
    
    # Make the explanation feel more natural
    explanation_summary = f"The current price of the stock is ${current_price:.2f}. In summary, the model suggests that factors such as "
    positive_factors = [feature for feature, imp in feature_importance.items() if imp > 0]
    negative_factors = [feature for feature, imp in feature_importance.items() if imp < 0]

    if positive_factors:
        explanation_summary += "increases in " + ", ".join(positive_factors) + " contribute to a higher price."
    
    if negative_factors:
        explanation_summary += " On the other hand, increases in " + ", ".join(negative_factors) + " tend to push the price down."
    
    explanation_summary += f" Based on these factors, the model predicts a price change of ${predicted_price_change:.2f}."

    return explanation_summary

def predict_price(ticker, future_steps=1):
    """Load a trained model and predict the future stock price with explanation."""
    model_path = get_model_path(ticker, future_steps)

    if not os.path.exists(model_path):
        print(f"Model for {ticker} ({future_steps} steps) not found. Training a new model...")
        train_model(ticker, future_steps)

    model, FEATURES = joblib.load(model_path)  # Load model with feature names
    df = get_stock_data(ticker)
    df = add_technical_indicators(df)
    df = add_fundamental_indicators(df, ticker)
    
    # Ensure we use only the features the model was trained with
    latest_data = df[FEATURES].iloc[-1].values.reshape(1, -1)

    # Get the current price (latest closing price)
    current_price = df['Close'].iloc[-1]

    # Make the prediction
    predicted_price = model.predict(latest_data)[0]

    # Calculate the predicted price change
    predicted_price_change = predicted_price - current_price
    
    # Get feature importance (coefficients)
    feature_importance = get_feature_importance(model, FEATURES)
    
    # Generate explanation based on the feature importance
    explanation = generate_explanation(feature_importance, current_price, predicted_price_change)
    
    return predicted_price, explanation

if __name__ == "__main__":
    train_model("AAPL", future_steps=1)  # Train for 1-day prediction
    train_model("AAPL", future_steps=2)  # Train for 2-day prediction

    predicted_price, explanation = predict_price("AAPL", future_steps=1)
    print(f"Predicted Price for AAPL (1-day ahead): ${predicted_price:.2f}")
    print("Explanation:")
    print(explanation)