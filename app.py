import os
import pandas as pd
import numpy as np
import yfinance as yf
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from flask import Flask, render_template, send_from_directory

# Initialize Flask app
app = Flask(__name__)

# Ensure the 'static' directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Define the stock symbol and the date range for data
stock_symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2024-01-01"

# Example financial news headlines
news_headlines = [
    "Apple reports record earnings for Q4 2023",
    "Apple stock drops due to supply chain issues",
    "Apple announces new iPhone model in record time",
    "Apple faces antitrust lawsuit over App Store policies",
    "Analysts optimistic about Apple's growth in 2024"
]

# Initialize sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Function to calculate sentiment scores for headlines
def get_sentiment_score(headlines):
    sentiment_scores = []
    for headline in headlines:
        sentiment = sia.polarity_scores(headline)
        sentiment_scores.append(sentiment['compound'])
    return sentiment_scores

# Flask route for the homepage
@app.route('/')
def index():
    # Download stock price data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Flatten the multi-level columns
    stock_data.columns = ['_'.join(col) for col in stock_data.columns]

    # Get sentiment scores for news headlines
    sentiment_scores = get_sentiment_score(news_headlines)

    # Create a DataFrame for sentiment data
    sentiment_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-11-01', periods=5, freq='D'),
        'Sentiment': sentiment_scores
    })

    # Resample stock data to daily frequency and merge with sentiment data
    stock_data_resampled = stock_data['Close_AAPL'].resample('D').last()
    sentiment_data['Date'] = sentiment_data['Date'].dt.tz_localize(None)
    merged_data = pd.merge(stock_data_resampled, sentiment_data, on='Date', how='inner')

    # Check for NaN values in the merged data
    print(merged_data.isnull().sum())

    # Drop rows with NaN values
    merged_data = merged_data.dropna()

    # Prepare features (X) and target (y)
    X = merged_data[['Sentiment']]
    y = merged_data['Close_AAPL']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate the Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)

    # Plot Actual vs Predicted Prices
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Stock Prices')
    plt.savefig(os.path.join('static', 'actual_vs_predicted.png'))

    # Fit an ARIMA model
    model_arima = ARIMA(stock_data_resampled, order=(5, 1, 0))
    model_arima_fit = model_arima.fit()

    # Make ARIMA predictions
    forecast = model_arima_fit.forecast(steps=10)

    # Plot ARIMA Forecast
    plt.plot(stock_data_resampled.index, stock_data_resampled, label='Actual Stock Prices')
    plt.plot(pd.date_range(stock_data_resampled.index[-1], periods=10, freq='D'), forecast, label='Forecasted Prices', linestyle='--')
    plt.legend()
    plt.title(f'ARIMA Forecast for {stock_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.savefig(os.path.join('static', 'arima_forecast.png'))

    # Render the HTML page with dynamic data
    return render_template('index.html', sentiment_scores=zip(news_headlines, sentiment_scores), mae=mae)

# Route to serve static files (like images)
@app.route('/static/<path:filename>')
def send_file(filename):
    return send_from_directory('static', filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
