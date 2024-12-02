import pandas as pd
import numpy as np
import yfinance as yf
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # Import ARIMA

# Define the stock symbol and the date range for data
stock_symbol = "AAPL"  # Apple Inc. as an example
start_date = "2020-01-01"
end_date = "2024-01-01"

# Download stock price data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Print the structure of the stock data
print("Stock Data Columns:")
print(stock_data.columns)

# Flatten the multi-level columns from Yahoo Finance data
stock_data.columns = ['_'.join(col) for col in stock_data.columns]

# Print the first few rows after flattening columns
print("Stock Data (after flattening columns):")
print(stock_data.head())

# Example financial news headlines (simulated)
news_headlines = [
    "Apple reports record earnings for Q4 2023",
    "Apple stock drops due to supply chain issues",
    "Apple announces new iPhone model in record time",
    "Apple faces antitrust lawsuit over App Store policies",
    "Analysts optimistic about Apple's growth in 2024"
]

# Initialize sentiment analyzer (using VADER)
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Function to calculate sentiment scores for headlines
def get_sentiment_score(headlines):
    sentiment_scores = []
    for headline in headlines:
        sentiment = sia.polarity_scores(headline)
        sentiment_scores.append(sentiment['compound'])  # Use compound score for overall sentiment
    return sentiment_scores

# Get sentiment scores
sentiment_scores = get_sentiment_score(news_headlines)
print("Sentiment Scores:", sentiment_scores)

# Create a DataFrame for sentiment scores for a specific date range (same as stock data)
sentiment_data = pd.DataFrame({
    'Date': pd.date_range(start='2023-11-01', periods=5, freq='D'),
    'Sentiment': sentiment_scores
})

# Merge the sentiment data with stock price data (resample stock data to daily frequency)
stock_data_resampled = stock_data['Close_AAPL'].resample('D').last()  # Closing price for each day

# Convert the 'Date' column in sentiment_data to naive datetime (no timezone)
sentiment_data['Date'] = sentiment_data['Date'].dt.tz_localize(None)

# Merge sentiment data with stock price data based on date
merged_data = pd.merge(stock_data_resampled, sentiment_data, on='Date', how='inner')

# Print the merged data to inspect the columns
print("Merged DataFrame:")
print(merged_data.head())

# Drop rows with NaN values in the target column ('Close_AAPL')
merged_data.dropna(subset=['Close_AAPL'], inplace=True)

# Ensure there are no NaN values in the features (X) or target (y)
print(f"NaN values in the target column: {merged_data['Close_AAPL'].isna().sum()}")  # Should be 0
print(f"NaN values in the features: {merged_data['Sentiment'].isna().sum()}")  # Should be 0

# Prepare features (X) and target (y) for training
X = merged_data[['Sentiment']]
y = merged_data['Close_AAPL']  # Use 'Close_AAPL' for the stock price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the stock prices using the test set
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error of the predictions
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Plot the predicted vs actual stock prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Stock Prices')
plt.show()

# ARIMA Model (Forecasting future stock prices)
# Fit an ARIMA model (you may need to adjust the order depending on your data)
model_arima = ARIMA(stock_data_resampled, order=(5, 1, 0))  # ARIMA is now imported
model_arima_fit = model_arima.fit()

# Make predictions for the next 10 days
forecast = model_arima_fit.forecast(steps=10)

# Plot the forecasted stock prices along with the historical data
plt.figure(figsize=(10, 6))
plt.plot(stock_data_resampled.index, stock_data_resampled, label='Actual Stock Prices')
plt.plot(pd.date_range(stock_data_resampled.index[-1], periods=10, freq='D'), forecast, label='Forecasted Prices', linestyle='--')
plt.legend()
plt.title(f'ARIMA Forecast for {stock_symbol}')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

