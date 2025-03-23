import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_absolute_error

# Define model directory
MODEL_DIR = "saved_models"

# Load dataset
data_path = "filtered_5_companies_stock_data.csv"
df = pd.read_csv(data_path)

# Convert date column to datetime if it exists
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Load available companies (tickers)
def get_available_tickers():
    return df['ticker'].unique()

# Load models
def load_models(ticker):
    trend_model_path = os.path.join(MODEL_DIR, f"{ticker}_trend_model.pkl")
    price_model_path = os.path.join(MODEL_DIR, f"{ticker}_price_model.pkl")
    
    trend_model = joblib.load(trend_model_path)
    price_model = joblib.load(price_model_path)
    
    return trend_model, price_model

# Predict and evaluate models
def predict_and_evaluate(ticker):
    trend_model, price_model = load_models(ticker)
    df_ticker = df[df['ticker'] == ticker]
    features = ['open', 'high', 'low', 'close', 'adj. close', 'volume', 'daily_return', 'volatility', '5_day_ma', '10_day_ma']
    
    X = df_ticker[features]
    y_trend = df_ticker['trend']
    y_price = df_ticker['next_day_close']
    
    trend_preds = trend_model.predict(X)
    price_preds = price_model.predict(X)
    
    trend_accuracy = accuracy_score(y_trend, trend_preds)
    price_mae = mean_absolute_error(y_price, price_preds)
    
    latest_trend_prediction = "Up" if trend_preds[-1] == 1 else "Down"
    latest_price_prediction = price_preds[-1]
    
    return trend_accuracy, price_mae, latest_trend_prediction, latest_price_prediction, df_ticker['date'].tail(20), y_price.tail(20), price_preds[-20:]

# Streamlit UI
st.title("Stock Trend and Price Predictor")

# Dropdown to select company
tickers = get_available_tickers()
ticker_selected = st.selectbox("Select a company:", tickers)

# Predict and evaluate models
if st.button("Show Prediction and Accuracy"):
    trend_acc, price_mae, trend_pred, price_pred, dates, actual_prices, predicted_prices = predict_and_evaluate(ticker_selected)
    
    st.subheader("Prediction and Model Performance")
    st.write(f"**Trend Prediction Accuracy:** {trend_acc*100:.2f}%")
    st.write(f"**Stock Price Prediction MAE:** {price_mae:.4f}")
    st.write(f"**Latest Trend Prediction:** {trend_pred}")
    st.write(f"**Latest Predicted Next Day Close Price:** {price_pred:.4f}")
    
    # Plot last 20 actual vs predicted stock prices with Seaborn, using dates for x-axis
    st.subheader("Stock Price Prediction vs Actual Values (Last 20 Days)")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_plot = pd.DataFrame({
        'Date': dates,
        'Actual Prices': actual_prices.values,
        'Predicted Prices': predicted_prices
    })
    
    sns.lineplot(x='Date', y='Actual Prices', data=df_plot, label='Actual Prices', marker='o', linestyle='-')
    sns.lineplot(x='Date', y='Predicted Prices', data=df_plot, label='Predicted Prices', marker='x', linestyle='--')
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"Actual vs Predicted Stock Prices for {ticker_selected} (Last 20 Days)")
    ax.legend()
    
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.success("Prediction and model evaluation completed!")
