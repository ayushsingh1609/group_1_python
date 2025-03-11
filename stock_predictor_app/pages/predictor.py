import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned stock data
df = pd.read_csv('cleaned_stock_data.csv')

# Get top 10 companies based on trading volume
top_10_companies = df.groupby('ticker')['volume'].sum().nlargest(10).index.tolist()

# Create a mapping of tickers to company names
company_mapping = df[['ticker', 'company name']].drop_duplicates().set_index('ticker').to_dict()['company name']

# Streamlit UI
st.set_page_config(page_title="Stock Trend Predictor", page_icon="📈", layout="wide")

st.title("📊 Stock Trend Predictor")
st.write("Select a company to predict whether its stock price will go **Up 📈 or Down 📉** the next trading day.")

# Reverse mapping to select by company name
ticker_to_company = {v: k for k, v in company_mapping.items() if k in top_10_companies}

# User selects a company name
company_name = st.selectbox("Choose a company:", list(ticker_to_company.keys()))

if company_name:
    ticker = ticker_to_company[company_name]
    model_filename = f"models/{ticker}_trend_model.pkl"

    if os.path.exists(model_filename):
        # Load Pre-Trained Model
        best_model = joblib.load(model_filename)
        
        # Get latest data for prediction
        df_company = df[df['ticker'] == ticker].copy()
        df_company['date'] = pd.to_datetime(df_company['date'])
        df_company = df_company.sort_values(by='date')

        # Feature Engineering (Same as Training)
        df_company['daily_return'] = df_company['close'].pct_change()
        df_company['volatility'] = df_company['close'].rolling(7).std()
        df_company['ma7'] = df_company['close'].rolling(7).mean()
        df_company['ma30'] = df_company['close'].rolling(30).mean()
        
        # Create target variable trend
        df_company['trend'] = (df_company['close'].shift(-1) > df_company['close']).astype(int)
        
        # Drop NaN values
        df_company.dropna(inplace=True)

        # Prepare Features
        features = ['close', 'daily_return', 'volatility', 'ma7', 'ma30']
        
        # Evaluate model accuracy
        y_test = df_company['trend'][-len(df_company)//5:]
        y_pred = best_model.predict(df_company[features][-len(df_company)//5:])
        accuracy = accuracy_score(y_test, y_pred) * 100
        st.write(f"✅ Model Accuracy for {company_name} ({ticker}): {accuracy:.2f} %")
        
        latest_data = df_company[features].iloc[[-1]]  # Get latest available day

        # Predict Next Day's Trend
        prediction = best_model.predict(latest_data)

        # Show Prediction
        trend_prediction = "Up 📈" if prediction[0] == 1 else "Down 📉"
        st.write(f"📊 Prediction for {company_name} on the next trading day: {trend_prediction}")

    else:
        st.write(f"❌ No pre-trained model found for {company_name} ({ticker}). Please train the model first.")
