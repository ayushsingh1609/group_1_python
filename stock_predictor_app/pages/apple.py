import streamlit as st
import pandas as pd
from datetime import datetime
import joblib
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from util import PySimFin, ETL, TradingStrategy
import plotly.graph_objects as go
import numpy as np

load_dotenv()

API_KEY = os.getenv("API_KEY") 

simfin = PySimFin(API_KEY)

st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Hide Streamlit's default page navigation menu
hide_nav_style = """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_nav_style, unsafe_allow_html=True)

# Custom dark theme with gray sidebar
st.markdown("""
    <style>
    /* Global app background */
    html, body, [class*="stApp"] {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #111111;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Accent headers */
    h1, h2, h3, h4, h5, h6 {
        color: #90caf9;
    }

    /* Inputs */
    input, textarea, select {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: none;
        border-radius: 5px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #222222;
        color: #ffffff;
        border: 1px solid #444444;
        border-radius: 5px;
    }

    /* Hide scrollbar */
    ::-webkit-scrollbar {
        width: 0px;
    }
    </style>
""", unsafe_allow_html=True)



# --- Sidebar Navigation ---
st.sidebar.title("🚀 Go Live")
st.sidebar.markdown("📍 [🏠 Home](homepage)")
st.sidebar.markdown("Select a company to explore predictions:")
st.sidebar.markdown("- 🍎 [Apple (AAPL)](apple)\n- 💻 [Microsoft (MSFT)](microsoft)\n- 🔍 [Google (GOOG)](google)\n- 🛒 [Walmart (WMT)](walmart)\n- 🎬 [Netflix (NFLX)](netflix)")



st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Apple_Store_logo.svg/1024px-Apple_Store_logo.svg.png", width=120)
st.title("Apple (AAPL) - Stock Trend and Price Prediction")
st.write("Analyze and predict Apple's stock price movements with AI-powered insights.")

st.subheader("About Apple Inc.")
st.write(
    """
Apple Inc., headquartered in Cupertino, California, is a global technology leader known for its premium devices such as the iPhone, Mac, iPad, and wearables, as well as its services ecosystem including the App Store, iCloud, and Apple Music.

**CEO:** Tim Cook

Apple's stock (AAPL) is one of the most actively traded and valuable in the world. It is a core component of the S&P 500 and Nasdaq 100, and it was the first U.S. company to surpass a $3 trillion market cap. Known for consistent revenue growth, high profit margins, and massive cash reserves, Apple stock is often considered a safe, long-term investment.
    """
)

st.subheader("📰 Apple in the News (Top 3)")
news_links = [
    ("Apple announces new iPads and Macs at upcoming event", "https://www.cnbc.com/2024/10/31/apple-ipad-mac-event.html"),
    ("Apple stock climbs after strong Q4 earnings report", "https://www.marketwatch.com/story/apple-stock-q4-earnings-boost-2024"),
    ("Apple's AI strategy takes shape with new chip technology", "https://www.theverge.com/2024/11/01/apple-ai-chip-strategy")
]

for title, url in news_links:
    st.markdown(f"- [{title}]({url})")

# Get share prices & financials
st.sidebar.header(" Select Date Range")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"), format="YYYY-MM-DD")
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"), format="YYYY-MM-DD")

# Allow user to input custom trading capital
initial_cash = st.sidebar.number_input("Enter Initial Capital ($)", min_value=1000, value=10000, step=500)

# Convert dates to string format
ticker = "AAPL"
start_date = start_date.strftime("%Y-%m-%d")
end_date = end_date.strftime("%Y-%m-%d")

# Get share prices & financials
df_share_prices = simfin.get_share_prices(ticker, start_date, end_date)
df_share_prices = simfin.rename_columns(df_share_prices)
df_share_prices['ticker'] = ticker

if df_share_prices is None or df_share_prices.empty:
    st.error("❌ Unable to fetch stock prices or financial data. Please try again later.")
    st.stop()

etl = ETL(share_prices_df=df_share_prices, tickers=[ticker])
df_cleaned = etl.run_pipeline()
filtered_df = df_cleaned[(df_cleaned['date'] >= start_date) & (df_cleaned['date'] <= end_date)]

st.subheader("🔧 Price Movement")
candlestick_fig = go.Figure(data=[
    go.Candlestick(
        x=filtered_df['date'],
        open=filtered_df['open'],
        high=filtered_df['high'],
        low=filtered_df['low'],
        close=filtered_df['close'],
        name="Candlestick"
    )
])

candlestick_fig.update_layout(
    title=f"AAPL Candlestick Chart ({start_date} to {end_date})",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(candlestick_fig, use_container_width=True)

# Load models
# Get absolute path to saved_models
model_dir = os.path.join(os.path.dirname(__file__), "saved_models")

# Load models using the full path
trend_model = joblib.load(os.path.join(model_dir, "AAPL_trend_model.pkl"))
price_model = joblib.load(os.path.join(model_dir, "AAPL_price_model.pkl"))


features = ['open', 'high', 'low', 'close', 'adj. close', 'volume',
            'daily_return', 'volatility', '5_day_ma', '10_day_ma']

X = filtered_df[features]
y_trend = filtered_df['trend']
y_price = filtered_df['next_day_close']

trend_preds = trend_model.predict(X)
price_preds = price_model.predict(X)

filtered_df['trend_pred'] = trend_preds
filtered_df['price_pred'] = price_preds

trend_acc = accuracy_score(y_trend, trend_preds)
price_rmse = np.sqrt(mean_squared_error(y_price, price_preds))


trend_pred = "Up" if trend_preds[-1] == 1 else "Down"
price_pred = price_preds[-1]

prediction_date = filtered_df.iloc[-1]['date']
st.write(f"🗓️ **Model is predicting for the next day based on data from:** `{prediction_date}`")
st.write("📆 Last date in prediction data:", filtered_df['date'].max())

st.subheader("Prediction and Model Performance")
st.write(f"**Trend Prediction Accuracy:** {trend_acc * 100:.2f}%")
st.write(f"**Price Prediction RMSE (Root Mean Squared Error):** {price_rmse:.2f} — this tells you how far the model's predictions are from the actual prices, on average. Lower is better.")
st.write(f"**Latest Trend Prediction:** {trend_pred} — the model expects the stock to go {trend_pred.lower()} the next day.")
st.write(f"**Latest Predicted Next Day Close Price:** ${price_pred:.2f}")

# Apply Trading Strategy
st.subheader("💼 Strategy Simulation: Rule-Based Trading")
strategy = TradingStrategy(initial_cash=initial_cash)
filtered_df = strategy.apply_strategy(filtered_df)
summary = strategy.summary()

st.write("**Final Portfolio Summary:**")
st.json(summary)

st.write("**Trade Log (last 10 actions):**")
st.dataframe(strategy.get_trade_log().tail(10))