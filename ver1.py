import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import time
import requests

# Streamlit Page Configuration
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# Custom CSS for Navigation Bar
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #1E1E1E;
        }
        .sidebar .sidebar-content .block-container {
            color: white;
        }
        .sidebar .sidebar-content .stRadio label {
            color: white;
            font-size: 16px;
        }
        .sidebar .sidebar-content .stRadio div[role="radiogroup"] label:hover {
            background-color: #444;
            border-radius: 10px;
            padding: 5px;
        }
        .sidebar .sidebar-content .stRadio div[role="radiogroup"] label {
            transition: all 0.3s ease;
            padding: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Navigation Bar
st.sidebar.title("🔍 Navigation")
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Stock Predictor"], index=0)

if page == "🏠 Home":
    st.title("Welcome to the Stock Price Movement Predictor 📊")

    st.markdown("""
    ### Overview
    This trading system allows users to predict whether a selected company's stock price will go **Up** or **Down** based on historical data. 
    It provides insights into stock trends using machine learning models.

    ### Core Functionalities
    - 📌 **Stock Selection**: Choose a company from the available list.
    - 📈 **Price Prediction**: The system forecasts if the price will increase or decrease the next day.
    - 📊 **Visualization**: View historical trends and model predictions.

    ### About the Development Team
    **Team Members:**
    - Ayush Singh
    - Silvana Cortes
    - Jorge Hiroshi
    - Chievin Tochkov
    - Christyana Kane

    ### Purpose & Objectives
    💡 The goal of this system is to **empower traders and investors** by providing **data-driven predictions** that enhance decision-making.
    """)

    # Add a decorative separator
    st.markdown("---")

    # News Section
    st.markdown("## 📰 Latest Stock Market News")
    
    def fetch_news():
        try:
            response = requests.get("https://newsapi.org/v2/top-headlines?category=business&apiKey=47c9b568642a4da3aea0900ac8d141fd")
            news_data = response.json()
            
            if "articles" in news_data:
                for article in news_data["articles"][:5]:
                    st.markdown(f"### [{article['title']}]({article['url']})")
                    st.write(article["description"])
                    st.markdown("---")
            else:
                st.write("No news available at the moment.")
        except Exception as e:
            st.write(f"Error fetching news: {e}")

    fetch_news()

    # Live Stock Price Graph for Top 5 Companies
    top_companies = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    st.markdown("## 📈 Live Stock Price Trends")

    fig, ax = plt.subplots(figsize=(10, 5))

    for ticker in top_companies:
        try:
            stock_data = yf.download(ticker, period="7d", interval="1h")
            time.sleep(1)  # Prevent rate limiting
            
            if not stock_data.empty:
                ax.plot(stock_data.index, stock_data['Close'], label=ticker)
            else:
                st.write(f"⚠️ No data available for {ticker}.")
        except Exception as e:
            st.write(f"❌ Error retrieving data for {ticker}: {e}")

    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.set_title("Stock Prices Over the Last 7 Days")
    st.pyplot(fig)

elif page == "📊 Stock Predictor":
    st.switch_page("predictor.py")