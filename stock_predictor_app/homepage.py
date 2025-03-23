import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import time
import requests
import seaborn as sns
import os
from dotenv import load_dotenv

load_dotenv()

api_key_news = os.getenv("API_KEY_NEWS")


st.set_page_config(
    page_title="TradeVision AI",
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
st.sidebar.markdown("Select a company to explore predictions:")
st.sidebar.markdown("- 🧠 [Apple (AAPL)](apple)\n- 💻 [Microsoft (MSFT)](microsoft)\n- 🔍 [Google (GOOG)](google)\n- 🛒 [Walmart (WMT)](walmart)\n- 🎬 [Netflix (NFLX)](netflix)")

# Main Page Content
st.title("Welcome to the TradeVision AI 📊")

st.markdown("""
### Overview
This system leverages machine learning models trained on historical stock data to predict both the trend (up or down) and future price of selected companies. It also applies a rule-based trading strategy to simulate real-world decision-making and optimize profitability based on these predictions.
### Core Features
- 📊 **Company-Specific Dashboards** – Explore dedicated pages for Apple, Google, Microsoft, Walmart, and Netflix.  
- 🧠 **ML-Based Predictions** – Get next-day forecasts for stock **trend direction** and **closing price** using trained machine learning models.  
- 💼 **Smart Trading Strategy** – Apply a rule-based strategy that simulates **buy/sell decisions** to optimize returns.  




### Development Team
- Ayush Singh
- Silvana Cortes
- Jorge Hiroshi
- Chevin Tochkov
- Christyana Kane

### Project Goal
💡 Deliver AI-driven insights to assist retail traders and finance enthusiasts in making more informed decisions.
""")

# --- Accordion Style News Section ---
st.markdown("""
    <style>
    .compact-news h4 {
        font-size: 16px !important;
        margin-bottom: 0.2rem;
    }
    .compact-news p {
        font-size: 13px;
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #ccc;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-top: 30px; margin-bottom: 10px;">
    <h3 style="color: #90caf9; font-size: 30px;">🗞️ Market Headlines (Top 5)</h3>
</div>
""", unsafe_allow_html=True)

def fetch_news():
    try:
        response = requests.get(f"https://newsapi.org/v2/top-headlines?category=business&apiKey={api_key_news}")
        news_data = response.json()
        if "articles" in news_data:
            for article in news_data["articles"][:5]:
                title = article['title']
                desc = article.get('description', '')
                url = article['url']

                with st.expander(f"📰 {title}"):
                    st.write(desc)
                    st.markdown(f"[Read full article ➜]({url})")
        else:
            st.info("No news available at the moment.")
    except Exception as e:
        st.error(f"Error fetching news: {e}")

fetch_news()





import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import streamlit as st


# Download recent S&P 500 data
ticker_symbol = "^GSPC"
end_date = datetime.today()
start_date = end_date - timedelta(days=10)

sp500_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1d", progress=False)

if not sp500_data.empty:
    latest_close = float(sp500_data['Close'].iloc[-1])
    previous_close = float(sp500_data['Close'].iloc[-2])
    change = latest_close - previous_close

    percent_change = (change / previous_close) * 100
    arrow = "🟢" if change > 0 else "🔴"

    color = "#00FF7F" if change > 0 else "#FF4C4C"

    st.markdown(f"""
        <h3 style="color: #90caf9; margin-top: 30px; font-size: 28px; margin-bottom: 10px;">📊 Market Summary: S&P 500</h3>

        <h4 style="color: #00ffc8; font-size: 22px; margin: 5px 0 0px;">Index: S&P 500 (📈 ^GSPC)</h4>

        <p style="font-size: 17px; line-height: 1.6; color: #ffffff;">
            <strong>Latest Close:</strong> ${latest_close:.2f} <br>
            <strong>Previous Close:</strong> ${previous_close:.2f} <br>
            <strong>Change:</strong> <span style="color: {color}; font-weight: bold;">{arrow} ${change:.2f} ({percent_change:.2f}%)</span>
        </p>
        """, unsafe_allow_html=True)


st.markdown(f"""<h3 style="color: #90caf9; margin-top: 30px; font-size: 28px; margin-bottom: 10px;"> 📈 Stock Trends (Normalized % Change Over 7 Days)</h3>""",unsafe_allow_html=True)


top_companies = ["AAPL", "MSFT", "GOOGL", "WMT", "NFLX"]
fig, ax = plt.subplots(figsize=(12, 6))
sns.set_style("darkgrid")

for ticker in top_companies:
    try:
        stock_data = yf.download(ticker, period="7d", interval="1h", progress=False)
        time.sleep(0.5)

        if not stock_data.empty:
            normalized = stock_data['Close'] / stock_data['Close'].iloc[0] * 100
            ax.plot(stock_data.index, normalized, label=ticker, linewidth=2)
        else:
            st.warning(f"⚠️ No data for {ticker}")
    except Exception as e:
        st.error(f"❌ Error retrieving {ticker}: {e}")

# Style it professionally
ax.set_title("7-Day Normalized Stock Performance (%)", fontsize=16, color="#00ffc8")
ax.set_xlabel("Date", fontsize=12, color="white")
ax.set_ylabel("Relative % Change", fontsize=12, color="white")
ax.tick_params(colors='gray')
ax.legend(facecolor="#111", edgecolor="#333", labelcolor="white", fontsize=10)
ax.set_facecolor("#000")
fig.patch.set_facecolor("#000")

# Blur gridlines
ax.grid(color='gray', linestyle='--', linewidth=0.3, alpha=0.1)

st.pyplot(fig)
