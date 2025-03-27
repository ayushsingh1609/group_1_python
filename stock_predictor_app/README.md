Readme

# StockForecast

## Overview
StockForecast is a machine learning-powered web application that we developed for our Python 2 Group Assignmenet that predicts stock price movements and simulates trading strategies. The system is designed to provide retail investors and finance enthusiasts with informed insights derived from real-time financial data and AI predictions.

## Features
- Company-specific dashboards with detailed analysis
- Trend prediction (Up/Down) using classification models
- Next-day price prediction using regression models
- Interactive candlestick charts and normalized performance plots
- Rule-based trading strategy simulation with portfolio tracking

## Technologies Used
- Python
- Streamlit
- scikit-learn
- yFinance
- SimFin API
- Plotly & Matplotlib for visualizations
- Joblib for model serialization

## Web App Layout
The application is organized into the following pages:

1. **Homepage**
   - Introduction and project overview
   - Latest stock market news headlines (accordion format)
   - S&P 500 Market Summary
   - Normalized stock trend chart (7-day performance)

2. **Company Pages** (One page for each: Apple, Google, Microsoft, Walmart, Netflix)
   - About the company section
   - Latest news headlines
   - Candlestick chart
   - Model predictions for trend and price
   - Performance metrics (Accuracy, RMSE)
   - Simulated trading strategy and portfolio summary

## Data Sources
- SimFin API for historical stock prices and financial data
- NewsAPI for latest business news

## Folder Structure
```
stock_predictor_app/
│
├── pages/                     # Individual company pages
│   ├── apple.py
│   ├── google.py
│   └── ...
│
├── saved_models/             # Serialized ML models (one per company)
│
├── util.py                   # PySimFin API wrapper, ETL logic, trading strategy class
├── homepage.py               # Main landing page
├── requirements.txt          # App dependencies
└── .streamlit/config.toml    # Streamlit theme configuration
```

## How to Run Locally
1. Clone the repository:
```bash
git clone https://github.com/yourusername/group_1_python.git
cd group_1_python/stock_predictor_app
```
2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Add your `.env` file in the root with:
```
API_KEY=your_simfin_api_key
API_KEY_NEWS=your_newsapi_key
```
5. Launch the app:
```bash
streamlit run homepage.py
```

## Deployment
This app is deployed using [Streamlit Community Cloud](https://streamlit.io/cloud). Ensure that all required models, dependencies, and `.env` variables are configured properly before pushing to GitHub.

## Deployed web app URL
https://group1python-p8hevvy7wfygvx5hqrbtq5.streamlit.app

## Contributors
- Ayush Singh
- Silvana Cortes
- Chevin Tochkov
- Christyana Kane
- Jorge Hiroshi

## License
This project is for academic and educational purposes only.

