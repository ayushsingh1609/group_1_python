# %%
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

class PySimFin():
    BASE_URL = "https://backend.simfin.com/api/v3/"  # Correct Base API URL

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Authorization": f"{self.api_key}"}

    def _get(self, endpoint: str, params: dict = None):
        url = self.BASE_URL + endpoint
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None

    def get_sim_id(self, ticker: str):
        endpoint = "companies/general/verbose"
        params = {"ticker": ticker}
        data = self._get(endpoint, params)
        if data and isinstance(data, list) and len(data) > 0 and "id" in data[0]:
            return data[0]["id"]
        print(f"Error: Could not find SimFin ID for ticker {ticker}")
        return None

    def get_share_prices(self, ticker: str, start: str, end: str):
        sim_id = self.get_sim_id(ticker)
        if not sim_id:
            return None

        endpoint = f"companies/prices/compact"
        params = {"id": sim_id, "start": start, "end": end}
        data = self._get(endpoint, params)

        if not data or not isinstance(data, list) or "columns" not in data[0] or "data" not in data[0]:
            print(f"Error: Invalid response format for ticker {ticker}")
            return None

        columns = data[0]["columns"]
        records = data[0]["data"]

        if not records:
            print(f"Warning: No share price data available for ticker {ticker}")
            return None

        df = pd.DataFrame(records, columns=columns)
        return df

    def rename_columns(self, df):
        return df.rename(columns={
            "Opening Price": "open",
            "Highest Price": "high",
            "Lowest Price": "low",
            "Last Closing Price": "close",
            "Adjusted Closing Price": "Adj. Close",
            "Trading Volume": "Volume"
        })

# (ETL and TradingStrategy classes remain unchanged)

# ... Rest of code stays unchanged ...
        
class ETL:
    def __init__(self, share_prices_df, tickers):
        self.df_share_prices = share_prices_df.copy()
        self.tickers = tickers
        self.df_cleaned = None

        # Final feature set
        self.final_features = [
            'ticker', 'date',
            'open', 'high', 'low', 'close', 'adj. close', 'volume',
            'daily_return', 'volatility',
            '5_day_ma', '10_day_ma',
            'trend', 'next_day_close'
        ]

    def reset_and_clean_columns(self):
        self.df_share_prices.reset_index(inplace=True)
        self.df_share_prices.columns = self.df_share_prices.columns.str.lower()

    def handle_dates_and_duplicates(self):
        self.df_share_prices['date'] = pd.to_datetime(self.df_share_prices['date'], errors='coerce')
        self.df_share_prices.drop_duplicates(inplace=True)

    def handle_nulls_and_missing(self):
        self.df_share_prices.ffill(inplace=True)
        self.df_share_prices.dropna(subset=['close'], inplace=True)

    def feature_engineering(self):
        group = self.df_share_prices.groupby('ticker')

        # Daily return
        self.df_share_prices['daily_return'] = group['close'].pct_change()

        # Volatility (7-day rolling std)
        self.df_share_prices['volatility'] = group['close'].transform(lambda x: x.rolling(7).std())

        # Moving averages
        self.df_share_prices['5_day_ma'] = group['close'].transform(lambda x: x.rolling(5).mean())
        self.df_share_prices['10_day_ma'] = group['close'].transform(lambda x: x.rolling(10).mean())

        # Trend variable (classification target)
        self.df_share_prices['trend'] = (self.df_share_prices['daily_return'] > 0).astype(int)

        # Price prediction target
        self.df_share_prices['next_day_close'] = group['close'].shift(-1)

    def filter_data(self):
        # Filter for selected tickers and last 3 years
        three_years_ago = pd.Timestamp.today() - pd.DateOffset(years=3)
        self.df_share_prices = self.df_share_prices[
            (self.df_share_prices['ticker'].isin(self.tickers)) &
            (self.df_share_prices['date'] >= three_years_ago)
        ]

    '''def merge_and_save(self):
        df_merged = self.df_share_prices.merge(self.df_companies, on='ticker', how='left')
        model_columns = [col for col in self.final_features if col not in ['ticker', 'date']]
        df_filtered = df_merged.dropna(subset=model_columns)

        # Optional: Keep only tickers with enough data
        df_filtered = df_filtered.groupby('ticker').filter(lambda x: len(x) >= 60)

        self.df_cleaned = df_filtered[self.final_features]
        self.df_cleaned.to_csv(self.output_file, index=False)'''
    
    

    def run_pipeline(self):
        self.reset_and_clean_columns()
        self.handle_dates_and_duplicates()
        self.handle_nulls_and_missing()
        self.feature_engineering()
        self.filter_data()

        model_columns = [col for col in self.final_features if col not in ['ticker', 'date']]
        df_filtered = self.df_share_prices.dropna(subset=model_columns)
        df_filtered = df_filtered.groupby('ticker').filter(lambda x: len(x) >= 60)

        self.df_cleaned = df_filtered[self.final_features]
        return self.df_cleaned
    
    def get_financial_statement(self, ticker: str, start: str, end: str):
        """
        Fetch financial statements (Income, Balance Sheet, Cash Flow) for a given ticker and time range.
        Returns a DataFrame.
        """
        # API endpoint
        url = f"https://backend.simfin.com/api/v3/companies/statements/compact?ticker={ticker}&statements=PL,CF,BS&period=&start={start}&end={end}"

        # API headers with authorization
        headers = {
            "accept": "application/json",
            "Authorization": "79f8076c-cdc4-4ffe-9827-a82f92215739"  # Replace with your valid API key
        }

        # Fetch data from the API
        response = requests.get(url, headers=headers)
        json_data = response.json()

        # Create an empty list to store the data
        statement_data = []

        # Extract statement and data from JSON
        for company in json_data:
            for stmt in company['statements']:
                statement_type = stmt['statement']
                columns = stmt['columns']
                # Iterate over all rows in the data list
                for data_row in stmt['data']:
                    # Create a dictionary with statement type and its data
                    row_dict = {'Statement': statement_type}
                    row_dict.update(dict(zip(columns, data_row)))
                    statement_data.append(row_dict)

        df = pd.DataFrame(statement_data)
        df["Report Date"] = pd.to_datetime(df["Report Date"], errors="coerce")

        return df
    
    def merge_data(self, prices_df, financials_df):
        """
        Merge share prices and financial statements based on the closest available dates.
        """
        if prices_df is None or financials_df is None:
            print("Error: One of the datasets is missing.")
            return None

        # Ensure datetime format for merging
        prices_df = prices_df.sort_values("Date")
        financials_df = financials_df.sort_values("Report Date")

        # Merge on closest available date
        merged_df = pd.merge_asof(prices_df, financials_df, left_on="Date", right_on="Report Date", direction="backward")
        return merged_df

    
class TradingStrategy:
    def __init__(self, initial_cash=10000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.shares = 0
        self.trade_log = []

    def decide_action(self, trend_pred):
        """
        Determine action based on trend prediction.
        Returns one of: ('BUY', 1), ('SELL', 1), or ('HOLD', 0)
        """
        if trend_pred == 1:
            return "BUY", 1
        elif trend_pred == 0 and self.shares > 0:
            return "SELL", 1
        else:
            return "HOLD", 0

    def apply_strategy(self, df):
        """
        Applies the strategy over the given dataframe which must contain:
        - 'date': date of the trade
        - 'close': actual closing price
        - 'trend_pred': predicted trend (1 for up, 0 for down)
        Returns the dataframe with action logs and final performance.
        """
        actions = []
        for _, row in df.iterrows():
            action, qty = self.decide_action(row['trend_pred'])
            price = row['close']

            if action == "BUY":
                self.cash -= price * qty
                self.shares += qty
            elif action == "SELL":
                self.cash += price * qty
                self.shares -= qty

            self.trade_log.append({
                "date": row['date'],
                "action": action,
                "price": price,
                "cash": self.cash,
                "shares": self.shares,
                "portfolio_value": self.cash + self.shares * price
            })

            actions.append(action)

        df['action'] = actions
        self.df_trades = pd.DataFrame(self.trade_log)
        return df

    def get_final_value(self, last_price):
        """Returns final portfolio value and profit."""
        final_value = self.cash + self.shares * last_price
        profit = final_value - self.initial_cash
        return final_value, profit

    def get_trade_log(self):
        return pd.DataFrame(self.trade_log)

    def summary(self):
        final_value, profit = self.get_final_value(self.df_trades.iloc[-1]['price'])
        return {
            "Initial Cash": self.initial_cash,
            "Final Portfolio Value": round(final_value, 2),
            "Total Profit": round(profit, 2),
            "Total Trades": len(self.trade_log),
            "Final Shares Held": self.shares
        }

'''if __name__ == '__main__':

    # Run ETL process
    etl = ETL(share_prices_df=df_share_prices, tickers=selected_tickers)
    df_cleaned = etl.run_pipeline()'''