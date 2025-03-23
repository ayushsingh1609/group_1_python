import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

class ETL:
    def __init__(self, share_prices_df, tickers, output_file='filtered_5_companies_stock_data.csv'):
        self.df_share_prices = share_prices_df.copy()
        self.tickers = tickers
        self.output_file = output_file
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
        #self.merge_and_save()
        return self.df_cleaned
    
if __name__ == '__main__':

    # Run ETL process
    etl = ETL(share_prices_df=df_share_prices, tickers=selected_tickers)
    df_cleaned = etl.run_pipeline()