import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from util import PySimFin, ETL
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
simfin = PySimFin(API_KEY)

ticker = "AAPL"
start_date = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Get and preprocess data
df = simfin.get_share_prices(ticker, start_date, end_date)
df = simfin.rename_columns(df)
df['ticker'] = ticker

etl = ETL(share_prices_df=df, tickers=[ticker])
df_cleaned = etl.run_pipeline()

# Define features and targets
features = ['open', 'high', 'low', 'close', 'adj. close', 'volume',
            'daily_return', 'volatility', '5_day_ma', '10_day_ma']
X = df_cleaned[features]
y_price = df_cleaned['next_day_close']
y_trend = df_cleaned['trend']

# Train/test split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_price, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_trend, test_size=0.2, random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [100],
    'max_depth': [10, None],
    'min_samples_split': [5],
    'min_samples_leaf': [2]
}

# Train regression model
reg = RandomForestRegressor(random_state=42)
grid_r = GridSearchCV(reg, param_grid, cv=3, n_jobs=-1)
grid_r.fit(X_train_r, y_train_r)
price_model = grid_r.best_estimator_

# Train classification model
clf = RandomForestClassifier(random_state=42)
grid_c = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1)
grid_c.fit(X_train_c, y_train_c)
trend_model = grid_c.best_estimator_

# Save models
os.makedirs("saved_models", exist_ok=True)
joblib.dump(price_model, "saved_models/AAPL_price_model.pkl")
joblib.dump(trend_model, "saved_models/AAPL_trend_model.pkl")

# Evaluate and print scores
y_pred_price = price_model.predict(X_test_r)
y_pred_trend = trend_model.predict(X_test_c)

print("✅ Models retrained and saved.")
print("Price MAE:", mean_absolute_error(y_test_r, y_pred_price))
print("Trend Accuracy:", accuracy_score(y_test_c, y_pred_trend))
