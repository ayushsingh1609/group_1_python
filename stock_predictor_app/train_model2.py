import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("filtered_5_companies_stock_data.csv")

# Ensure output directory exists
output_dir = "stock_predictor_app/saved_models"
os.makedirs(output_dir, exist_ok=True)

# Hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

# Features matching the updated ETL
features = [
    'open', 'high', 'low', 'close', 'adj. close', 'volume',
    'daily_return', 'volatility',
    '5_day_ma', '10_day_ma'
]

# Unique tickers
tickers = df['ticker'].unique()

for ticker in tickers:
    print(f"Processing {ticker}...")
    df_ticker = df[df['ticker'] == ticker]

    # 1. Stock Trend Prediction (Classification)
    X_class = df_ticker[features]
    y_class = df_ticker['trend']

    # Scale features
    scaler = StandardScaler()
    X_class_scaled = scaler.fit_transform(X_class)

    # Balance classes using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_class_scaled, y_class)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    grid_search_clf = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, verbose=0)
    grid_search_clf.fit(X_train_c, y_train_c)

    best_clf = grid_search_clf.best_estimator_
    y_pred_c = best_clf.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    f1 = f1_score(y_test_c, y_pred_c)
    print(f"Trend Prediction Accuracy for {ticker}: {acc:.4f}, F1 Score: {f1:.4f}")

    # Save classification model
    joblib.dump(best_clf, os.path.join(output_dir, f"{ticker}_trend_model.pkl"))

    # 2. Stock Price Prediction (Regression)
    X_reg = df_ticker[features]
    y_reg = df_ticker['next_day_close']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(random_state=42)
    grid_search_reg = GridSearchCV(reg, param_grid, cv=3, n_jobs=-1, verbose=0)
    grid_search_reg.fit(X_train_r, y_train_r)

    best_reg = grid_search_reg.best_estimator_
    y_pred_r = best_reg.predict(X_test_r)
    mae = mean_absolute_error(y_test_r, y_pred_r)
    print(f"Stock Price Prediction MAE for {ticker}: {mae:.4f}")

    # Save regression model
    joblib.dump(best_reg, os.path.join(output_dir, f"{ticker}_price_model.pkl"))

print("All models saved successfully.")
