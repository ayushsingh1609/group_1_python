import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned stock data
df = pd.read_csv('cleaned_stock_data.csv')

# Select top 10 companies based on highest trading volume
top_10_companies = df.groupby('ticker')['volume'].sum().nlargest(10).index.tolist()

print(f"🔹 Training models for top 10 companies: {top_10_companies}")

for ticker in top_10_companies:
    print(f"🔹 Training model for {ticker}...")

    # Filter data for selected company
    df_company = df[df['ticker'] == ticker].copy()
    
    # Convert 'date' column to datetime
    df_company['date'] = pd.to_datetime(df_company['date'])
    df_company = df_company.sort_values(by='date')

    # Create Target Variable (1 = Up, 0 = Down)
    df_company['trend'] = (df_company['close'].shift(-1) > df_company['close']).astype(int)
    
    # Drop last row
    df_company = df_company[:-1]

    # Feature Engineering
    df_company['daily_return'] = df_company['close'].pct_change()
    df_company['volatility'] = df_company['close'].rolling(7).std()
    df_company['ma7'] = df_company['close'].rolling(7).mean()
    df_company['ma30'] = df_company['close'].rolling(30).mean()
    
    # Handle missing or infinite values
    df_company.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
    df_company.dropna(inplace=True)  # Drop rows with NaN values

    # Prepare Data
    features = ['close', 'daily_return', 'volatility', 'ma7', 'ma30']
    X = df_company[features]
    y = df_company['trend']

    # Skip companies with too little data
    if len(X) < 5:
        print(f"❌ Skipping {ticker} due to insufficient data ({len(X)} samples).")
        continue  # Skip this company

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter Tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)

    # Dynamically adjust cross-validation splits
    cv_splits = min(5, len(y_train))  # Ensure cv is never greater than the dataset size
    grid_search = GridSearchCV(rf, param_grid, cv=cv_splits, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get Best Model
    best_model = grid_search.best_estimator_
    print(f"✅ Best Parameters for {ticker}: {grid_search.best_params_}")

    # Save Model
    model_filename = f"models/{ticker}_trend_model.pkl"
    joblib.dump(best_model, model_filename)
    print(f"✅ Model saved as '{model_filename}'")

print("🎯 All models trained and saved successfully!")
