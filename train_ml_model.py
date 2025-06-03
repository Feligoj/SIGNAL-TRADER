import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# --- CONFIG ---
HISTORICAL_CSV = 'historical_signals.csv'  # Path to your historical data
MODEL_PATH = 'ml_signal_model.pkl'

# --- Load data ---
df = pd.read_csv(HISTORICAL_CSV)

# --- Feature selection ---
# Adjust these columns to match your data structure
feature_cols = [
    'rsi', 'ema_fast', 'ema_slow', 'macd_line', 'macd_signal', 'macd_hist',
    'upper_bb', 'lower_bb', 'stoch_k', 'stoch_d', 'adx', 'tr', 'volatility',
    'close', 'open', 'high', 'low'
]
label_col = 'win'  # 1 for win, 0 for loss

# Drop rows with missing values
X = df[feature_cols].dropna()
y = df.loc[X.index, label_col]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Validation accuracy: {acc:.2%}")

# --- Save model ---
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
