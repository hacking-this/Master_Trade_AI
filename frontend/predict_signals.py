# predict_signals.py (The Corrected Script)

import pandas as pd
import numpy as np # Added import for np.where
import joblib
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://user:password@localhost:5432/stock_market_db"
engine = create_engine(DATABASE_URL)
MODEL_FILENAME = '../data/models/stock_predictor_xgboost.pkl'

def get_latest_signals():
    """Loads the model and generates buy/sell signals for the latest known data."""
    try:
        model = joblib.load(MODEL_FILENAME)
    except FileNotFoundError:
        return pd.DataFrame({'Signal': ['Model not found. Rerun training.']})

    # Load the latest data that has a Target of -1 (unknown future)
    query = """
        SELECT *
        FROM transformed_stock_data
        WHERE "Target" = -1 
        ORDER BY timestamp DESC;
    """
    df_predict = pd.read_sql(text(query), engine)
    
    if df_predict.empty:
        return pd.DataFrame({'Signal': ['No new data points found for prediction.']})

    # --- CRITICAL FIX 1: DEDUPLICATION ON PREDICTION DATA ---
    # Even if you deduplicated the training set, the raw query data might still have duplicates.
    df_predict.drop_duplicates(subset=['timestamp', 'ticker'], keep='first', inplace=True)
    df_predict.set_index('timestamp', inplace=True) # Set index for cleaner feature selection

    # Prepare feature set (must match the training features exactly)
    COLUMNS_TO_EXCLUDE = ['ticker', 'Future_Return', 'Target', 'RSI_14', 'BBANDS_Upper', 'BBANDS_Middle', 'BBANDS_Lower']
    FEATURE_COLS = df_predict.columns.drop(COLUMNS_TO_EXCLUDE, errors='ignore')
    X_predict = df_predict[FEATURE_COLS]
    
    # Generate probability scores 
    probabilities = model.predict_proba(X_predict)[:, 1]
    
    # --- CRITICAL FIX 2: Reconstruct DataFrame from clean sources ---
    
    # Define a clean threshold (adjusting for low precision score)
    threshold_strong = 0.60
    threshold_weak = 0.52 
    
    signals = pd.DataFrame({
        'Date': df_predict.index,
        'Ticker': df_predict['ticker'].values,
        'Latest Price (₹)': df_predict['close'].values, # Pull close price directly from the clean DF
        'Probability_Buy': probabilities,
        'Signal': np.where(probabilities >= threshold_strong, 'STRONG BUY', 
                           np.where(probabilities >= threshold_weak, 'WEAK BUY', 'HOLD/SELL'))
    })
    
    return signals.sort_values(by='Probability_Buy', ascending=False).reset_index(drop=True)

if __name__ == '__main__':
    # Test the function (optional)
    result = get_latest_signals()
    print(result.head())