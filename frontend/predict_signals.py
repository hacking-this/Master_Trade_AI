import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
import os 

# --- 1. CONFIGURATION ---
# IMPORTANT: Use the full external URL that works for your MLOps system
DATABASE_URL = "postgresql+psycopg2://ai_stock_trader_user:WcPpqu1IDRnqv95NoV1dUsMp17RCbTMR@dpg-d3rijvur433s73e6adeg-a.oregon-postgres.render.com/ai_stock_trader"
engine = create_engine(DATABASE_URL)

# Define the path to the model file, relative to the frontend/ directory
MODEL_PATH = '../data/models/stock_predictor_xgboost.pkl' 

# Load model once when the script starts (XGBoost is loaded via joblib)
try:
    if os.path.exists(MODEL_PATH):
        XGB_MODEL = joblib.load(MODEL_PATH)
    else:
        XGB_MODEL = None
except Exception:
    XGB_MODEL = None


# --- 2. PREDICTION LOGIC ---

# These are the *exact* features saved by train_ai_model.py. 
# The prediction data MUST match this list precisely.
FINAL_TRAINING_FEATURES = [
    'open', 'high', 'low', 'close', 'adj_close', 'volume', 
    'SMA_20', 'EMA_50', 'MACD', 'MACD_Signal', 'MACD_Hist', 
    'VIX_Close', 'BNO_Crude_Close', 
    'Lag_Return_1d', 'Lag_Return_3d', 'Lag_Return_5d', 'Lag_Return_10d', 
    'DayOfWeek', 'DayOfMonth', 'Quarter'
]

def get_latest_signals():
    """Loads the latest data (Target = -1) and generates buy/sell probabilities."""
    
    if XGB_MODEL is None:
        return pd.DataFrame({'Signal': ['AI Model Offline. Rerun training script.']})

    # Load all data with unknown future targets (Target = -1)
    query = """
        SELECT *
        FROM transformed_stock_data
        WHERE "Target" = -1 
        ORDER BY timestamp DESC;
    """
    try:
        df_predict = pd.read_sql(text(query), engine).set_index('timestamp')
    except Exception:
        return pd.DataFrame({'Signal': ['Database Error: Could not retrieve prediction data.']})

    if df_predict.empty:
        return pd.DataFrame({'Signal': ['No new data points found for prediction.']})

    # CRITICAL FIX: ISOLATE THE LATEST DATE for a clean dashboard view
    latest_date = df_predict.index.max()
    df_predict = df_predict[df_predict.index == latest_date].copy()
    
    # 1. Deduplicate records
    df_predict.drop_duplicates(subset=['ticker'], keep='first', inplace=True)
    
    # 2. Select ONLY the features the saved model expects (20 columns)
    X_predict = df_predict[FINAL_TRAINING_FEATURES]
    
    # Generate probability scores (proba[1] is the chance of Target=1 / BUY)
    probabilities = XGB_MODEL.predict_proba(X_predict)[:, 1]
    
    # --- 3. COMPILE RESULTS ---
    
    # Define thresholds (based on your initial 44% precision score, we use a cautious threshold)
    threshold_strong = 0.60
    threshold_weak = 0.52 
    
    signals = pd.DataFrame({
        'Date': df_predict.index,
        'Ticker': df_predict['ticker'],
        'Latest Price (₹)': df_predict['close'], 
        'Probability_Buy': probabilities,
        'Signal': np.where(probabilities >= threshold_strong, 'STRONG BUY', 
                           np.where(probabilities >= threshold_weak, 'WEAK BUY', 'HOLD/SELL'))
    })
    
    return signals.sort_values(by='Probability_Buy', ascending=False).reset_index(drop=True)

if __name__ == '__main__':
    result = get_latest_signals()
    print(result.head(10))