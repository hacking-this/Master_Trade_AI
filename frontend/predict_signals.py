import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
import os # Required for model path management
import streamlit as st # Ensure this is imported at the top

# --- 1. CONFIGURATION ---
# IMPORTANT: Use the full cloud URL for the database connection
try:
    DATABASE_URL = st.secrets["DATABASE_URL"]
except:
    # Fallback for testing outside Streamlit/in Airflow if not using secrets management
    DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://default:url@localhost:5432/default")

engine = create_engine(DATABASE_URL)

# Define the path to the model file, relative to the project root
# (frontend/predict_signals.py needs to go up one directory to find data/models)
MODEL_PATH = '../data/models/stock_predictor_xgboost.pkl' 

# --- 2. LOAD MODEL AND DATA ---

# Load model once when the script starts (XGBoost is loaded via joblib)
try:
    if os.path.exists(MODEL_PATH):
        XGB_MODEL = joblib.load(MODEL_PATH)
    else:
        XGB_MODEL = None
except Exception as e:
    XGB_MODEL = None

# --- 3. PREDICTION LOGIC ---

# The original model was trained ONLY on these 11 features. 
# We must select these exact columns and ignore the new WFO features.
FINAL_TRAINING_FEATURES = [
    'open', 'high', 'low', 'close', 'adj_close', 'volume', 
    'SMA_20', 'EMA_50', 'MACD', 
    'VIX_Close', 'BNO_Crude_Close'
]

def get_latest_signals():
    """Loads the model and generates buy/sell signals for the latest known data."""
    
    if XGB_MODEL is None:
        return pd.DataFrame({'Signal': ['AI Model Offline. Rerun training script.']})

    # Load the latest data that has a Target of -1 (unknown future)
    # This represents the data that is ready for a *new* prediction.
    query = """
        SELECT *
        FROM transformed_stock_data
        WHERE "Target" = -1 
        ORDER BY timestamp DESC;
    """
    try:
        df_predict = pd.read_sql(text(query), engine)
    except Exception as e:
        return pd.DataFrame({'Signal': [f"Database Error: {str(e)}"]})
    
    if df_predict.empty:
        return pd.DataFrame({'Signal': ['No new data points found for prediction.']})

    # CRITICAL FIX: DEDUPLICATION AND FEATURE SELECTION
    
    # 1. Deduplicate records (removes the 5-row error artifact)
    df_predict.drop_duplicates(subset=['timestamp', 'ticker'], keep='first', inplace=True)
    
    # 2. Select ONLY the features the saved model expects (11 columns)
    X_predict = df_predict[FINAL_TRAINING_FEATURES]
    
    # Generate probability scores (proba[1] is the chance of Target=1 / BUY)
    probabilities = XGB_MODEL.predict_proba(X_predict)[:, 1]
    
    # --- 4. COMPILE RESULTS ---
    
    # Define thresholds based on analysis (40% precision requires a cautious approach)
    threshold_strong = 0.60
    threshold_weak = 0.52 
    
    signals = pd.DataFrame({
        'Date': df_predict['timestamp'],
        'Ticker': df_predict['ticker'],
        'Latest Price (₹)': df_predict['close'], 
        'Probability_Buy': probabilities,
        'Signal': np.where(probabilities >= threshold_strong, 'STRONG BUY', 
                           np.where(probabilities >= threshold_weak, 'WEAK BUY', 'HOLD/SELL'))
    })
    
    # Merge is unnecessary now; all data is pulled in one clean DF
    return signals.sort_values(by='Probability_Buy', ascending=False).reset_index(drop=True)

if __name__ == '__main__':
    # Test the function (optional)
    result = get_latest_signals()
    print(result)