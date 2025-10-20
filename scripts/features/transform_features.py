import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# --- CONFIGURATION ---
DATABASE_URL = "postgresql://user:password@ai_stock_project-db-1:5432/stock_market_db"
engine = create_engine(DATABASE_URL)
STOCK_TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'HINDUNILVR.NS']
MACRO_TICKERS = ['^VIX', 'BNO']

# --- PURE PANDAS RSI CALCULATION (Keep for best effort) ---
def calculate_rsi(series, period=14):
    diff = series.diff(1).dropna()
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean()
    return 100 - (100 / (1 + rs))

def calculate_and_store_features():
    print("--- Starting Feature Engineering (VIX/BNO PRIORITY) ---")

    # A. LOAD ALL RAW DATA
    try:
        query = "SELECT timestamp, ticker, open, high, low, close, adj_close, volume FROM raw_stock_data ORDER BY timestamp ASC"
        df_all = pd.read_sql(text(query), engine).set_index('timestamp')
    except Exception as e:
        print(f"ERROR: Failed to load raw data. Reason: {e}")
        return

    # B. EXTRACT MACRO FEATURES
    df_macro = df_all[df_all['ticker'].isin(MACRO_TICKERS)][['close', 'ticker']].copy()
    df_macro_pivoted = df_macro.pivot_table(index='timestamp', columns='ticker', values='close')
    df_macro_pivoted.rename(columns={'^VIX': 'VIX_Close', 'BNO': 'BNO_Crude_Close'}, inplace=True)
    df_macro_pivoted = df_macro_pivoted.ffill() 
    
    all_transformed_data = []

    # C. ITERATE OVER STOCKS AND CALCULATE FEATURES
    for ticker in STOCK_TICKERS:
        print(f"-> Processing features for stock: {ticker}")
        df_stock = df_all[df_all['ticker'] == ticker].copy()
        
        # CRITICAL: Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'adj_close']:
            df_stock[col] = pd.to_numeric(df_stock[col], errors='coerce') 

        # 1. MERGE MACRO DATA & FORWARD-FILL
        df_stock = df_stock.join(df_macro_pivoted, how='left')
        df_stock['VIX_Close'] = df_stock['VIX_Close'].ffill().fillna(0) # Fill VIX/BNO NaNs with 0
        df_stock['BNO_Crude_Close'] = df_stock['BNO_Crude_Close'].ffill().fillna(0) 
        
        # 2. CORE TECHNICAL FEATURE ENGINEERING (Simple, Stable Features Only)
        
        # SMA/EMA (Highly Stable)
        df_stock['SMA_20'] = df_stock['close'].rolling(window=20).mean().fillna(0)
        df_stock['EMA_50'] = df_stock['close'].ewm(span=50, adjust=False).mean().fillna(0)
        
        # MACD (Stable)
        ema_fast = df_stock['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df_stock['close'].ewm(span=26, adjust=False).mean()
        df_stock['MACD'] = (ema_fast - ema_slow).fillna(0)
        
        # 3. TARGET VARIABLE CREATION
        df_stock['Future_Return'] = (df_stock['close'].shift(-5) / df_stock['close']) - 1
        df_stock['Target'] = np.where(df_stock['Future_Return'] >= 0.01, 1, 0)
        df_stock.loc[df_stock['Future_Return'].isna(), 'Target'] = -1 # Mark unknown future targets

        # 4. FINAL COLLECTION
        final_cols = ['ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 
                      'SMA_20', 'EMA_50', 'MACD', 
                      'VIX_Close', 'BNO_Crude_Close', 'Future_Return', 'Target']
        
        df_clean = df_stock[final_cols]
        all_transformed_data.append(df_clean)

    # --- D. SAVE ALL TRANSFORMED DATA ---
    if not all_transformed_data:
        print("\nCRITICAL: No stock data was successfully processed. Exiting.")
        return

    df_final_transformed = pd.concat(all_transformed_data)
    
    df_final_transformed.to_sql(
        'transformed_stock_data', 
        engine, 
        if_exists='replace', 
        index=True, 
        index_label='timestamp' 
    )
    
    print("\nSUCCESS: All feature engineering complete (VIX/BNO AND STABLE FEATURES).")
    print(f"Total rows saved to 'transformed_stock_data': {len(df_final_transformed)}")

if __name__ == "__main__":
    calculate_and_store_features()
    print("Transformation phase finished. Ready for AI Training (train_ai_model.py).")