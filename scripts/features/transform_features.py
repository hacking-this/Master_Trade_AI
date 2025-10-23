import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# --- 1. CONFIGURATION ---
DATABASE_URL = ("postgresql+psycopg2://ai_stock_trader_user:WcPpqu1IDRnqv95NoV1dUsMp17RCbTMR@dpg-d3rijvur433s73e6adeg-a.oregon-postgres.render.com/ai_stock_trader""?sslmode=require")

engine = create_engine(DATABASE_URL)

STOCK_TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'HINDUNILVR.NS']
MACRO_TICKERS = ['^VIX', 'BNO']

# --- 2. PURE PANDAS TECHNICAL INDICATORS (Stable Substitutes) ---

# NOTE: These functions use pure Pandas to avoid the TA-Lib dependency issues.
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

# --- 3. TRANSFORMATION FUNCTION ---
def calculate_and_store_features():
    print("--- Starting Feature Engineering (WFO & Lagged Features) ---")

    # A. LOAD ALL RAW DATA
    try:
        query = "SELECT timestamp, ticker, open, high, low, close, adj_close, volume FROM raw_stock_data ORDER BY timestamp ASC"
        df_all = pd.read_sql(text(query), engine).set_index('timestamp')
    except Exception as e:
        print(f"ERROR: Failed to load raw data. Reason: {e}")
        return

    # --- B. EXTRACT MACRO FEATURES ---
    df_macro = df_all[df_all['ticker'].isin(MACRO_TICKERS)][['close', 'ticker']].copy()
    df_macro_pivoted = df_macro.pivot_table(index='timestamp', columns='ticker', values='close')
    df_macro_pivoted.rename(columns={'^VIX': 'VIX_Close', 'BNO': 'BNO_Crude_Close'}, inplace=True)
    df_macro_pivoted = df_macro_pivoted.ffill() 

    all_transformed_data = []

    # --- C. ITERATE OVER STOCKS AND CALCULATE FEATURES ---
    for ticker in STOCK_TICKERS:
        print(f"-> Processing features for stock: {ticker}")
        
        df_stock = df_all[df_all['ticker'] == ticker].copy()
        
        # CRITICAL FIX: Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
            df_stock[col] = pd.to_numeric(df_stock[col], errors='coerce') 

        # 1. MERGE MACRO DATA & FORWARD-FILL
        df_stock = df_stock.join(df_macro_pivoted, how='left')
        df_stock['VIX_Close'] = df_stock['VIX_Close'].ffill().fillna(0)
        df_stock['BNO_Crude_Close'] = df_stock['BNO_Crude_Close'].ffill().fillna(0)
        
        # 2. CORE TECHNICAL FEATURE ENGINEERING
        
        # Trend
        df_stock['SMA_20'] = df_stock['close'].rolling(window=20).mean().fillna(0)
        df_stock['EMA_50'] = calculate_ema(df_stock['close'], 50).fillna(0)
        
        # Momentum
        macd, macd_signal, macd_hist = calculate_macd(df_stock['close'])
        df_stock['MACD'] = macd.fillna(0)
        df_stock['MACD_Signal'] = macd_signal.fillna(0)
        df_stock['MACD_Hist'] = macd_hist.fillna(0)

        # --- NEW FEATURES: LAGS AND SEASONALITY ---

        # Calculate Daily Return
        df_stock['Daily_Return'] = df_stock['close'].pct_change()
        
        # Create Lagged Return Features (AI memory)
        for lag in [1, 3, 5, 10]:
            df_stock[f'Lag_Return_{lag}d'] = df_stock['Daily_Return'].shift(lag).fillna(0)

        # Calendar Features (Seasonality)
        df_stock.index = pd.to_datetime(df_stock.index)
        df_stock['DayOfWeek'] = df_stock.index.dayofweek
        df_stock['DayOfMonth'] = df_stock.index.day
        df_stock['Quarter'] = df_stock.index.quarter
        
        # 4. TARGET VARIABLE CREATION & ZERO-DROP IMPUTATION
        df_stock['Future_Return'] = (df_stock['close'].shift(-5) / df_stock['close']) - 1
        df_stock['Target'] = np.where(df_stock['Future_Return'] >= 0.01, 1, 0)
        df_stock.loc[df_stock['Future_Return'].isna(), 'Target'] = -1 # Mark unknown future targets

        # 5. FINAL COLLECTION (Update final_cols to include all new features)
        final_cols = ['ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 
                      'SMA_20', 'EMA_50', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                      'VIX_Close', 'BNO_Crude_Close', 'Future_Return', 'Target',
                      'Lag_Return_1d', 'Lag_Return_3d', 'Lag_Return_5d', 'Lag_Return_10d', 
                      'DayOfWeek', 'DayOfMonth', 'Quarter']
        
        df_clean = df_stock[final_cols]
        all_transformed_data.append(df_clean)

    # --- D. SAVE ALL TRANSFORMED DATA ---
    df_final_transformed = pd.concat(all_transformed_data)
    df_final_transformed.to_sql(
        'transformed_stock_data', 
        engine, 
        if_exists='replace', 
        index=True, 
        index_label='timestamp' 
    )
    print(f"\nSUCCESS: Features (WFO Ready) saved. Total rows: {len(df_final_transformed)}")

if __name__ == "__main__":
    calculate_and_store_features()