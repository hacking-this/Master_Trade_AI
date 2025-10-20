import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import time

# --- 1. CONFIGURATION ---
DATABASE_URL = "postgresql://user:password@ai_stock_project-db-1:5432/stock_market_db"
engine = create_engine(DATABASE_URL)

# List of all tickers, including stocks and macro proxies
# These are used for iteration and config table population
TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'HINDUNILVR.NS', '^VIX', 'BNO']

# --- 2. DYNAMIC TICKER LOAD AND WATERMARK FUNCTIONS ---

def create_config_table():
    """Creates the configuration table and inserts initial tickers."""
    INITIAL_TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'HINDUNILVR.NS', '^VIX', 'BNO']

    try:
        with engine.connect() as conn:
            with conn.begin():
                # Create config table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS ticker_list_config (
                        ticker VARCHAR(20) PRIMARY KEY
                    );
                """))
                
                # Insert initial tickers only if the table is empty
                current_tickers = conn.execute(text("SELECT ticker FROM ticker_list_config")).scalars().all()
                if not current_tickers:
                    print("Inserting initial ticker list...")
                    for ticker in INITIAL_TICKERS:
                        conn.execute(text("INSERT INTO ticker_list_config (ticker) VALUES (:t)"), {"t": ticker})
    except Exception as e:
        print(f"ERROR: Could not create/populate config table: {e}")
        raise

def load_dynamic_tickers():
    """Reads the current list of tickers from the database."""
    try:
        query = "SELECT ticker FROM ticker_list_config"
        df = pd.read_sql(text(query), engine)
        return df['ticker'].tolist()
    except Exception as e:
        print(f"ERROR: Failed to load tickers dynamically: {e}")
        return []

def get_watermark_date():
    """Queries the database to find the latest timestamp recorded for any stock."""
    FALLBACK_DATE = datetime(2020, 10, 19)
    
    try:
        query = "SELECT MAX(timestamp) FROM raw_stock_data"
        with engine.connect() as conn:
            max_date = conn.execute(text(query)).scalar()

        if max_date:
            start_date = max_date + timedelta(days=1)
            return start_date.strftime('%Y-%m-%d')
        else:
            return FALLBACK_DATE.strftime('%Y-%m-%d')
            
    except Exception as e:
        print(f"Watermarking failed ({e}). Starting from historical date.")
        return FALLBACK_DATE.strftime('%Y-%m-%d')

# --- 3. DATA FETCHING AND STORAGE FUNCTION (ROBUST SINGLE-TICKER LOOP) ---

def fetch_and_store_single_ticker(ticker, start_date):
    """Fetches data for a single ticker using the stable Ticker().history() method."""
    print(f"-> Fetching data for {ticker} from {start_date}...")
    
    try:
        # A. FETCH DATA: Use Ticker().history() which is more stable for single symbols
        t = yf.Ticker(ticker)
        # Fetch data up to today
        data = t.history(start=start_date, interval="1d", auto_adjust=True)
        
        if data.empty:
            print(f"WARNING: No data found for {ticker} since {start_date}.")
            return 0

        # B. CLEANUP AND STANDARDIZATION
        data.reset_index(inplace=True)
        
        # 1. Standardize YFinance column names to match schema (simple strings)
        data.rename(columns={
            'Date': 'timestamp', 
            'Adj Close': 'adj_close', # This column may or may not exist due to auto_adjust=True
            'Open': 'open', 
            'High': 'high', 
            'Low': 'low', 
            'Close': 'close', 
            'Volume': 'volume'
        }, inplace=True)
        
        # 2. Guarantee 'adj_close' Column Exists
        if 'adj_close' not in data.columns:
            # If the column was dropped, the 'close' column holds the adjusted price.
            data['adj_close'] = data['close']
        
        # 3. Final Column Selection and Ordering (Crucial for database insertion)
        final_cols = ['timestamp', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        data = data[final_cols].copy()
        data['ticker'] = ticker # Add ticker identifier

        # C. INSERT INTO POSTGRESQL (This structure guarantees single-level columns)
        data.to_sql('raw_stock_data', engine, if_exists='append', index=False)
        
        print(f"-> Successfully loaded {len(data)} rows for {ticker}.")
        return len(data)
        
    except Exception as e:
        print(f"CRITICAL ERROR during fetching/loading {ticker}: {e}")
        # Return 0 to allow the MLOps pipeline to continue to the next stock
        return 0

def fetch_and_store_all_data(start_date, tickers):
    """Orchestrates the incremental load for all tickers."""
    total_rows = 0
    
    print(f"\n--- Starting Incremental Load for {len(tickers)} tickers ---")
    for ticker in tickers:
        total_rows += fetch_and_store_single_ticker(ticker, start_date)
        time.sleep(1) # Be polite to the YFinance API

    print(f"\n--- Total Incremental Rows Loaded: {total_rows} ---")

# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    
    # 4.1: ENSURE CONFIG AND RAW DATA TABLES EXIST
    create_config_table()
    
    try:
        with engine.connect() as conn:
            with conn.begin(): 
                conn.execute(text(
                    """
                    CREATE TABLE IF NOT EXISTS raw_stock_data (
                        timestamp DATE NOT NULL,
                        ticker VARCHAR(20) NOT NULL,
                        open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, 
                        adj_close NUMERIC, volume BIGINT,
                        PRIMARY KEY (timestamp, ticker)
                    );
                    SELECT create_hypertable('raw_stock_data', 'timestamp', if_not_exists => TRUE);
                    """
                ))
        print("Database schema 'raw_stock_data' ensured.")
    except Exception as e:
        print(f"CRITICAL ERROR during schema setup: {e}")
        exit() 

    # 4.2: EXECUTE INCREMENTAL LOAD
    TICKER_LIST = load_dynamic_tickers()
    if not TICKER_LIST:
        print("FATAL: Could not load any tickers from config table. Exiting.")
        exit()
        
    START_DATE = get_watermark_date()
    fetch_and_store_all_data(START_DATE, TICKER_LIST)

    print("\n--- Incremental Ingestion Complete ---")