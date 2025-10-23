import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

# Import core prediction signals and chatbot logic
from predict_signals import get_latest_signals 
from chatbot_agent import generate_insights, LLM 
import os

# --- 1. CONFIGURATION ---
# Set the layout to wide for better chart viewing
st.set_page_config(layout="wide") 

# Read the database URL securely from Streamlit's secrets.toml
try:
    # Use the full external URL saved in .streamlit/secrets.toml
    DATABASE_URL = st.secrets["database"]["url"] 
except KeyError:
    # Fallback for local testing outside Streamlit Cloud
    DATABASE_URL = os.environ.get("DATABASE_URL", 
                                  "postgresql+psycopg2://user:password@localhost:5432/stock_market_db")

engine = create_engine(DATABASE_URL)


# --- 2. DATA LOAD FUNCTIONS ---

@st.cache_data
def load_historical_data(ticker):
    """Loads historical transformed data for charting."""
    query = f"SELECT * FROM transformed_stock_data WHERE ticker='{ticker}' ORDER BY timestamp DESC"
    df = pd.read_sql(text(query), engine, index_col='timestamp')
    return df

@st.cache_data
def load_feature_importance():
    """Loads feature importance for XAI reasoning."""
    query = "SELECT feature, average_importance FROM model_feature_importance ORDER BY average_importance DESC"
    df = pd.read_sql(text(query), engine)
    
    # Filter out raw OHLCV columns for cleaner display
    COLUMNS_FOR_DISPLAY = ['MACD', 'EMA_50', 'SMA_20', 'VIX_Close', 'BNO_Crude_Close']
    df_filtered = df[df['feature'].isin(COLUMNS_FOR_DISPLAY)].copy()
    return df_filtered.sort_values(by='average_importance', ascending=False)


# --- 3. DASHBOARD UI SETUP ---

# Fetch signals and feature importance once at the start
signal_df = get_latest_signals()
df_importance = load_feature_importance()

st.title("AI Trading Console (BSE/NSE)")
st.markdown("---")

# 3.1 Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I am your Master Trader AI. Ask for a recommendation, market analysis, or geopolitical insight."})

with st.sidebar:
    st.title("💬 Master Trader AI")
    
    # Display Model Status
    if LLM is None:
         st.error("AI OFFLINE: Ollama connection failed.")
    else:
         st.success("AI ONLINE: Systems operational.")
    st.markdown("---")
    
    # Create the chat history container
    chat_history_placeholder = st.container(height=450, border=True)

    # Display History in the container
    with chat_history_placeholder:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Sidebar Buttons
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.info("Tip: Ask about VIX, Crude Oil, or a specific stock.")

# --- 4. MAIN CONTENT (Signals and Charts) ---

# A. Display AI Trading Signals
st.subheader("Current AI Signals")

if not signal_df.empty:
    st.dataframe(
        signal_df,
        use_container_width=True,
        hide_index=True,
        column_order=['Date', 'Ticker', 'Latest Price (₹)', 'Signal', 'Probability_Buy']
    )
else:
    st.warning("No current signals generated.")
st.markdown("---")


# B. Chart and Feature Importance 
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Technical View")
    # Ticker Selection
    TICKERS = list(signal_df['Ticker'].unique()) if not signal_df.empty else ['RELIANCE.NS', 'HDFCBANK.NS']
    selected_ticker = st.selectbox("Stock Ticker:", TICKERS)
    
    df_chart = load_historical_data(selected_ticker)
    
    # --- CANDLESTICK CHART ---
    fig = go.Figure(data=[go.Candlestick(
        x=df_chart.index,
        open=df_chart['open'],
        high=df_chart['high'],
        low=df_chart['low'],
        close=df_chart['close'],
        name='Price'
    )])

    # Add Moving Averages
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_50'], name='EMA 50', line={'color': 'blue', 'width': 1.0}))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_20'], name='SMA 20', line={'color': 'red', 'width': 1.0}))
    
    # Minimalist Layout
    fig.update_layout(
        title=f'{selected_ticker} Price Action & Key Averages',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(t=30, b=30), 
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)


with col2:
    st.subheader("AI Reasoning Drivers")
    st.markdown("---")
    
    # Display Feature Importance
    if not df_importance.empty:
        st.dataframe(
            df_importance,
            column_config={
                "feature": "Feature",
                "average_importance": st.column_config.ProgressColumn(
                    "Importance Score",
                    format="%.2f",
                    min_value=0,
                    max_value=df_importance['average_importance'].max(),
                )
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Run the training script to generate AI reasoning data.")


# --- 5. CHAT INPUT (Pinned to Bottom of Page) ---

# This input is placed in the main body, ensuring it sticks to the bottom.
if prompt := st.chat_input("Chat with the Master Trader AI..."):
    
    # 1. Store and Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Get AI Response
    with st.spinner("AI is synthesizing data and geopolitical news..."):
        try:
            # Call the core logic function
            full_response = generate_insights(prompt)
            
            # 3. Store AI Message and Rerun
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun() 
        
        except Exception as e:
             error_msg = f"ERROR: Analysis failed. Check Ollama status. Details: {e}"
             st.error(error_msg)
             st.session_state.messages.append({"role": "assistant", "content": error_msg})
             st.rerun()