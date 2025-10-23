import pandas as pd
from sqlalchemy import create_engine, text
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate # Corrected import path
from langchain_core.output_parsers import StrOutputParser
from predict_signals import get_latest_signals 
import json


# --- 1. CONFIGURATION & SETUP ---
DATABASE_URL = "postgresql+psycopg2://ai_stock_trader_user:WcPpqu1IDRnqv95NoV1dUsMp17RCbTMR@dpg-d3rijvur433s73e6adeg-a.oregon-postgres.render.com/ai_stock_trader"
engine = create_engine(DATABASE_URL)

# Initialize Ollama LLM
try:
    LLM = Ollama(model="mistral", temperature=0.2) 
except Exception as e:
    LLM = None 


# --- 2. DATA RETRIEVAL FUNCTIONS ---

def load_feature_importance():
    """Loads the AI model's top Feature Importance for XAI reasoning."""
    query = "SELECT feature, average_importance FROM model_feature_importance ORDER BY average_importance DESC LIMIT 5"
    df = pd.read_sql(text(query), engine)
    
    importance_list = [f"{row['feature']} (Influence: {row['average_importance']:.4f})" for index, row in df.iterrows()]
    return "\n".join(importance_list)

def load_geopolitical_context():
    """Loads current VIX and Crude Oil levels (for numeric context)."""
    query = """
        SELECT "VIX_Close", "BNO_Crude_Close", "timestamp"
        FROM transformed_stock_data 
        ORDER BY timestamp DESC LIMIT 1;
    """
    df = pd.read_sql(text(query), engine)
    
    if df.empty:
        return "Geopolitical context data not available."
        
    return (
        f"Latest Data Date: {df['timestamp'].iloc[0].strftime('%Y-%m-%d')}\n"
        f"Global Risk (VIX Index): {df['VIX_Close'].iloc[0]:.2f}\n"
        f"Crude Oil Proxy (BNO): {df['BNO_Crude_Close'].iloc[0]:.2f}"
    )

# --- 3. DYNAMIC RESEARCH FUNCTION (SIMULATION FIX) ---

def perform_dynamic_research():
    """
    Generates structured, simulated geopolitical research results for citation.
    This replaces the external search tool call with high-value mock data for stability.
    """
    simulated_news = [
        {"source_title": "Economic Times (Simulated)", 
         "snippet": "FIIs have reversed months of selling, injecting fresh capital into Indian markets, driven by easing global inflation fears. This trend strongly favors large-cap banking and IT sectors."},
        
        {"source_title": "Mint Business Daily (Simulated)", 
         "snippet": "Crude oil futures dropped 3% following OPEC+'s unexpected decision to maintain output, dramatically reducing import costs and boosting profits for transport and FMCG companies in India."},
        
        {"source_title": "Geojit Financial Report (Simulated)", 
         "snippet": "The current VIX level (below 15) signals low immediate systemic risk. The primary risk remains election uncertainty and US-China trade tensions, but domestic liquidity is currently strong."},
    ]
    
    formatted_research = []
    for item in simulated_news:
        formatted_research.append(f"[{item['source_title']}]: {item['snippet']}")

    return "\n".join(formatted_research)


# --- 4. THE CHATBOT CHAIN (RAG & XAI LOGIC) ---

def generate_insights(user_query: str):
    """Combines AI signals, feature data, and geopolitical context to answer user query."""
    
    if LLM is None:
        return "AI service offline. Please ensure Ollama is running and the model is loaded."
    
    signals_df = get_latest_signals()
    
    # 1. Retrieve necessary context data
    research_text = perform_dynamic_research()
    signals_text = signals_df.to_markdown(index=False)
    feature_importance_text = load_feature_importance()
    geo_context_text = load_geopolitical_context()
    
    # --- The System Prompt: Guiding the LLM for Human-Friendly XAI and Citation ---
    template = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a Master Trader AI for the Indian Stock Market (BSE/NSE). "
         "Your primary goal is to translate complex data into clear, human-friendly investment advice. "
         "Your response MUST be concise, professional, and directly address the user's inquiry based ONLY on the provided data and context. "
         "Follow these instructions:"
         "1. **Analyze Signals:** Review the Trading Signals list. Determine the primary recommendation."
         "2. **Identify Drivers:** Use the Top Feature Importance and the Macro Context (VIX/Crude Oil) to explain the prediction."
         "3. **Provide Reasoning & Citation:** Synthesize the reasons, explicitly citing the source(s) from the 'Dynamic Research' section (e.g., 'citing The Economic Times (Simulated)')."
         "4. **Give Clear Action:** Provide a clear BUY, HOLD, or SELL recommendation."
         
         "\n\n--- CURRENT AI DATA ---\n"
         "1. Trading Signals:\n{signals}\n"
         "2. Top Feature Importance (XAI):\n{importance}\n"
         "3. Macro/Geopolitical Context (Current Data):\n{geo_context}\n"
         "4. Dynamic Research (News for Citation):\n{research_context}"
        ),
        ("human", "{query}")
    ])
    
    # Create the LLM chain
    chain = template | LLM | StrOutputParser()
    
    # Invoke the chain with all data
    response = chain.invoke({
        "query": user_query,
        "signals": signals_text,
        "importance": feature_importance_text,
        "geo_context": geo_context_text,
        "geopolitical_research": research_text
    })
    
    return response

# --- 5. MAIN EXECUTION (Chat Loop) ---

if __name__ == "__main__":
    print("--- Master Trader AI Initialized ---")
    
    # Initial checks
    try:
        load_feature_importance()
        load_geopolitical_context()
        print("\nDatabase and Model connections successful. AI is online.")
    except Exception as e:
        print(f"ERROR: Database or Model not accessible. Details: {e}")
        exit()

    while True:
        user_input = input("\n[Trader] >>> ")
        if user_input.lower() == 'exit':
            print("\n[AI Master] > Session closed. Happy trading.")
            break
        
        try:
            response = generate_insights(user_input)
            
            print(f"\n[AI Master] >\n{response}")
        except Exception as e:
            # Catch errors during LLM generation or chain failure
            print(f"\n[AI Master] > Error processing request: {e}")
            break