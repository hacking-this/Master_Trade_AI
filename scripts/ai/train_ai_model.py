import pandas as pd
from sqlalchemy import create_engine, text
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib 
# --- Imports for the directory fix ---
import os 
from pathlib import Path 
import numpy as np

# --- 1. CONFIGURATION ---
DATABASE_URL = "postgresql://user:password@ai_stock_project-db-1:5432/stock_market_db"
engine = create_engine(DATABASE_URL)

# Correct path relative to the script's location (scripts/ai/)
MODEL_FILENAME = '../../data/models/stock_predictor_xgboost.pkl' 
print(f"Model will be saved as: {MODEL_FILENAME}")

# --- 2. LOAD & PREPARE DATA ---
def load_and_prepare_data():
    print("Loading transformed data and preparing features...")
    
    query = "SELECT * FROM transformed_stock_data ORDER BY timestamp ASC"
    
    try:
        # Load the data, assuming the table structure is clean now
        df = pd.read_sql(text(query), engine, index_col='timestamp')
    except Exception as e:
        print(f"ERROR: Failed to read data from transformed_stock_data. Make sure the table exists.")
        raise e

    # --- CRITICAL FIX: DEDUPLICATE ROWS (to handle residual errors from transformation) ---
    # We must ensure only one entry exists per day/ticker for training.
    df = df.reset_index().drop_duplicates(subset=['timestamp', 'ticker'], keep='first').set_index('timestamp')

    # --- FILTER AND DEFINE SETS ---
    
    # Exclude rows with unknown future target (-1) from training
    df_trainable = df[df['Target'] != -1].copy()
    df_predict = df[df['Target'] == -1].copy()

    # --- FEATURE EXCLUSION ---
    # We exclude unreliable indicator columns (RSI, BBANDS) and admin columns.
    # We rely on stable features: EMA_50, SMA_20, MACD, VIX_Close, BNO_Crude_Close.
    COLUMNS_TO_EXCLUDE = ['ticker', 'Future_Return', 'Target', 
                          'RSI_14', 'BBANDS_Upper', 'BBANDS_Middle', 'BBANDS_Lower']
                          
    FEATURE_COLS = df_trainable.columns.drop(COLUMNS_TO_EXCLUDE, errors='ignore')
    
    # Define feature sets for training and prediction
    X_trainable = df_trainable[FEATURE_COLS]
    y_trainable = df_trainable['Target']
    
    X_predict = df_predict[FEATURE_COLS] # Features for current-day prediction

    # --- Split for Training/Testing (Walk-Forward Simulation) ---
    split_point = int(len(X_trainable) * 0.80)
    
    X_train, X_test = X_trainable.iloc[:split_point], X_trainable.iloc[split_point:]
    y_train, y_test = y_trainable.iloc[:split_point], y_trainable.iloc[split_point:]
    
    print(f"Data split: Training on {len(X_train)} days, Testing on {len(X_test)} days.")
    print(f"Current data points ready for prediction: {len(X_predict)}.")
    print(f"Total features used: {len(FEATURE_COLS)} ({list(FEATURE_COLS)})")
    
    return X_train, X_test, y_train, y_test, X_predict

# --- 3. TRAIN AND EVALUATE MODEL ---
def train_and_save_model(X_train, X_test, y_train, y_test, X_predict):
    print("\nStarting XGBoost Classifier Training...")
    
    model = XGBClassifier(
        objective='binary:logistic', 
        n_estimators=150, 
        learning_rate=0.08,
        eval_metric='logloss', # Removed use_label_encoder=False as it's deprecated
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predict and evaluate on the test set
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    print("\n--- Model Performance on Test Data (Walk-Forward Validation) ---")
    print(f"Accuracy (Overall Correctness): {accuracy:.4f}")
    print(f"Precision (When model says BUY, how often is it right): {precision:.4f}")
    print(f"Recall (How many true BUY signals did the model find): {recall:.4f}")
    
    # --- CRITICAL FIX: ENSURE DIRECTORY EXISTS BEFORE SAVING ---
    model_path = Path(MODEL_FILENAME)
    os.makedirs(model_path.parent, exist_ok=True)
    
    # Save the trained model file
    joblib.dump(model, MODEL_FILENAME)
    print(f"\nSUCCESS: Trained model saved as: {MODEL_FILENAME}")
    
    # --- SAVE FEATURE IMPORTANCE TO DB FOR LLM REASONING ---
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'average_importance': model.feature_importances_
    }).sort_values(by='average_importance', ascending=False)
    
    feature_importance.to_sql('model_feature_importance', engine, if_exists='replace', index=False)
    print("SUCCESS: Feature importance saved to 'model_feature_importance' table for Chatbot.")
    
    return X_predict # Return data needed for prediction

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, X_predict = load_and_prepare_data()
        train_and_save_model(X_train, X_test, y_train, y_test, X_predict)
        
        print("\nAI Model Phase Complete: Ready for Dashboard and Chatbot.")
    except Exception as e:
        print(f"\n*** FATAL ERROR: AI TRAINING ABORTED ***")
        print(f"Reason: {e}")
        print("ACTION REQUIRED: Ensure 'transformed_stock_data' is fully populated.")