import pandas as pd
from sqlalchemy import create_engine, text
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import joblib 
import os 
from pathlib import Path 

# --- 1. CONFIGURATION ---
DATABASE_URL = ("postgresql+psycopg2://ai_stock_trader_user:WcPpqu1IDRnqv95NoV1dUsMp17RCbTMR@dpg-d3rijvur433s73e6adeg-a.oregon-postgres.render.com/ai_stock_trader""?sslmode=require")

engine = create_engine(DATABASE_URL)
MODEL_FILENAME = '../../data/models/stock_predictor_xgboost.pkl'

# --- 2. LOAD & PREPARE DATA ---
def load_full_trainable_data():
    """Loads all historical data, filtering out current-day unknown targets."""
    print("Loading transformed data for WFO training...")
    query = "SELECT * FROM transformed_stock_data ORDER BY timestamp ASC"
    
    try:
        df = pd.read_sql(text(query), engine, index_col='timestamp')
    except Exception as e:
        print(f"ERROR: Failed to read data from transformed_stock_data. Reason: {e}")
        raise e

    # Filter out unknown targets (Target = -1)
    df_trainable = df[df['Target'] != -1].copy()
    
    # Define columns to exclude from training features
    COLUMNS_TO_EXCLUDE = ['ticker', 'Future_Return', 'Target', 
                          'RSI_14', 'BBANDS_Upper', 'BBANDS_Middle', 'BBANDS_Lower']
                          
    FEATURE_COLS = df_trainable.columns.drop(COLUMNS_TO_EXCLUDE, errors='ignore')
    
    # Final feature and target sets
    X = df_trainable[FEATURE_COLS]
    y = df_trainable['Target']
    
    return X, y

# --- 3. TRAIN AND EVALUATE MODEL (WFO Implementation) ---
def train_and_evaluate_wfo(X_full, y_full):
    """Performs Walk-Forward Optimization (WFO) and saves the final model."""
    print("\nStarting Walk-Forward Optimization (WFO) Training...")

    # WFO Parameters
    TRAIN_WINDOW = 1000  # Train on approx 4 years of data (1000 trading days)
    TEST_WINDOW = 100    # Validate/test parameters every 100 days
    
    results = [] # Store metrics from each test window

    # Initial split point for the training window
    start_point = TRAIN_WINDOW

    while start_point < len(X_full):
        # 1. Define Training Set (Fixed window size)
        X_train = X_full.iloc[start_point - TRAIN_WINDOW : start_point]
        y_train = y_full.iloc[start_point - TRAIN_WINDOW : start_point]
        
        # 2. Define Test Set (The next 100 days for out-of-sample validation)
        end_point = min(start_point + TEST_WINDOW, len(X_full))
        X_test = X_full.iloc[start_point : end_point]
        y_test = y_full.iloc[start_point : end_point]

        if X_test.empty:
            break

        # 3. Model Training (Simulating Parameter Optimization)
        model = XGBClassifier(
            objective='binary:logistic', n_estimators=150, learning_rate=0.08,
            eval_metric='logloss', random_state=42
        )
        model.fit(X_train, y_train)
        
        # 4. Evaluation and Results Storage
        y_pred = model.predict(X_test)
        
        # Store metrics from this window
        results.append({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
        })

        # Move the window forward
        start_point += TEST_WINDOW
    
    # --- Final Training and Saving ---
    
    # Train the final model on the largest possible dataset (all available data)
    final_model = XGBClassifier(
        objective='binary:logistic', n_estimators=150, learning_rate=0.08,
        eval_metric='logloss', random_state=42
    )
    final_model.fit(X_full, y_full)

    # Calculate and save metrics
    metrics_df = pd.DataFrame(results)
    
    print("\n--- Walk-Forward Validation Metrics (Averaged Across All Windows) ---")
    print(f"Average Precision: {metrics_df['precision'].mean():.4f}")
    print(f"Average Recall:    {metrics_df['recall'].mean():.4f}")
    
    # Save Model File
    model_path = Path(MODEL_FILENAME)
    os.makedirs(model_path.parent, exist_ok=True)
    joblib.dump(final_model, MODEL_FILENAME)
    print(f"\nSUCCESS: Final model saved as: {MODEL_FILENAME}")
    
    # Save Feature Importance to DB
    feature_importance = pd.DataFrame({
        'feature': X_full.columns,
        'average_importance': final_model.feature_importances_
    }).sort_values(by='average_importance', ascending=False)
    
    feature_importance.to_sql('model_feature_importance', engine, if_exists='replace', index=False)
    print("SUCCESS: Feature importance saved for Chatbot.")

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # Load all data
        X_full, y_full = load_full_trainable_data()
        
        # Perform WFO and final save
        train_and_evaluate_wfo(X_full, y_full)
        
        print("\nAI Model Optimization Complete.")
    except Exception as e:
        print(f"\n*** FATAL ERROR: AI TRAINING ABORTED ***")
        print(f"Reason: {e}")