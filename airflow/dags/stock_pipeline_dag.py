from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

# --- CONFIGURATION ---
# The path where your project root is mounted inside the Docker container
PROJECT_ROOT = "/usr/local/airflow/project_root" 

# CRITICAL FIX: Use the system Python executable inside the container
PYTHON_EXECUTABLE = "python3" 

# --- 1. DEFINE THE DAG (PIPELINE SCHEDULE) ---
with DAG(
    dag_id="daily_stock_market_pipeline",
    start_date=pendulum.datetime(2025, 10, 1, tz="UTC"),
    # Schedule: Run daily at 19:00 UTC (12:30 AM IST for next day's data)
    schedule="0 19 * * *", 
    catchup=False,
    tags=["finance", "mlops", "daily"],
    default_args={
        "owner": "airflow",
        "email_on_failure": False,
        "retries": 1,
    }
) as dag:
    
    # --- 2. DEFINE THE TASKS (STEPS) ---
    
    # Task 1: Ingest Raw Data (Initial 5-year load or daily increment via watermarking)
    t1_ingest_raw_data = BashOperator(
        task_id="ingest_raw_data",
        # Command: Execute the script using the stable PYTHON_EXECUTABLE
        bash_command=f"{PYTHON_EXECUTABLE} {PROJECT_ROOT}/scripts/ingestion/ingest_data.py",
        cwd=f"{PROJECT_ROOT}"
    )

    # Task 2: Transform Data & Engineer Features (Calculate indicators, VIX/BNO merge)
    t2_feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"{PYTHON_EXECUTABLE} {PROJECT_ROOT}/scripts/features/transform_features.py",
        cwd=f"{PROJECT_ROOT}"
    )

    # Task 3: Train/Retrain AI Model (Generate new predictions and update model file)
    t3_train_ai_model = BashOperator(
        task_id="train_ai_model",
        bash_command=f"{PYTHON_EXECUTABLE} {PROJECT_ROOT}/scripts/ai/train_ai_model.py",
        cwd=f"{PROJECT_ROOT}"
    )

    # --- 3. SET THE TASK ORDER (DEPENDENCIES) ---
    t1_ingest_raw_data >> t2_feature_engineering >> t3_train_ai_model