# Dockerfile for Airflow: Builds a custom image with dependencies
FROM apache/airflow:2.8.1-python3.9

# Switch to root to perform system updates and dependency installation
USER root

# Install necessary system dependencies for PostgreSQL (libpq-dev) and building packages (gcc)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Revert to the default non-root 'airflow' user for package installation and execution
USER airflow

# Copy your custom requirements file into the container
COPY requirements.txt /requirements.txt

# Install Python packages required for the DAG to run
RUN pip install --no-cache-dir -r /requirements.txt