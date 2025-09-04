#!/bin/bash

# Install Python dependencies
cd /workspace/python-ml
pip install -e .

# Initialize MLflow
mlflow server --backend-store-uri file:///mlflow --default-artifact-root file:///mlflow/artifacts --host 0.0.0.0 --port 5000 &

# Download sample data if not exists
if [ ! -f "/data/movies.csv" ]; then
    echo "Downloading sample movie dataset..."
    python scripts/download_data.py
fi

# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/workspace &

echo "ML environment initialized!"
echo "- MLflow UI: http://localhost:5000"
echo "- Jupyter Lab: http://localhost:8888"