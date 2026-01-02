# Heart Disease Prediction – End-to-End MLOps

## Overview
This project demonstrates a complete MLOps lifecycle including:
- Data ingestion & EDA
- Model training & experiment tracking (MLflow)
- CI/CD with GitHub Actions
- Dockerized API
- Kubernetes deployment
- Logging & monitoring

## Setup
```bash
pip install -r requirements.txt
python src/train.py
mlflow ui
