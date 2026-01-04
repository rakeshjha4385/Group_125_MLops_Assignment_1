from fastapi import FastAPI
import pandas as pd
import mlflow.sklearn
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

MODEL_URI = "models:/Heart-Disease-MLOps/Production"

# Load model once at startup
model = mlflow.sklearn.load_model(MODEL_URI)

# MUST match training features exactly
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

@app.post("/predict")
def predict(features: dict):
    logging.info("Prediction request received")

    # #  THIS LINE FIXES EVERYTHING
    # X = pd.DataFrame([features])

    # # Enforce correct column order
    # X = X[FEATURE_COLUMNS]

    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return {
        "prediction": int(pred),
        "confidence": float(prob)
    }
