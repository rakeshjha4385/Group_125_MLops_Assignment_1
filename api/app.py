from fastapi import FastAPI
import mlflow.sklearn
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

MODEL_URI = "models:/Heart-Disease-MLOps/Production"
model = mlflow.sklearn.load_model(MODEL_URI)

@app.post("/predict")
def predict(features: dict):
    logging.info("Prediction request received")
    X = [list(features.values())]
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return {
        "prediction": int(pred),
        "confidence": float(prob)
    }
