import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report

def evaluate(run_id, data_path="data/processed/heart_clean.csv"):
    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    preds = model.predict(X)
    print(classification_report(y, preds))
