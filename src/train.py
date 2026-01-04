import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from data_loader import load_and_clean_data
from preprocessing import build_preprocessor
from mlflow.tracking import MlflowClient

def train():
    df = load_and_clean_data()
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42
        )
    }

    mlflow.set_experiment("Heart-Disease-MLOps")

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, preds)

            mlflow.log_param("model", name)
            mlflow.log_metric("roc_auc", roc)
            mlflow.sklearn.log_model(pipeline, "model")

            print(f"{name} ROC-AUC: {roc}")           

            client = MlflowClient()

            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"

            # Register model
            model_details = mlflow.register_model(
                model_uri=model_uri,
                name="Heart-Disease-MLOps"
            )

            # Promote to Production
            client.transition_model_version_stage(
                name="Heart-Disease-MLOps",
                version=model_details.version,
                stage="Production",
                archive_existing_versions=True
            )


if __name__ == "__main__":
    train()
