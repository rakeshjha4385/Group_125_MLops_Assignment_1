# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training & Experiment Tracking
# MAGIC ## Heart Disease Prediction – MLOps Assignment
# MAGIC
# MAGIC **Objective:**  
# MAGIC Train multiple classification models, compare performance, and track experiments using MLflow.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Installing required libraries

# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC %pip install databricks-sdk --upgrade

# COMMAND ----------

# MAGIC %md
# MAGIC ## Restarting the kernel

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Import required libraries

# COMMAND ----------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)

import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import seaborn as sns


# COMMAND ----------

# MAGIC %md
# MAGIC # Load cleaned dataset

# COMMAND ----------

data_path = "/Volumes/trinity_dev_rgmx/baseline_forecast/mlflow_test/heart+disease/clean_data/heart_clean.csv"
df = pd.read_csv(data_path)

X = df.drop("target", axis=1)
y = df["target"]

df.head()


# COMMAND ----------

# MAGIC %md
# MAGIC # Train-Test split

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocessing pipeline

# COMMAND ----------

numeric_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ]
)


# COMMAND ----------

# MAGIC %md
# MAGIC # Define Models

# COMMAND ----------

models = {
    "Logistic_Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Random_Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
}


# COMMAND ----------

# MAGIC %md
# MAGIC # Configure MLflow

# COMMAND ----------

mlflow.set_experiment("/Shared/Heart_Disease")

# COMMAND ----------

# MAGIC %md
# MAGIC # Train models and log experiment

# COMMAND ----------

results = []

for model_name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(pipeline, "model")

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "ROC-AUC": roc_auc
        })

        print(f"\n{model_name} Results")
        print(classification_report(y_test, y_pred))


# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Importance Plots (Random Forest)

# COMMAND ----------

# Retrieve the trained Random Forest model from MLflow run
rf_pipeline = None

for model_name, model in models.items():
    if model_name == "Random_Forest":
        rf_pipeline = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("classifier", model)
        ])
        rf_pipeline.fit(X_train, y_train)

# Extract feature importances
rf_model = rf_pipeline.named_steps["classifier"]
importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

feature_importance_df


# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance plot

# COMMAND ----------

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance_df
)
plt.title("Feature Importance – Random Forest Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Compare model performance

# COMMAND ----------

results_df = pd.DataFrame(results)
results_df


# COMMAND ----------

# MAGIC %md
# MAGIC # Visual comparison of ROC-AUC

# COMMAND ----------


results_df.set_index("Model")["ROC-AUC"].plot(kind="bar")
plt.title("ROC-AUC Comparison of Models")
plt.ylabel("ROC-AUC Score")
plt.xticks(rotation=0)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Grid Search for Random Forest

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [4, 6, 8],
    "classifier__min_samples_split": [2, 5]
}

rf_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Tuned Model

# COMMAND ----------

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Tuned Model to MLflow

# COMMAND ----------

with mlflow.start_run(run_name="Random_Forest_Tuned"):
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
    mlflow.sklearn.log_model(best_model, "model")


# COMMAND ----------

# MAGIC %md
# MAGIC # Cross validation

# COMMAND ----------

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    cv_score = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="roc_auc"
    ).mean()

    print(f"{model_name} CV ROC-AUC: {cv_score:.4f}")


# COMMAND ----------

# MAGIC %md
# MAGIC # Model selection summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Selection Summary
# MAGIC
# MAGIC - Logistic Regression served as a strong baseline model.
# MAGIC - Random Forest consistently achieved higher ROC-AUC, Precision, and Recall.
# MAGIC - Random Forest was selected for deployment due to its superior performance and ability to capture non-linear relationships.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Register Best Model

# COMMAND ----------

# Register best model manually from MLflow UI as Production
print("Register the best model in MLflow Model Registry as 'Production'")
