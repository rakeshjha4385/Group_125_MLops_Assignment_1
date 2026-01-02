# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory Data Analysis (EDA)
# MAGIC ## Heart Disease Prediction – MLOps Assignment
# MAGIC
# MAGIC **Dataset:** UCI Heart Disease Dataset  
# MAGIC **Objective:** Understand data distribution, relationships, and class balance  
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Import Libraries

# COMMAND ----------

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


# COMMAND ----------

# MAGIC %md
# MAGIC # Load Dataset

# COMMAND ----------

import pandas as pd

base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
data_files = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data"
]

columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

dfs = []
for file in data_files:
    url = base_url + file
    df = pd.read_csv(url, names=columns)
    df.replace("?", pd.NA, inplace=True)
    df = df.apply(pd.to_numeric)
    df.dropna(inplace=True)
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Dataset Overview

# COMMAND ----------

df.info()

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC #Target Variable Distribution (Class Balance)

# COMMAND ----------

sns.countplot(x="target", data=df)
plt.title("Target Class Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Age Distribution

# COMMAND ----------

sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Cholesterol Distribution

# COMMAND ----------

sns.histplot(df["chol"], bins=20, kde=True)
plt.title("Cholesterol Level Distribution")
plt.xlabel("Serum Cholesterol (mg/dl)")
plt.ylabel("Frequency")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Correlation Heatmap

# COMMAND ----------

corr = df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=False,
    linewidths=0.5
)
plt.title("Feature Correlation Heatmap")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Feature vs Target (Key Clinical Insights)

# COMMAND ----------

features = ["age", "chol", "thalach", "trestbps"]

for feature in features:
    sns.boxplot(x="target", y=feature, data=df)
    plt.title(f"{feature} vs Heart Disease")
    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Categorical Feature Analysis

# COMMAND ----------

categorical_features = ["sex", "cp", "exang", "fbs"]

for col in categorical_features:
    sns.countplot(x=col, hue="target", data=df)
    plt.title(f"{col} vs Heart Disease")
    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA Summary
# MAGIC
# MAGIC - The dataset contains clean and well-structured clinical data.
# MAGIC - Target classes are reasonably balanced.
# MAGIC - Age, maximum heart rate, and exercise-induced angina show strong relationships with heart disease.
# MAGIC - Correlation analysis guided feature standardization and model selection.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Save Clean Dataset (Safety Step)

# COMMAND ----------

df.to_csv("/Volumes/trinity_dev_rgmx/baseline_forecast/mlflow_test/heart+disease/clean_data/heart_clean.csv", index=False)