import pandas as pd
import os

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

def load_and_clean_data():
    df = pd.read_csv(URL, names=COLUMNS)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)
    df["target"] = df["target"].astype(int).apply(lambda x: 1 if x > 0 else 0)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/heart_clean.csv", index=False)
    return df

if __name__ == "__main__":
    load_and_clean_data()
