import pandas as pd

def test_data_has_no_nulls():
    df = pd.read_csv("data/processed/heart_clean.csv")
    assert df.isnull().sum().sum() == 0
