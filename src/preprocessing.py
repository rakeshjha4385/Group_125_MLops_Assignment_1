from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def build_preprocessor(X):
    numeric_features = X.columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features)
        ]
    )
    return preprocessor
