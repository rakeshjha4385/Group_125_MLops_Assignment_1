from sklearn.ensemble import RandomForestClassifier

def test_model_creation():
    model = RandomForestClassifier()
    assert model is not None
