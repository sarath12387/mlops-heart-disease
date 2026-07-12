import pandas as pd
from src.train import build_models

def _data():
    df = pd.read_csv("data/processed/heart_cleaned.csv")
    return df.drop(columns=["target"]), df["target"]

def test_pipelines_train_and_predict():
    X, y = _data()
    for name, pipe in build_models().items():
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert set(preds).issubset({0, 1})

def test_predict_proba_valid():
    X, y = _data()
    pipe = list(build_models().values())[0]
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert abs(proba[0].sum() - 1.0) < 1e-6