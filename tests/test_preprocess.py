import pandas as pd
from src.preprocess import build_preprocessor

def _sample_df():
    return pd.read_csv("data/processed/heart_cleaned.csv").drop(columns=["target"]).head(20)

def test_preprocessor_fits_and_transforms():
    X = _sample_df()
    out = build_preprocessor().fit_transform(X)
    assert out.shape[0] == 20

def test_unknown_category_does_not_crash():
    X = _sample_df()
    prep = build_preprocessor().fit(X)
    X2 = X.copy()
    X2.loc[X2.index[0], "cp"] = 99   # category never seen in training
    prep.transform(X2)               # should not raise