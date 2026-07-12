import pandas as pd
# pyrefly: ignore [missing-import]
import pytest

DATA_PATH = "data/processed/heart_cleaned.csv"

@pytest.fixture
def df():
    return pd.read_csv(DATA_PATH)

def test_no_missing_values(df):
    assert df.isna().sum().sum() == 0

def test_target_is_binary(df):
    assert set(df["target"].unique()) == {0, 1}

def test_expected_columns(df):
    expected = {"age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"}
    assert set(df.columns) == expected

def test_row_count(df):
    assert len(df) == 297