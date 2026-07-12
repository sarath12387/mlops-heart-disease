import os
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]


def download_and_clean():
    df = pd.read_csv(URL, header=None, names=COLS, na_values="?")

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/heart_disease_raw.csv", index=False)

    print("Missing values:\n", df.isna().sum())
    df = df.dropna()
    df["target"] = (df["target"] > 0).astype(int)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/heart_cleaned.csv", index=False)
    print(f"Cleaned dataset saved: {df.shape[0]} rows")
    return df


if __name__ == "__main__":
    download_and_clean()