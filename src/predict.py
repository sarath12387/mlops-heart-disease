import joblib
import pandas as pd

MODEL_PATH = "models/model.pkl"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_one(model, patient: dict):
    """Predict for a single patient given as a dict of feature values."""
    df = pd.DataFrame([patient])
    prediction = int(model.predict(df)[0])
    confidence = float(model.predict_proba(df)[0][prediction])
    return {"prediction": prediction, "confidence": round(confidence, 3)}


if __name__ == "__main__":
    sample = {
        "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 240,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3,
    }
    model = load_model()
    print(predict_one(model, sample))