import logging

from fastapi import FastAPI

from api.schemas import PatientInput, PredictionOutput
from src.predict import load_model, predict_one

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-api")

app = FastAPI(title="Heart Disease Prediction API")
model = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    result = predict_one(model, patient.model_dump())
    logger.info("prediction=%s confidence=%s", result["prediction"], result["confidence"])
    return result