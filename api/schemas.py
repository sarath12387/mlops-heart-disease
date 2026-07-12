from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int
    trestbps: float
    chol: float
    fbs: int = Field(..., ge=0, le=1)
    restecg: int
    thalach: float
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float
    slope: int
    ca: int
    thal: int


class PredictionOutput(BaseModel):
    prediction: int
    confidence: float