from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Titanic Survival Predictor",
    description="Voting Ensemble model classifying survival on the Titanic",
    version="1.0.0",
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/titanic_model.pkl")
THRESH_PATH = os.getenv("THRESHOLD_PATH", "models/titanic_threshold.pkl")

try:
    model = joblib.load(MODEL_PATH)
    threshold = float(joblib.load(THRESH_PATH))
except FileNotFoundError:
    # Model might not be trained yet
    model = None
    threshold = 0.5


class PassengerFeatures(BaseModel):
    Pclass: int
    Name: str
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str | None = None
    Embarked: str

    class Config:
        json_schema_extra = {
            "example": {
                "Pclass": 3,
                "Name": "Braund, Mr. Owen Harris",
                "Sex": "male",
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Ticket": "A/5 21171",
                "Fare": 7.25,
                "Cabin": None,
                "Embarked": "S"
            }
        }


@app.get("/")
def root():
    return {"status": "ok", "model": "Titanic Voting Ensemble v1.0"}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
    return {"status": "healthy"}


@app.post("/predict")
def predict(features: PassengerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
        
    try:
        from src.features import engineer_features
        data = features.model_dump()
        df = pd.DataFrame([data])
        
        # apply feature engineering on the incoming row
        df, _ = engineer_features(df, df.copy())
        
        prob = model.predict_proba(df)[0][1]
        label = int(prob >= threshold)
        
        return {
            "survived": label,
            "probability": round(float(prob), 4),
            "verdict": "Survived" if label == 1 else "Did not survive",
            "threshold_used": round(float(threshold), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
