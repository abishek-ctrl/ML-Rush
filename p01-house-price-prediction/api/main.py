from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(
    title="House Price Predictor",
    description="GradientBoosting model trained on Ames Housing dataset",
    version="1.0.0",
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/house_price_model_tuned.pkl")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train.py first.")


class HouseFeatures(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    GarageArea: float
    TotalBsmtSF: float
    FirstFlrSF: float
    SecondFlrSF: float
    FullBath: int
    HalfBath: int
    BsmtFullBath: int
    BsmtHalfBath: int
    YearBuilt: int
    YearRemodAdd: int
    YrSold: int
    Fireplaces: int
    PoolArea: float
    GarageArea: float
    LotArea: float
    OverallCond: int
    Neighborhood: str
    BldgType: str
    HouseStyle: str

    class Config:
        json_schema_extra = {
            "example": {
                "OverallQual": 7,
                "GrLivArea": 1710.0,
                "GarageCars": 2,
                "GarageArea": 548.0,
                "TotalBsmtSF": 856.0,
                "FirstFlrSF": 856.0,
                "SecondFlrSF": 854.0,
                "FullBath": 2,
                "HalfBath": 1,
                "BsmtFullBath": 1,
                "BsmtHalfBath": 0,
                "YearBuilt": 2003,
                "YearRemodAdd": 2003,
                "YrSold": 2010,
                "Fireplaces": 0,
                "PoolArea": 0.0,
                "LotArea": 8450.0,
                "OverallCond": 5,
                "Neighborhood": "CollgCr",
                "BldgType": "1Fam",
                "HouseStyle": "2Story",
            }
        }


@app.get("/")
def root():
    return {"status": "ok", "model": "GradientBoosting House Price v1.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        data = features.model_dump()
        data["1stFlrSF"] = data.pop("FirstFlrSF")
        data["2ndFlrSF"] = data.pop("SecondFlrSF")

        df = pd.DataFrame([data])
        log_pred = model.predict(df)[0]
        price = round(np.expm1(log_pred), 2)

        return {
            "predicted_price_usd": price,
            "predicted_price_formatted": f"${price:,.2f}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
