from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(title="Titanic Survival Prediction API")

MODEL_PATH = "best_rf_model.pkl"

# Kiểm tra model tồn tại và load
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file '{MODEL_PATH}' not found. Please contact our team to upload model .")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")

class Passenger(BaseModel):
    Pclass: int = Field(..., example=3)
    Sex: str = Field(..., example="male")
    Age: float = Field(..., example=22)
    SibSp: int = Field(..., example=1)
    Parch: int = Field(..., example=0)
    Fare: float = Field(..., example=7.25)
    Embarked: str = Field(..., example="S")

@app.get("/")
def root():
    return {"message": "Titanic Survival Prediction API. See /docs for usage."}

@app.post("/predict", summary="Dự đoán khả năng sống sót", response_description="Kết quả dự đoán")
def predict(passenger: Passenger):
    try:
        X = pd.DataFrame([passenger.dict()])
        pred = model.predict(X)[0]
        result = "Sống sót" if pred == 1 else "Không sống sót"
        return {"Kết quả": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

