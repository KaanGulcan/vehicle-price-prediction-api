from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="Vehicle Price Prediction API",
    description="This api project can predict vehicle prices with linear regression and xgboost models",
    version="1.0"
)

xgboost_model = joblib.load("../models/xgboost_model.pkl")
linRegr_model = joblib.load("../models/linRegr_model.pkl")

class ModelInput(BaseModel):
    brand: str
    model: str
    year: int
    km_driven: float = 0
    engine: float
    seats: float = 5
    fuel: str = "Petrol"
    transmission: str = "Manuel"
    owner: str = "First Owner"
    seller_type: str = "Individual"

@app.get("/",deprecated=True)
def homePage():
    return {"message": "API is working! You can go /docs adress to predict"}

@app.post("/predict/xgboost", description="This endpoint predict vehicle prices with XGBoost model")
def predict_vehicle_price(data: ModelInput):
    new_data = pd.DataFrame([data.model_dump()])
    prediction = xgboost_model.predict(new_data)
    
    return {"Vehicle_Price": round(float(np.expm1(prediction[0])),2)}

@app.post("/predict/linear_regression", description="This endpoint predict vehicle prices with Linear Regression model")
def predict_vehicle_price(data: ModelInput):
    new_data = pd.DataFrame([data.model_dump()])
    prediction = linRegr_model.predict(new_data)

    return {"Vehicle_Price": round(np.expm1(prediction[0]),2)}