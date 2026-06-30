from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(title="ChurnSense Prediction API", description="REST API for predicting telecom customer churn", version="1.0.0")

MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/best_model.pkl"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"

# Check if model exists before starting
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("Warning: Model or scaler not found. Please run pipeline.py first.")
    model, scaler = None, None
else:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

class CustomerData(BaseModel):
    tenure: int = Field(..., description="Number of months the customer has stayed with the company")
    MonthlyCharges: float = Field(..., description="The amount charged to the customer monthly")
    TotalCharges: float = Field(..., description="The total amount charged to the customer")
    Contract: int = Field(..., description="0: Month-to-month, 1: One year, 2: Two year")
    InternetService: int = Field(..., description="0: DSL, 1: Fiber optic, 2: No")
    PaymentMethod: int = Field(..., description="0: Electronic check, 1: Mailed check, 2: Bank transfer, 3: Credit card")
    TechSupport: int = Field(..., description="0: No, 1: Yes, 2: No internet service")
    SeniorCitizen: int = Field(..., description="0: No, 1: Yes")
    Dependents: int = Field(..., description="0: No, 1: Yes")
    NumServices: int = Field(..., description="Number of add-on services")

@app.get("/")
def home():
    return {"message": "Welcome to the ChurnSense Prediction API"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Please train the model first.")

    try:
        # Convert pydantic model to dict
        cust_dict = customer.dict()
        
        # Make a dataframe
        X = pd.DataFrame([cust_dict])
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]
        
        return {
            "churn_prediction": "CHURN" if pred == 1 else "RETAINED",
            "churn_probability": round(float(proba), 4),
            "risk_level": "HIGH" if proba > 0.7 else "MEDIUM" if proba > 0.4 else "LOW"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
