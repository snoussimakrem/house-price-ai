from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Load the trained model
try:
    model = joblib.load('model/model.pkl')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# Create FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="Simple API for predicting house prices",
    version="1.0.0"
)

# Define input schema
class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

class PredictionResponse(BaseModel):
    prediction: float
    prediction_formatted: str
    features_used: Dict[str, Any]

@app.get("/")
async def root():
    return {
        "message": "House Price Prediction API",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "model_not_loaded",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return PredictionResponse(
            prediction=float(prediction),
            prediction_formatted=f"${prediction:,.2f}",
            features_used=features.dict()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)