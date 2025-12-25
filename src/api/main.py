from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn
from typing import List, Optional
import logging
import time

from src.models.predict_model import PricePredictor
from src.api.schemas import HouseFeatures, PredictionResponse, BatchPredictionRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices based on features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize predictor
predictor = PricePredictor()

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "features": "/features"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": predictor.model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    """
    Predict house price for a single property
    
    - **features**: House features (bedrooms, bathrooms, sqft_living, etc.)
    """
    try:
        logger.info(f"Received prediction request: {features.dict()}")
        
        # Make prediction
        result = predictor.predict(features.dict())
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        # Prepare response
        response = PredictionResponse(
            prediction=result["prediction"],
            prediction_formatted=result["prediction_formatted"],
            top_features=result.get("top_features", {}),
            features=features.dict(),
            timestamp=time.time()
        )
        
        logger.info(f"Prediction successful: ${result['prediction']:,.2f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch-predict")
async def batch_predict(batch_request: BatchPredictionRequest):
    """
    Predict house prices for multiple properties
    
    - **houses**: List of house features
    """
    try:
        logger.info(f"Received batch prediction request for {len(batch_request.houses)} houses")
        
        results = []
        for house in batch_request.houses:
            result = predictor.predict(house.dict())
            results.append(result)
        
        # Calculate statistics
        predictions = [r["prediction"] for r in results if r["prediction"]]
        stats = {}
        if predictions:
            stats = {
                "average_price": sum(predictions) / len(predictions),
                "min_price": min(predictions),
                "max_price": max(predictions),
                "total_houses": len(predictions)
            }
        
        return {
            "results": results,
            "statistics": stats,
            "total_processed": len(results),
            "successful": len(predictions),
            "failed": len(results) - len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/features")
async def get_feature_info():
    """Get information about required features"""
    return {
        "required_features": [
            "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
            "floors", "waterfront", "view", "condition", "grade",
            "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
            "zipcode", "lat", "long"
        ],
        "optional_features": ["sqft_living15", "sqft_lot15"],
        "feature_descriptions": {
            "bedrooms": "Number of bedrooms",
            "bathrooms": "Number of bathrooms",
            "sqft_living": "Square footage of living area",
            "sqft_lot": "Square footage of lot",
            "floors": "Number of floors",
            "waterfront": "Waterfront view (0=No, 1=Yes)",
            "view": "Quality of view (0-4)",
            "condition": "Overall condition (1-5)",
            "grade": "Overall grade (1-13)",
            "sqft_above": "Square footage above ground",
            "sqft_basement": "Square footage of basement",
            "yr_built": "Year built",
            "yr_renovated": "Year renovated (0 if never)",
            "zipcode": "Zip code",
            "lat": "Latitude",
            "long": "Longitude"
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": time.time()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )