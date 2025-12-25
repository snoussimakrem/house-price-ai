from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class HouseFeatures(BaseModel):
    """Schema for house features input"""
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0.5, le=8, description="Number of bathrooms")
    sqft_living: int = Field(..., ge=500, le=10000, description="Square footage of living area")
    sqft_lot: int = Field(..., ge=500, le=100000, description="Square footage of lot")
    floors: float = Field(..., ge=1, le=3.5, description="Number of floors")
    waterfront: int = Field(0, ge=0, le=1, description="Waterfront view (0=No, 1=Yes)")
    view: int = Field(0, ge=0, le=4, description="Quality of view (0-4)")
    condition: int = Field(..., ge=1, le=5, description="Overall condition (1-5)")
    grade: int = Field(..., ge=1, le=13, description="Overall grade (1-13)")
    sqft_above: int = Field(..., ge=500, le=10000, description="Square footage above ground")
    sqft_basement: int = Field(0, ge=0, le=5000, description="Square footage of basement")
    yr_built: int = Field(..., ge=1900, le=2024, description="Year built")
    yr_renovated: int = Field(0, ge=0, le=2024, description="Year renovated (0 if never)")
    zipcode: int = Field(..., ge=98001, le=98199, description="Zip code")
    lat: float = Field(..., ge=47.0, le=48.0, description="Latitude")
    long: float = Field(..., ge=-123.0, le=-121.0, description="Longitude")
    sqft_living15: Optional[int] = Field(None, description="Living area of 15 nearest neighbors")
    sqft_lot15: Optional[int] = Field(None, description="Lot area of 15 nearest neighbors")
    
    @validator('yr_renovated')
    def validate_renovation_year(cls, v, values):
        if 'yr_built' in values and v > 0:
            if v < values['yr_built']:
                raise ValueError('Renovation year must be after built year')
        return v
    
    @validator('sqft_basement')
    def validate_basement(cls, v, values):
        if 'sqft_above' in values and v > values['sqft_above']:
            raise ValueError('Basement cannot be larger than above ground area')
        return v

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    prediction: float = Field(..., description="Predicted price")
    prediction_formatted: str = Field(..., description="Formatted price with currency")
    top_features: Dict[str, float] = Field(default_factory=dict, description="Top important features")
    features: Dict[str, Any] = Field(..., description="Input features")
    timestamp: float = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 550000.0,
                "prediction_formatted": "$550,000.00",
                "top_features": {
                    "sqft_living": 0.35,
                    "grade": 0.25,
                    "location": 0.20
                },
                "features": {
                    "bedrooms": 3,
                    "bathrooms": 2.0,
                    "sqft_living": 1500
                },
                "timestamp": 1672531200.0
            }
        }

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request"""
    houses: List[HouseFeatures] = Field(..., description="List of houses to predict")
    
    @validator('houses')
    def validate_houses_length(cls, v):
        if len(v) > 100:
            raise ValueError('Batch size cannot exceed 100 houses')
        return v

class ModelInfo(BaseModel):
    """Schema for model information"""
    model_type: str
    training_date: str
    performance: Dict[str, float]
    features_used: List[str]
    version: str

class APIStatus(BaseModel):
    """Schema for API status"""
    status: str
    version: str
    uptime: float
    model_loaded: bool
    requests_served: int