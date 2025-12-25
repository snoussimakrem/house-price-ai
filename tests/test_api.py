import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src.api.main import app
from src.api.schemas import HouseFeatures

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "House Price Prediction API" in data["message"]

def test_features_endpoint():
    """Test features information endpoint"""
    response = client.get("/features")
    assert response.status_code == 200
    data = response.json()
    assert "required_features" in data
    assert "bedrooms" in data["required_features"]

def test_predict_endpoint():
    """Test prediction endpoint with valid data"""
    test_data = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1500,
        "sqft_lot": 4000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1500,
        "sqft_basement": 0,
        "yr_built": 1995,
        "yr_renovated": 0,
        "zipcode": 98178,
        "lat": 47.5112,
        "long": -122.257
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "prediction_formatted" in data
    assert data["prediction"] > 0

def test_predict_invalid_data():
    """Test prediction with invalid data"""
    test_data = {
        "bedrooms": -1,  # Invalid: negative bedrooms
        "bathrooms": 2.0,
        "sqft_living": 1500,
        # Missing other required fields
    }
    
    response = client.post("/predict", json=test_data)
    # Should return 422 for validation error
    assert response.status_code == 422

def test_batch_predict():
    """Test batch prediction"""
    test_data = {
        "houses": [
            {
                "bedrooms": 3,
                "bathrooms": 2.0,
                "sqft_living": 1500,
                "sqft_lot": 4000,
                "floors": 1.0,
                "waterfront": 0,
                "view": 0,
                "condition": 3,
                "grade": 7,
                "sqft_above": 1500,
                "sqft_basement": 0,
                "yr_built": 1995,
                "yr_renovated": 0,
                "zipcode": 98178,
                "lat": 47.5112,
                "long": -122.257
            },
            {
                "bedrooms": 4,
                "bathrooms": 2.5,
                "sqft_living": 2200,
                "sqft_lot": 7500,
                "floors": 2.0,
                "waterfront": 0,
                "view": 2,
                "condition": 4,
                "grade": 8,
                "sqft_above": 2200,
                "sqft_basement": 0,
                "yr_built": 2005,
                "yr_renovated": 0,
                "zipcode": 98105,
                "lat": 47.6616,
                "long": -122.313
            }
        ]
    }
    
    response = client.post("/batch-predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert "statistics" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])