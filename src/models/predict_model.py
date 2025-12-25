import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PricePredictor:
    """Make predictions using trained model"""
    
    def __init__(self, model_path="model/model.pkl", preprocessor_path="model/preprocessor.pkl"):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.model = None
        self.preprocessor = None
        self.load_model()
    
    def load_model(self):
        """Load model and preprocessor"""
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info("Model and preprocessor loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_input(self, input_data: dict) -> np.ndarray:
        """Preprocess input data"""
        # Convert dict to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply preprocessing
        processed_data = self.preprocessor.transform(input_df)
        
        return processed_data
    
    def predict(self, input_data: dict) -> dict:
        """Make prediction"""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            
            # Get feature importance if available
            importance = {}
            if hasattr(self.model, 'feature_importances_'):
                feature_names = self.preprocessor.get_feature_names_out()
                importances = self.model.feature_importances_
                importance = dict(zip(feature_names, importances))
                # Sort by importance
                importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
            
            return {
                "prediction": float(prediction),
                "prediction_formatted": f"${prediction:,.2f}",
                "top_features": importance,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "prediction": None,
                "error": str(e),
                "status": "error"
            }
    
    def batch_predict(self, input_list: list) -> list:
        """Make batch predictions"""
        results = []
        for input_data in input_list:
            result = self.predict(input_data)
            results.append(result)
        return results

if __name__ == "__main__":
    # Test the predictor
    predictor = PricePredictor()
    
    # Sample input
    sample_input = {
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft_living": 1500,
        "sqft_lot": 4000,
        "floors": 1,
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
    
    result = predictor.predict(sample_input)
    print(f"Predicted price: {result['prediction_formatted']}")