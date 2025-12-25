import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src.data.make_dataset import DataLoader
from src.data.preprocessing import DataCleaner, FeatureEngineer

def test_data_loader():
    """Test data loading functionality"""
    # Create a test CSV
    test_data = pd.DataFrame({
        'price': [221900, 538000, 180000],
        'bedrooms': [3, 3, 2],
        'bathrooms': [1.0, 2.25, 1.0],
        'sqft_living': [1180, 2570, 770]
    })
    
    test_file = "test_data.csv"
    test_data.to_csv(test_file, index=False)
    
    try:
        loader = DataLoader(test_file)
        df = loader.load_data()
        
        assert df is not None
        assert len(df) == 3
        assert 'price' in df.columns
        
        validation = loader.validate_data()
        assert validation['total_rows'] == 3
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

def test_data_cleaner():
    """Test data cleaning"""
    cleaner = DataCleaner()
    
    # Create test data with some issues
    X = pd.DataFrame({
        'price': [100000, 200000, 300000, 1000000],  # Has outlier
        'bedrooms': [2, 3, 4, None],  # Has NaN
        'bathrooms': [1.0, 2.0, 3.0, 4.0]
    })
    
    cleaner.fit(X)
    X_clean = cleaner.transform(X)
    
    assert X_clean is not None
    assert len(X_clean) == 3  # Outlier removed
    assert X_clean['bedrooms'].isnull().sum() == 0  # NaN filled

def test_feature_engineer():
    """Test feature engineering"""
    engineer = FeatureEngineer()
    
    X = pd.DataFrame({
        'sqft_living': [1000, 2000, 3000],
        'sqft_lot': [2000, 4000, 6000],
        'bedrooms': [2, 3, 4],
        'bathrooms': [1, 2, 3],
        'yr_built': [1990, 2000, 2010]
    })
    
    engineer.fit(X)
    X_engineered = engineer.transform(X)
    
    # Check new features were created
    assert 'living_to_lot_ratio' in X_engineered.columns
    assert 'rooms_total' in X_engineered.columns
    assert 'house_age' in X_engineered.columns
    
    # Check calculations
    assert X_engineered['living_to_lot_ratio'][0] == 0.5  # 1000/2000
    assert X_engineered['rooms_total'][0] == 3  # 2+1
    assert X_engineered['house_age'][0] == 34  # 2024-1990

if __name__ == "__main__":
    pytest.main([__file__, "-v"])