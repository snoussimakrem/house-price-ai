import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging

logger = logging.getLogger(__name__)

class DataCleaner(BaseEstimator, TransformerMixin):
    """Clean and prepare the data"""
    
    def __init__(self):
        self.numerical_cols = None
        self.categorical_cols = None
        
    def fit(self, X, y=None):
        # Identify column types
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Remove outliers in price (if target is in X)
        if 'price' in X.columns:
            Q1 = X['price'].quantile(0.25)
            Q3 = X['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X = X[(X['price'] >= lower_bound) & (X['price'] <= upper_bound)]
        
        # Handle missing values
        for col in self.numerical_cols:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
        
        for col in self.categorical_cols:
            if X[col].isnull().any():
                X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
        
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create new features from existing ones"""
    
    def __init__(self):
        self.feature_names = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Create new features
        if all(col in X.columns for col in ['sqft_living', 'sqft_lot']):
            X['living_to_lot_ratio'] = X['sqft_living'] / X['sqft_lot']
        
        if all(col in X.columns for col in ['bedrooms', 'bathrooms']):
            X['rooms_total'] = X['bedrooms'] + X['bathrooms']
            X['bed_bath_ratio'] = X['bedrooms'] / (X['bathrooms'] + 0.001)  # Avoid division by zero
        
        if 'yr_built' in X.columns:
            X['house_age'] = 2024 - X['yr_built']
            X['is_renovated'] = (X.get('yr_renovated', 0) > 0).astype(int)
        
        if all(col in X.columns for col in ['sqft_living', 'bedrooms']):
            X['sqft_per_bedroom'] = X['sqft_living'] / (X['bedrooms'] + 1)
        
        # Log transformation for skewed features
        skewed_features = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']
        for feature in skewed_features:
            if feature in X.columns:
                X[f'log_{feature}'] = np.log1p(X[feature])
        
        return X

def create_preprocessing_pipeline():
    """Create complete preprocessing pipeline"""
    
    # Define columns
    numerical_features = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
        'floors', 'grade', 'sqft_above', 'sqft_basement',
        'yr_built', 'lat', 'long'
    ]
    
    categorical_features = ['waterfront', 'view', 'condition']
    
    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Full pipeline with feature engineering
    full_pipeline = Pipeline([
        ('cleaner', DataCleaner()),
        ('engineer', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

def save_preprocessor(preprocessor, path="model/preprocessor.pkl"):
    """Save the preprocessor"""
    joblib.dump(preprocessor, path)
    logger.info(f"Preprocessor saved to {path}")

def load_preprocessor(path="model/preprocessor.pkl"):
    """Load the preprocessor"""
    try:
        preprocessor = joblib.load(path)
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor
    except FileNotFoundError:
        logger.error(f"Preprocessor not found at {path}")
        raise

if __name__ == "__main__":
    # Test the preprocessing
    from make_dataset import DataLoader
    
    loader = DataLoader()
    df = loader.load_data()
    
    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Create and fit pipeline
    pipeline = create_preprocessing_pipeline()
    X_processed = pipeline.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Processed shape: {X_processed.shape}")
    print(f"Sample processed data:\n{X_processed[:5]}")