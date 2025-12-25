import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and validate house price data"""
    
    def __init__(self, data_path: str = "data/raw/house_data.csv"):
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self) -> dict:
        """Validate data quality"""
        validation_report = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": self.df.duplicated().sum(),
            "data_types": self.df.dtypes.to_dict()
        }
        
        # Check for required columns
        required_columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        
        if missing_cols:
            validation_report["missing_required_columns"] = missing_cols
            logger.warning(f"Missing required columns: {missing_cols}")
        
        return validation_report
    
    def get_sample(self, n: int = 5) -> pd.DataFrame:
        """Get a sample of the data"""
        return self.df.sample(n) if self.df is not None else None

if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    df = loader.load_data()
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())