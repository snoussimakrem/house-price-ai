import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import mlflow
import mlflow.sklearn
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Import our modules
from src.data.make_dataset import DataLoader
from src.data.preprocessing import create_preprocessing_pipeline, save_preprocessor
from src.visualization.visualize import plot_feature_importance, plot_residuals

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and evaluate multiple models"""
    
    def __init__(self, model_dir="model"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.best_model = None
        self.best_score = -np.inf
        self.results = {}
        
    def prepare_data(self):
        """Load and prepare data"""
        logger.info("Loading data...")
        loader = DataLoader()
        df = loader.load_data()
        
        # Separate features and target
        X = df.drop('price', axis=1)
        y = df['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and fit preprocessing pipeline
        logger.info("Creating preprocessing pipeline...")
        preprocessor = create_preprocessing_pipeline()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Save preprocessor
        save_preprocessor(preprocessor, self.model_dir / "preprocessor.pkl")
        
        return X_train_processed, X_test_processed, y_train, y_test, preprocessor
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }
        
        logger.info("Training models...")
        trained_models = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            self.results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            logger.info(f"{name}: CV R2 = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return trained_models
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models on test set"""
        logger.info("\nEvaluating models on test set...")
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            self.results[name].update({
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            })
            
            logger.info(f"\n{name}:")
            logger.info(f"  MAE: ${mae:,.2f}")
            logger.info(f"  RMSE: ${rmse:,.2f}")
            logger.info(f"  R²: {r2:.4f}")
            
            # Update best model
            if r2 > self.best_score:
                self.best_score = r2
                self.best_model = model
                self.best_model_name = name
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning on best model"""
        logger.info("\nPerforming hyperparameter tuning...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            rf, param_grid, n_iter=20, cv=5, 
            scoring='r2', random_state=42, n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def save_best_model(self):
        """Save the best model"""
        if self.best_model is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.model_dir / f"best_model_{timestamp}.pkl"
            joblib.dump(self.best_model, model_path)
            
            # Also save as latest
            latest_path = self.model_dir / "model.pkl"
            joblib.dump(self.best_model, latest_path)
            
            logger.info(f"Best model ({self.best_model_name}) saved to {model_path}")
            logger.info(f"Also saved as latest to {latest_path}")
    
    def track_with_mlflow(self, X_test, y_test):
        """Track experiment with MLflow"""
        logger.info("\nTracking with MLflow...")
        
        mlflow.set_experiment("House Price Prediction")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", self.best_model_name)
            mlflow.log_param("best_score", self.best_score)
            
            # Log metrics
            y_pred = self.best_model.predict(X_test)
            mlflow.log_metric("test_r2", r2_score(y_test, y_pred))
            mlflow.log_metric("test_rmse", np.sqrt(mean_squared_error(y_test, y_pred)))
            mlflow.log_metric("test_mae", mean_absolute_error(y_test, y_pred))
            
            # Log model
            mlflow.sklearn.log_model(self.best_model, "model")
            
            # Log artifacts
            mlflow.log_artifact(str(self.model_dir / "preprocessor.pkl"))
            
            logger.info("Experiment logged to MLflow")
    
    def run(self, use_mlflow=True):
        """Run complete training pipeline"""
        logger.info("Starting model training pipeline...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, preprocessor = self.prepare_data()
        
        # Train models
        models = self.train_models(X_train, y_train)
        
        # Evaluate models
        self.evaluate_models(models, X_test, y_test)
        
        # Hyperparameter tuning (optional)
        tuned_model = self.hyperparameter_tuning(X_train, y_train)
        models['tuned_rf'] = tuned_model
        self.evaluate_models({'tuned_rf': tuned_model}, X_test, y_test)
        
        # Save best model
        self.save_best_model()
        
        # Track with MLflow
        if use_mlflow:
            self.track_with_mlflow(X_test, y_test)
        
        # Generate visualizations
        plot_feature_importance(self.best_model, X_train)
        plot_residuals(y_test, self.best_model.predict(X_test))
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"Best model: {self.best_model_name}")
        logger.info(f"Best R² score: {self.best_score:.4f}")
        logger.info("="*50)
        
        return self.best_model, self.results

if __name__ == "__main__":
    # Initialize MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Run training
    trainer = ModelTrainer()
    best_model, results = trainer.run()