import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def plot_feature_importance(model, feature_names=None, top_n=20, save_path=None):
    """Plot feature importance for tree-based models"""
    plt.figure(figsize=(10, 8))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Create DataFrame for sorting
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)
        
        # Plot
        plt.barh(range(len(feat_imp)), feat_imp['importance'])
        plt.yticks(range(len(feat_imp)), feat_imp['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        
    elif hasattr(model, 'coef_'):
        # For linear models
        coefficients = model.coef_
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(coefficients))]
        
        # Create DataFrame for sorting
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        })
        coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
        coef_df = coef_df.sort_values('abs_coef', ascending=True).tail(top_n)
        
        # Plot
        plt.barh(range(len(coef_df)), coef_df['coefficient'])
        plt.yticks(range(len(coef_df)), coef_df['feature'])
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {top_n} Feature Coefficients (Linear Model)')
        plt.tight_layout()
    
    else:
        print("Model doesn't have feature_importances_ or coef_ attribute")
        return
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()

def plot_residuals(y_true, y_pred, save_path=None):
    """Plot residuals for regression model"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residuals plot saved to {save_path}")
    
    plt.show()

def plot_predictions(y_true, y_pred, save_path=None):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(8, 6))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Perfect prediction line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to {save_path}")
    
    plt.show()

def plot_correlation_matrix(df, columns=None, save_path=None):
    """Plot correlation matrix for selected columns"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation matrix
    corr = df[columns].corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Visualization module loaded successfully")