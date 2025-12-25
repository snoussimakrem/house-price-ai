import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("HOUSE PRICE PREDICTION MODEL TRAINING")
print("=" * 60)

# Create necessary directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('model', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)

# Check if data exists
data_path = 'data/raw/house_data.csv'
if not os.path.exists(data_path):
    print("Creating sample dataset...")
    np.random.seed(42)
    n_samples = 500
    
    # Generate realistic house data WITHOUT id and date columns
    data = {
        'price': np.random.normal(500000, 200000, n_samples).astype(int),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0], n_samples),
        'sqft_living': np.random.randint(800, 4000, n_samples),
        'sqft_lot': np.random.randint(1000, 15000, n_samples),
        'floors': np.random.choice([1.0, 1.5, 2.0, 2.5], n_samples),
        'waterfront': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'view': np.random.randint(0, 5, n_samples),
        'condition': np.random.randint(1, 6, n_samples),
        'grade': np.random.randint(4, 11, n_samples),
        'yr_built': np.random.randint(1950, 2020, n_samples),
        'zipcode': np.random.randint(98001, 98200, n_samples),
        'lat': np.random.uniform(47.0, 48.0, n_samples),
        'long': np.random.uniform(-123.0, -121.0, n_samples),
    }
    
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)
    print(f"✓ Created sample data with {n_samples} rows")
else:
    print("✓ Data file found")

# Load data
print("\n📊 Loading and preparing data...")
df = pd.read_csv(data_path)
print(f"   Original dataset shape: {df.shape}")
print(f"   Original columns: {', '.join(df.columns.tolist())}")

# REMOVE NON-NUMERICAL COLUMNS - FIX FOR THE DATE ISSUE
print("\n🔧 Cleaning data...")
# List columns to drop (non-numerical or non-predictive)
columns_to_drop = ['id', 'date']

# Find which columns actually exist in our dataframe
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

if existing_columns_to_drop:
    print(f"   Dropping columns: {', '.join(existing_columns_to_drop)}")
    df = df.drop(columns=existing_columns_to_drop, errors='ignore')

# Check for non-numeric columns and convert them
print("\n🔍 Checking data types...")
non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    print(f"   Found non-numeric columns: {non_numeric_cols}")
    print("   Attempting to convert to numeric...")
    
    # Try to convert each non-numeric column
    for col in non_numeric_cols:
        # Check what type of data we have
        sample_values = df[col].dropna().head(3).tolist()
        print(f"     Column '{col}' sample values: {sample_values}")
        
        # Try to convert to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # After conversion, check again
    still_non_numeric = df.select_dtypes(include=['object']).columns.tolist()
    if still_non_numeric:
        print(f"   Still non-numeric after conversion: {still_non_numeric}")
        print("   Dropping these columns...")
        df = df.drop(columns=still_non_numeric)

print(f"\n✅ Cleaned dataset shape: {df.shape}")
print(f"   Cleaned columns: {', '.join(df.columns.tolist())}")

# Check for missing values
print("\n🔍 Checking for missing values...")
missing_before = df.isnull().sum()
if missing_before.any():
    print(f"   Missing values found BEFORE cleaning:")
    for col, count in missing_before[missing_before > 0].items():
        print(f"     - {col}: {count} missing ({count/len(df)*100:.1f}%)")
    
    # Fill missing values
    print("\n   Handling missing values...")
    
    # For numerical columns, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"     - {col}: filled {df[col].isnull().sum()} NaN with median {median_val:.2f}")
    
    # Check again after filling
    missing_after = df.isnull().sum()
    if missing_after.any():
        print(f"\n   ⚠️  Still missing values after filling:")
        for col, count in missing_after[missing_after > 0].items():
            print(f"     - {col}: {count} missing")
        print("   Dropping rows with remaining missing values...")
        df = df.dropna()
else:
    print("   ✓ No missing values found")

print(f"\n✅ Final dataset shape: {df.shape}")

# Check if we have enough data
if len(df) < 50:
    print(f"\n⚠️  WARNING: Small dataset ({len(df)} samples)! Creating synthetic data...")
    
    # Check what features we have
    available_features = df.columns.tolist()
    if 'price' in available_features:
        available_features.remove('price')
    
    print(f"   Available features: {available_features}")
    
    # Create synthetic data that matches our existing features
    np.random.seed(42)
    n_samples = 500 - len(df)  # Add enough to reach 500
    
    synthetic_data = {}
    
    # For each feature, create synthetic data
    for feature in available_features:
        if feature in df.columns:
            # Get statistics from existing data
            if df[feature].dtype in [np.int64, np.int32]:
                # Integer feature
                min_val = max(0, int(df[feature].min() * 0.8))
                max_val = int(df[feature].max() * 1.2)
                synthetic_data[feature] = np.random.randint(min_val, max_val + 1, n_samples)
            else:
                # Float feature
                mean_val = df[feature].mean()
                std_val = df[feature].std()
                if std_val == 0:
                    std_val = mean_val * 0.1 if mean_val != 0 else 1.0
                synthetic_data[feature] = np.random.normal(mean_val, std_val, n_samples)
                # Ensure no negative values for certain features
                if feature in ['sqft_living', 'sqft_lot', 'sqft_above']:
                    synthetic_data[feature] = np.abs(synthetic_data[feature])
    
    # Add price column
    if 'price' not in synthetic_data:
        price_mean = df['price'].mean() if 'price' in df.columns else 500000
        price_std = df['price'].std() if 'price' in df.columns else 150000
        synthetic_data['price'] = np.random.normal(price_mean, price_std, n_samples)
        synthetic_data['price'] = np.abs(synthetic_data['price'])  # Ensure positive prices
    
    # Create synthetic DataFrame
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # Combine with original data
    df = pd.concat([df, synthetic_df], ignore_index=True)
    print(f"   ✓ Added {n_samples} synthetic samples")
    print(f"   ✓ New dataset size: {len(df)}")

# Prepare features and target
print("\n🔧 Preparing features and target...")
# Make sure 'price' column exists
if 'price' not in df.columns:
    print("❌ ERROR: 'price' column not found in data!")
    print(f"   Available columns: {df.columns.tolist()}")
    exit(1)

X = df.drop('price', axis=1)
y = df['price']

print(f"   Number of features: {X.shape[1]}")
print(f"   Target (price) statistics:")
print(f"     - Mean: ${y.mean():,.2f}")
print(f"     - Min:  ${y.min():,.2f}")
print(f"     - Max:  ${y.max():,.2f}")

# Final check for NaN values
print("\n🔍 Final data check...")
if X.isnull().any().any():
    print("   ⚠️  Found NaN in features! Using imputer...")
    # Use SimpleImputer to handle any remaining NaN
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)
    print("   ✓ Imputed remaining NaN values")
else:
    print("   ✓ No NaN values in features")

if y.isnull().any():
    print("   ⚠️  Found NaN in target! Removing rows...")
    # Remove rows where target is NaN
    valid_indices = ~y.isnull()
    X = X[valid_indices]
    y = y[valid_indices]
    print(f"   ✓ Removed {sum(~valid_indices)} rows with NaN target")

print(f"\n✅ Final dataset ready for training:")
print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")

# Split data
test_size = 0.2
print(f"\n📊 Splitting data (test_size={test_size})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, shuffle=True
)
print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# Train models
print("\n🤖 Training models...")
models = {}

# 1. Random Forest (more robust to data issues)
print("   1. Training Random Forest...")
try:
    # Adjust parameters for the dataset size
    n_estimators = min(100, len(X_train))
    max_depth = min(10, max(3, len(X_train) // 20))
    
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    print(f"      Parameters: {n_estimators} trees, max_depth={max_depth}")
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    print("      ✓ Random Forest trained successfully")
except Exception as e:
    print(f"      ✗ Failed to train Random Forest: {e}")
    print(f"      Error details: {str(e)[:200]}...")

# 2. Try Gradient Boosting if RandomForest fails
if not models:
    print("\n   2. Trying Gradient Boosting as fallback...")
    from sklearn.ensemble import GradientBoostingRegressor
    try:
        gb = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=42,
            learning_rate=0.1
        )
        gb.fit(X_train, y_train)
        models['gradient_boosting'] = gb
        print("      ✓ Gradient Boosting trained successfully")
    except Exception as e:
        print(f"      ✗ Failed to train Gradient Boosting: {e}")

# Check if any models were trained
if not models:
    print("\n❌ CRITICAL: No models could be trained!")
    print("   This usually indicates data issues. Let's debug:")
    
    print("\n   Debugging data issues:")
    print(f"   - X_train shape: {X_train.shape}")
    print(f"   - X_train has NaN: {X_train.isnull().any().any()}")
    print(f"   - X_train has Inf: {np.any(np.isinf(X_train.values))}")
    print(f"   - y_train shape: {y_train.shape}")
    print(f"   - y_train has NaN: {y_train.isnull().any()}")
    print(f"   - y_train has Inf: {np.any(np.isinf(y_train.values))}")
    
    # Show sample of problematic data
    print(f"\n   Sample of X_train (first 3 rows):")
    print(X_train.head(3))
    print(f"\n   Sample of y_train (first 3 values):")
    print(y_train.head(3))
    
    exit(1)

# Evaluate models
print("\n📈 Evaluating models...")
print("-" * 50)

best_model = None
best_score = -np.inf
results = {}

for name, model in models.items():
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model': model
        }
        
        # Print results
        print(f"\n{name.upper().replace('_', ' ')}:")
        print(f"  MAE:  ${mae:,.2f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  R²:   {r2:.4f}")
        
        # Update best model
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name
    except Exception as e:
        print(f"\n✗ Failed to evaluate {name}: {e}")

if best_model is not None:
    print("\n" + "=" * 50)
    print(f"🏆 BEST MODEL: {best_model_name.upper().replace('_', ' ')}")
    print(f"   R² Score: {best_score:.4f}")
    print("=" * 50)
else:
    print("\n❌ No valid model evaluation available")
    exit(1)

# Save the best model
print("\n💾 Saving model...")
model_path = 'model/model.pkl'
try:
    joblib.dump(best_model, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Also save the imputer if we used one
    if 'imputer' in locals():
        imputer_path = 'model/imputer.pkl'
        joblib.dump(imputer, imputer_path)
        print(f"✓ Imputer saved to: {imputer_path}")
except Exception as e:
    print(f"✗ Failed to save model: {e}")

# Save preprocessing info
print("\n📝 Saving model metadata...")
preprocessing_info = {
    'features': X.columns.tolist(),
    'target': 'price',
    'model_type': best_model_name,
    'performance': {
        'r2': float(results[best_model_name]['r2']),
        'mae': float(results[best_model_name]['mae']),
        'rmse': float(results[best_model_name]['rmse'])
    },
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_size': len(df),
    'test_size': test_size,
    'feature_count': X.shape[1]
}

info_path = 'model/model_info.json'
try:
    import json
    with open(info_path, 'w') as f:
        json.dump(preprocessing_info, f, indent=4)
    print(f"✓ Model info saved to: {info_path}")
except Exception as e:
    print(f"✗ Failed to save model info: {e}")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    print("\n🔍 Feature Importance (Top 10):")
    importances = best_model.feature_importances_
    feature_names = X.columns
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(10)
    
    print(importance_df.to_string(index=False))
    
    # Save feature importance
    importance_path = 'model/feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved to: {importance_path}")

# Make sample prediction
print("\n🔮 Sample Prediction Demo:")
if len(X_test) > 0:
    sample_idx = 0
    sample_house = X_test.iloc[sample_idx:sample_idx+1]
    predicted_price = best_model.predict(sample_house)[0]
    actual_price = y_test.iloc[sample_idx]
    
    print(f"   Predicted Price: ${predicted_price:,.2f}")
    print(f"   Actual Price:    ${actual_price:,.2f}")
    print(f"   Difference:      ${abs(predicted_price - actual_price):,.2f}")
    
    # Calculate percentage error
    if actual_price != 0:
        pct_error = abs(predicted_price - actual_price) / actual_price * 100
        print(f"   Error:           {pct_error:.1f}%")
    
    print("\n   Sample house features:")
    for i, (feature, value) in enumerate(sample_house.iloc[0].items()):
        if i < 8:  # Show first 8 features
            print(f"     - {feature}: {value:.2f}")
    if len(sample_house.columns) > 8:
        print(f"     ... and {len(sample_house.columns) - 8} more features")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\n📊 TRAINING SUMMARY:")
print("-" * 40)
print(f"Dataset size: {len(df)} samples")
print(f"Features used: {X.shape[1]}")
print(f"Models trained: {len(models)}")
print(f"Best model: {best_model_name}")
print(f"Best R² score: {best_score:.4f}")
print(f"Test set size: {len(X_test)} samples")
print("-" * 40)

print("\n🚀 Next steps:")
print("1. Start API: uvicorn src.api.main:app --reload")
print("2. Start Dashboard: streamlit run app/dashboard.py")
print("3. Test API: curl -X POST http://localhost:8000/predict")
print("4. View predictions: http://localhost:8501")

print("\n" + "=" * 60)
print("✨ Your house price prediction model is ready!")
print("=" * 60)