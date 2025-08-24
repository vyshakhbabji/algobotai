#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append('/Users/vyshakhbabji/Desktop/AlgoTradingBot')

from backtrader_system.strategies.enhanced_ml_signal_generator import EnhancedMLSignalGenerator
import yfinance as yf

def main():
    print("🔍 Debugging ML Training Issues...")
    
    # Download test data
    print("📊 Downloading test data...")
    data = yf.download('AAPL', start='2022-08-22', end='2024-08-22', progress=False)
    # Flatten MultiIndex columns if present - keep simple names for ML generator
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    print(f"✅ Downloaded {len(data)} rows of data")
    print(f"   Columns: {list(data.columns)}")
    
    # Initialize the ML generator
    print("🚀 Initializing ML generator...")
    config = {'logger': None}  # Simple config
    ml_gen = EnhancedMLSignalGenerator(config)
    
    # Calculate features
    print("🧮 Calculating features...")
    df_features = ml_gen.calculate_advanced_features(data.copy(), 'AAPL')
    print(f"✅ Calculated features: {len(df_features.columns)} columns")
    
    # Create targets
    print("🎯 Creating targets...")
    df_with_targets = ml_gen.create_enhanced_targets(df_features)
    print(f"✅ Created targets: {len(df_with_targets.columns)} columns")
    print(f"   All columns: {list(df_with_targets.columns)}")
    
    # Check target data types
    target_cols = ['ml_signal_strength', 'direction_target', 'regime_target', 'prob_profit']
    print("🔍 Target data types:")
    available_targets = []
    for col in target_cols:
        if col in df_with_targets.columns:
            dtype = df_with_targets[col].dtype
            non_null_count = df_with_targets[col].notna().sum()
            sample_values = df_with_targets[col].dropna().head(5).tolist()
            print(f"   {col}: {dtype} - Non-null: {non_null_count}/{len(df_with_targets)} - Sample: {sample_values}")
            if non_null_count > 0:
                available_targets.append(col)
        else:
            print(f"   {col}: NOT FOUND")
    
    # Check intermediate columns
    print("🔍 Intermediate values:")
    for col in ['future_return_5d', 'future_volatility_5d', 'risk_adjusted_return_5d']:
        if col in df_with_targets.columns:
            non_null_count = df_with_targets[col].notna().sum()
            sample_values = df_with_targets[col].dropna().head(5).tolist()
            print(f"   {col}: Non-null: {non_null_count}/{len(df_with_targets)} - Sample: {sample_values}")
    
    if not available_targets:
        print("❌ No target columns found!")
        return
    
    # Clean data
    print("🧹 Cleaning data...")
    clean_data = df_with_targets.dropna(subset=available_targets)
    print(f"✅ Clean data: {len(clean_data)} rows")
    
    if len(clean_data) < 50:
        print("❌ Insufficient clean data")
        return
    
    # Prepare features
    feature_cols = [col for col in clean_data.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] + target_cols
                   and not col.startswith('future_') and not col.startswith('risk_adjusted_')]
    
    print(f"🔧 Feature columns: {len(feature_cols)}")
    print(f"   Sample features: {feature_cols[:5]}")
    
    # Check for any string columns
    string_cols = []
    for col in feature_cols:
        if col in clean_data.columns:
            if clean_data[col].dtype == 'object':
                string_cols.append(col)
                unique_vals = clean_data[col].dropna().unique()[:5]
                print(f"   ⚠️  String column {col}: {unique_vals}")
    
    if string_cols:
        print(f"❌ Found {len(string_cols)} string columns that need fixing")
    else:
        print("✅ All feature columns are numeric")
    
    # Try simple model training
    print("🤖 Testing simple model training...")
    X = clean_data[feature_cols].fillna(0).values
    y_strength = clean_data['ml_signal_strength'].values
    y_direction = clean_data['direction_target'].values
    
    print(f"   X shape: {X.shape}")
    print(f"   y_strength shape: {y_strength.shape}, dtype: {y_strength.dtype}")
    print(f"   y_direction shape: {y_direction.shape}, dtype: {y_direction.dtype}")
    print(f"   y_direction unique: {np.unique(y_direction)}")
    
    # Test simple XGB training
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        
        print("   Testing XGBoost regressor...")
        X_train, X_test, y_train, y_test = train_test_split(X, y_strength, test_size=0.2, random_state=42)
        xgb_reg = xgb.XGBRegressor(n_estimators=10, random_state=42)
        xgb_reg.fit(X_train, y_train)
        print("   ✅ XGBoost regressor works")
        
        print("   Testing XGBoost classifier...")
        X_train, X_test, y_train, y_test = train_test_split(X, y_direction, test_size=0.2, random_state=42)
        xgb_clf = xgb.XGBClassifier(n_estimators=10, random_state=42)
        xgb_clf.fit(X_train, y_train)
        print("   ✅ XGBoost classifier works")
        
    except Exception as e:
        print(f"   ❌ Model training failed: {e}")
    
    print("🏁 Debug complete!")

if __name__ == "__main__":
    main()
