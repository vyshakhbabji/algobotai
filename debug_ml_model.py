#!/usr/bin/env python3
"""
Debug the ML model training and prediction process
"""

import pandas as pd
import numpy as np
from pathlib import Path

def debug_ml_model():
    """
    Test if the ML model can be trained and produce predictions for NVDA
    """
    
    # Try to import the ML components
    try:
        from algobot.ml.features import FeatureEngineer
        from algobot.ml.model import MLModel
        print("✓ Successfully imported ML components")
    except Exception as e:
        print(f"✗ Error importing ML components: {e}")
        return
    
    # Get NVDA data
    try:
        import yfinance as yf
        print("\n1. Downloading NVDA data...")
        
        # Get training data (2 years) and test data (3 months)
        train_start = "2022-05-13"
        train_end = "2024-05-13"
        test_start = "2024-05-13"
        test_end = "2024-08-13"
        
        train_data = yf.download('NVDA', start=train_start, end=train_end, progress=False)
        test_data = yf.download('NVDA', start=test_start, end=test_end, progress=False)
        
        print(f"   Training data: {len(train_data)} days ({train_data.index[0]} to {train_data.index[-1]})")
        print(f"   Test data: {len(test_data)} days ({test_data.index[0]} to {test_data.index[-1]})")
        
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        return
    
    # Test feature engineering
    try:
        print("\n2. Testing feature engineering...")
        fe = FeatureEngineer()
        
        # Prepare training features
        train_features = fe.prepare_features(train_data)
        print(f"   Training features shape: {train_features.shape}")
        print(f"   Feature columns: {list(train_features.columns)[:10]}...")  # Show first 10
        
        # Check for NaN values
        nan_count = train_features.isnull().sum().sum()
        print(f"   NaN values in features: {nan_count}")
        
        if nan_count > 0:
            print("   Dropping NaN values...")
            train_features = train_features.dropna()
            print(f"   Features after dropping NaN: {train_features.shape}")
        
    except Exception as e:
        print(f"✗ Error in feature engineering: {e}")
        return
    
    # Test model training
    try:
        print("\n3. Testing ML model training...")
        model = MLModel()
        
        # Create target variable (forward returns)
        target = train_data['Close'].pct_change(5).shift(-5)  # 5-day forward return
        target = target.reindex(train_features.index).dropna()
        
        # Align features and target
        common_idx = train_features.index.intersection(target.index)
        train_features_aligned = train_features.loc[common_idx]
        target_aligned = target.loc[common_idx]
        
        print(f"   Aligned training data: {len(train_features_aligned)} samples")
        print(f"   Target statistics: mean={target_aligned.mean():.4f}, std={target_aligned.std():.4f}")
        
        # Train the model
        model.fit(train_features_aligned, target_aligned)
        print("   ✓ Model training completed")
        
        # Test predictions on training data
        train_predictions = model.predict_proba(train_features_aligned)
        print(f"   Training predictions shape: {train_predictions.shape}")
        print(f"   Prediction range: [{train_predictions.min():.4f}, {train_predictions.max():.4f}]")
        
    except Exception as e:
        print(f"✗ Error in model training: {e}")
        return
    
    # Test predictions on test data
    try:
        print("\n4. Testing predictions on test data...")
        
        test_features = fe.prepare_features(test_data)
        test_features = test_features.dropna()
        
        print(f"   Test features shape: {test_features.shape}")
        
        if len(test_features) > 0:
            test_predictions = model.predict_proba(test_features)
            print(f"   Test predictions shape: {test_predictions.shape}")
            print(f"   Test prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
            
            # Show some sample predictions
            print("\n   Sample predictions (first 10 days):")
            for i in range(min(10, len(test_predictions))):
                date = test_features.index[i].strftime('%Y-%m-%d')
                pred = test_predictions[i]
                print(f"     {date}: {pred:.4f}")
                
        else:
            print("   ✗ No valid test features after preprocessing")
        
    except Exception as e:
        print(f"✗ Error in test predictions: {e}")
        return
    
    print("\n✓ ML model debug completed successfully!")


if __name__ == '__main__':
    debug_ml_model()
