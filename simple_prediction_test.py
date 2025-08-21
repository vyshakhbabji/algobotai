#!/usr/bin/env python3
"""
Simple test to understand what ML predictions are being generated
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def test_predictions():
    """Test what predictions the ML model generates for NVDA"""
    
    print("Testing NVDA ML predictions...")
    
    # Import the feature engineering
    try:
        from algobot.features.advanced import build_advanced_features, ADV_FEATURE_COLUMNS
        print("✓ Imported feature engineering modules")
    except Exception as e:
        print(f"✗ Error importing: {e}")
        return
    
    # Get NVDA data
    print("\nDownloading NVDA data...")
    train_data = yf.download('NVDA', start='2022-05-13', end='2024-05-13', progress=False)
    test_data = yf.download('NVDA', start='2024-05-13', end='2024-08-13', progress=False)
    
    print(f"Training data: {len(train_data)} days")
    print(f"Test data: {len(test_data)} days")
    
    # Build features
    print("\nBuilding features...")
    try:
        train_features = build_advanced_features(train_data)
        test_features = build_advanced_features(test_data)
        
        print(f"Training features: {train_features.shape}")
        print(f"Test features: {test_features.shape}")
        
        # Check target distribution
        target_up = train_features['target_up']
        print(f"Target distribution - Up days: {target_up.mean():.2%}")
        
    except Exception as e:
        print(f"✗ Error building features: {e}")
        return
    
    # Train model (same as in advanced_forward_sim)
    print("\nTraining model...")
    try:
        X_train = train_features[ADV_FEATURE_COLUMNS]
        y_train = train_features['target_up']
        
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {list(X_train.columns)}")
        
        # Train models
        gb = GradientBoostingClassifier(random_state=42)
        lr = LogisticRegression(max_iter=500)
        
        gb.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        
        print("✓ Model training completed")
        
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return
    
    # Make predictions on test data
    print("\nMaking predictions...")
    try:
        X_test = test_features[ADV_FEATURE_COLUMNS]
        
        # Get individual model predictions
        p_gb = gb.predict_proba(X_test)[:, 1]
        p_lr = lr.predict_proba(X_test)[:, 1]
        
        # Blend predictions (same as in advanced_forward_sim)
        p_blend = 0.5 * p_gb + 0.5 * p_lr
        
        print(f"Prediction statistics:")
        print(f"  Min: {p_blend.min():.4f}")
        print(f"  Max: {p_blend.max():.4f}")
        print(f"  Mean: {p_blend.mean():.4f}")
        print(f"  Std: {p_blend.std():.4f}")
        
        # Check how many predictions exceed thresholds
        high_conf = (p_blend >= 0.60).sum()
        med_conf = (p_blend >= 0.50).sum()
        low_conf = (p_blend <= 0.40).sum()
        
        print(f"\nThreshold analysis:")
        print(f"  Predictions >= 0.60 (buy threshold): {high_conf}/{len(p_blend)} ({high_conf/len(p_blend):.1%})")
        print(f"  Predictions >= 0.50: {med_conf}/{len(p_blend)} ({med_conf/len(p_blend):.1%})")
        print(f"  Predictions <= 0.40: {low_conf}/{len(p_blend)} ({low_conf/len(p_blend):.1%})")
        
        # Show daily predictions
        print(f"\nDaily predictions (first 20 days):")
        for i in range(min(20, len(p_blend))):
            date = X_test.index[i].strftime('%Y-%m-%d')
            pred = p_blend[i]
            gb_pred = p_gb[i]
            lr_pred = p_lr[i]
            signal = "BUY" if pred >= 0.60 else "SELL" if pred <= 0.40 else "HOLD"
            print(f"  {date}: {pred:.4f} (GB: {gb_pred:.4f}, LR: {lr_pred:.4f}) -> {signal}")
        
        # Check actual returns during test period
        actual_returns = test_features['target_return_1'].dropna()
        print(f"\nActual returns during test period:")
        print(f"  Mean daily return: {actual_returns.mean():.4f} ({actual_returns.mean()*252:.2%} annualized)")
        print(f"  Up days: {(actual_returns > 0).mean():.2%}")
        
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        return
    
    print("\n✓ Prediction analysis completed")


if __name__ == '__main__':
    test_predictions()
