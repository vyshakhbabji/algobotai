#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def test_ensemble():
    print("🧪 Testing ensemble model creation...")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y_reg = np.random.randn(1000)  # Regression target
    y_clf = np.random.randint(0, 3, 1000)  # Classification target
    
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    
    # Test individual models first
    print("1. Testing individual models...")
    
    # XGBoost
    try:
        xgb_reg = xgb.XGBRegressor(n_estimators=10, random_state=42)
        xgb_reg.fit(X_train, y_reg_train)
        print("   ✅ XGBRegressor works")
    except Exception as e:
        print(f"   ❌ XGBRegressor failed: {e}")
        return
    
    # LightGBM
    try:
        lgb_reg = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
        lgb_reg.fit(X_train, y_reg_train)
        print("   ✅ LGBMRegressor works")
    except Exception as e:
        print(f"   ❌ LGBMRegressor failed: {e}")
        return
    
    # Neural Network
    try:
        nn_reg = MLPRegressor(hidden_layer_sizes=(10,), max_iter=10, random_state=42)
        nn_reg.fit(X_train, y_reg_train)
        print("   ✅ MLPRegressor works")
    except Exception as e:
        print(f"   ❌ MLPRegressor failed: {e}")
        return
    
    # Test VotingRegressor
    print("2. Testing VotingRegressor...")
    try:
        voting_reg = VotingRegressor([
            ('xgb', xgb.XGBRegressor(n_estimators=10, random_state=42)),
            ('lgb', lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)),
            ('nn', MLPRegressor(hidden_layer_sizes=(10,), max_iter=10, random_state=42))
        ])
        voting_reg.fit(X_train, y_reg_train)
        print("   ✅ VotingRegressor works")
    except Exception as e:
        print(f"   ❌ VotingRegressor failed: {e}")
        return
    
    # Test VotingClassifier
    print("3. Testing VotingClassifier...")
    try:
        voting_clf = VotingClassifier([
            ('xgb', xgb.XGBClassifier(n_estimators=10, random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)),
            ('nn', MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, random_state=42))
        ], voting='soft')
        voting_clf.fit(X_train, y_clf_train)
        print("   ✅ VotingClassifier works")
    except Exception as e:
        print(f"   ❌ VotingClassifier failed: {e}")
        return
    
    print("🏁 All ensemble tests passed!")

if __name__ == "__main__":
    test_ensemble()
