#!/usr/bin/env python3
"""
Improved NVDA Prediction Model
Focus on reliability over complexity
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class ImprovedNVDAModel:
    def __init__(self):
        self.symbol = "NVDA"
        
    def build_simple_reliable_model(self):
        """Build a simpler, more reliable model"""
        print("🛠️ BUILDING IMPROVED NVDA MODEL")
        print("=" * 35)
        
        # Get data
        data = yf.download("NVDA", period="1y", interval="1d", progress=False)
        
        # Simple but effective features
        features = self.calculate_reliable_features(data)
        
        # Target: next day return
        target = data['Close'].pct_change().shift(-1) * 100
        
        # Align data
        combined = pd.concat([features, target.rename('target')], axis=1).dropna()
        
        if len(combined) < 100:
            print("❌ Insufficient data")
            return None
            
        # Split data
        split_idx = int(len(combined) * 0.8)
        train_data = combined.iloc[:split_idx]
        test_data = combined.iloc[split_idx:]
        
        # Train simple model
        X_train = train_data.drop('target', axis=1)
        y_train = train_data['target']
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        # Use simpler Random Forest
        model = RandomForestRegressor(
            n_estimators=50,  # Less trees
            max_depth=5,      # Limit depth
            min_samples_split=10,  # Prevent overfitting
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Direction accuracy
        train_direction = np.mean(np.sign(train_pred) == np.sign(y_train))
        test_direction = np.mean(np.sign(test_pred) == np.sign(y_test))
        
        print(f"📊 IMPROVED MODEL PERFORMANCE:")
        print(f"   Train R²: {train_r2:.3f} {'🟢' if train_r2 > 0 else '🔴'}")
        print(f"   Test R²: {test_r2:.3f} {'🟢' if test_r2 > 0 else '🔴'}")
        print(f"   Train Direction: {train_direction:.1%} {'🟢' if train_direction > 0.5 else '🔴'}")
        print(f"   Test Direction: {test_direction:.1%} {'🟢' if test_direction > 0.5 else '🔴'}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 TOP FEATURES:")
        for _, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
            
        # Current prediction
        latest_features = features.iloc[-1:].drop('target', axis=1, errors='ignore')
        current_pred = model.predict(latest_features)[0]
        
        # Calculate confidence based on model performance
        confidence = max(0.1, min(0.9, test_direction))
        
        print(f"\n🔮 CURRENT PREDICTION:")
        print(f"   Return: {current_pred:.2f}%")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Signal: {'BUY' if current_pred > 1 else 'SELL' if current_pred < -1 else 'HOLD'}")
        
        return {
            'prediction': current_pred,
            'confidence': confidence,
            'test_r2': test_r2,
            'test_direction': test_direction,
            'model_quality': 'Good' if test_r2 > 0 and test_direction > 0.55 else 'Poor'
        }
        
    def calculate_reliable_features(self, data):
        """Calculate only the most reliable features"""
        
        # Price features
        data['returns_1d'] = data['Close'].pct_change()
        data['returns_5d'] = data['Close'].pct_change(5)
        data['returns_20d'] = data['Close'].pct_change(20)
        
        # Moving averages
        data['sma_10'] = data['Close'].rolling(10).mean()
        data['sma_20'] = data['Close'].rolling(20).mean()
        data['price_vs_sma20'] = (data['Close'] / data['sma_20'] - 1)
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume
        data['volume_sma'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma']
        
        # Volatility
        data['volatility_10d'] = data['returns_1d'].rolling(10).std()
        
        # Select features
        features = data[[
            'returns_1d', 'returns_5d', 'price_vs_sma20', 
            'rsi', 'volume_ratio', 'volatility_10d'
        ]].copy()
        
        return features

def main():
    """Test improved model"""
    model = ImprovedNVDAModel()
    results = model.build_simple_reliable_model()
    
    if results:
        print(f"\n🎯 VERDICT:")
        if results['model_quality'] == 'Good':
            print(f"   ✅ This model is RELIABLE for NVDA")
            print(f"   💡 Use this prediction: {results['prediction']:.2f}%")
        else:
            print(f"   ❌ Even simplified model struggles with NVDA")
            print(f"   💡 NVDA may be too unpredictable for ML models")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   • Use fundamental analysis over ML models for NVDA")
    print(f"   • Focus on AI sector trends, not daily price predictions")
    print(f"   • Consider position sizing over timing")
    
if __name__ == "__main__":
    main()
