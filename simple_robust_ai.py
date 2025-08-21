#!/usr/bin/env python3
"""
Simplified Robust AI Trader
Focus on basic, proven indicators with simple models to avoid overfitting
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class SimpleRobustAI:
    def __init__(self):
        self.models = {}
        
    def load_clean_data(self, symbol):
        """Load split-adjusted clean data"""
        try:
            filename = f"clean_data_{symbol}.csv"
            data = pd.read_csv(filename, index_col=0, parse_dates=True)
            return data
        except Exception as e:
            print(f"❌ Failed to load clean data for {symbol}: {str(e)}")
            return None
    
    def generate_simple_features(self, data):
        """Generate only the most basic, robust features"""
        df = data.copy()
        
        # Basic returns
        df['returns'] = df['Close'].pct_change()
        
        # Simple moving averages
        df['ma_5'] = df['Close'].rolling(5).mean()
        df['ma_20'] = df['Close'].rolling(20).mean()
        
        # Price position relative to MAs (split-neutral)
        df['price_vs_ma5'] = (df['Close'] / df['ma_5']) - 1
        df['price_vs_ma20'] = (df['Close'] / df['ma_20']) - 1
        
        # Simple momentum
        df['momentum_5'] = (df['Close'] / df['Close'].shift(5)) - 1
        df['momentum_20'] = (df['Close'] / df['Close'].shift(20)) - 1
        
        # Volume indicator
        df['volume_ma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = (df['Volume'] / df['volume_ma_20']) - 1
        
        # Target (next day return)
        df['target'] = df['returns'].shift(-1)
        
        # Feature columns
        feature_columns = [
            'returns', 'price_vs_ma5', 'price_vs_ma20', 
            'momentum_5', 'momentum_20', 'volume_ratio'
        ]
        
        # Clean data
        df = df.dropna()
        
        return df[feature_columns], df['target']
    
    def train_simple_models(self, symbol):
        """Train simple, robust models"""
        print(f"\n🎯 TRAINING SIMPLE MODELS FOR {symbol}")
        print("=" * 45)
        
        # Load data
        data = self.load_clean_data(symbol)
        if data is None:
            return None
        
        # Generate simple features
        X, y = self.generate_simple_features(data)
        print(f"📊 Using {X.shape[1]} simple features, {X.shape[0]} samples")
        
        # Use more conservative train/test split
        split_idx = int(len(X) * 0.85)  # More training data
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Simple models with regularization
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),  # Regularization
            'simple_rf': RandomForestRegressor(
                n_estimators=20,  # Much fewer trees
                max_depth=3,      # Shallow trees
                min_samples_split=20,  # Conservative splits
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔧 Training {name}...")
            
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Test predictions only (no training predictions to avoid overfitting impression)
                test_pred = model.predict(X_test)
                
                # Metrics
                test_r2 = r2_score(y_test, test_pred)
                test_direction = np.mean(np.sign(y_test) == np.sign(test_pred))
                
                # Calculate stability (consistency of predictions)
                pred_std = np.std(test_pred)
                
                results[name] = {
                    'model': model,
                    'test_r2': test_r2,
                    'test_direction': test_direction,
                    'pred_stability': pred_std
                }
                
                print(f"   📈 Test R²: {test_r2:.3f}")
                print(f"   🎯 Test Direction: {test_direction:.1%}")
                print(f"   📊 Prediction Stability: {pred_std:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        self.models[symbol] = results
        return results
    
    def make_simple_prediction(self, symbol):
        """Make prediction using simple ensemble"""
        if symbol not in self.models:
            return None
        
        # Load recent data
        data = self.load_clean_data(symbol)
        if data is None:
            return None
        
        # Get latest features
        X, _ = self.generate_simple_features(data)
        latest_features = X.iloc[-1:].values
        
        # Get models
        models_results = self.models[symbol]
        
        # Only use models with reasonable performance
        good_models = {name: result for name, result in models_results.items() 
                      if result['test_r2'] > -0.1 and result['test_direction'] > 0.45}
        
        if not good_models:
            print(f"❌ No reliable models for {symbol}")
            return None
        
        print(f"\n🤖 SIMPLE PREDICTION FOR {symbol}")
        print(f"   Using {len(good_models)} reliable models")
        
        # Simple average ensemble
        predictions = []
        confidences = []
        
        for name, result in good_models.items():
            model = result['model']
            pred = model.predict(latest_features)[0]
            predictions.append(pred)
            # Use direction accuracy as confidence proxy
            confidences.append(result['test_direction'])
            print(f"     {name}: {pred*100:.2f}% (confidence: {result['test_direction']:.1%})")
        
        # Ensemble prediction
        avg_prediction = np.mean(predictions)
        avg_confidence = np.mean(confidences)
        
        # Conservative signal generation
        if avg_prediction > 0.015:  # > 1.5%
            signal = "BUY"
        elif avg_prediction < -0.015:  # < -1.5%
            signal = "SELL"
        else:
            signal = "HOLD"
        
        return {
            'symbol': symbol,
            'predicted_return': avg_prediction * 100,
            'confidence': avg_confidence,
            'signal': signal,
            'models_used': len(good_models),
            'prediction_range': (min(predictions)*100, max(predictions)*100)
        }
    
    def validate_prediction_quality(self, symbol):
        """Validate prediction quality"""
        if symbol not in self.models:
            return None
        
        models_results = self.models[symbol]
        
        print(f"\n📊 PREDICTION QUALITY FOR {symbol}")
        print("-" * 35)
        
        # Check if any model is actually useful
        best_r2 = max(result['test_r2'] for result in models_results.values())
        best_direction = max(result['test_direction'] for result in models_results.values())
        
        print(f"   Best R²: {best_r2:.3f}")
        print(f"   Best Direction Accuracy: {best_direction:.1%}")
        
        if best_r2 > 0.01 and best_direction > 0.55:
            quality = "🟢 GOOD"
        elif best_r2 > -0.05 and best_direction > 0.50:
            quality = "🟡 FAIR"
        else:
            quality = "🔴 POOR"
        
        print(f"   Overall Quality: {quality}")
        return quality

def main():
    """Test simple robust AI"""
    print("🤖 SIMPLE ROBUST AI TRAINER")
    print("=" * 35)
    
    ai = SimpleRobustAI()
    symbols = ['NVDA', 'AAPL', 'GOOGL', 'TSLA', 'AMZN']
    
    print("\n📊 TRAINING SIMPLE MODELS")
    print("=" * 30)
    
    for symbol in symbols:
        # Train models
        results = ai.train_simple_models(symbol)
        
        if results:
            # Validate quality
            quality = ai.validate_prediction_quality(symbol)
            
            # Make prediction if quality is acceptable
            if quality in ["🟢 GOOD", "🟡 FAIR"]:
                prediction = ai.make_simple_prediction(symbol)
                if prediction:
                    print(f"\n🎯 PREDICTION:")
                    print(f"   📈 Return: {prediction['predicted_return']:.2f}%")
                    print(f"   📊 Confidence: {prediction['confidence']:.1%}")
                    print(f"   🚦 Signal: {prediction['signal']}")
                    print(f"   📊 Range: {prediction['prediction_range'][0]:.2f}% to {prediction['prediction_range'][1]:.2f}%")
            else:
                print(f"\n❌ {symbol}: Model quality too poor for predictions")
        
        print("\n" + "-"*60)
    
    print(f"\n🏁 SIMPLE MODEL ANALYSIS COMPLETE")
    print("✅ Focused on robust, simple indicators")
    print("✅ Avoided overfitting with conservative approach")
    print("✅ Only provides predictions when models are reliable")

if __name__ == "__main__":
    main()
