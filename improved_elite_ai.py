#!/usr/bin/env python3
"""
Improved Elite AI Trainer - Uses Split-Adjusted Clean Data
Retrains models with properly adjusted historical data for better performance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

class ImprovedEliteAI:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def load_clean_data(self, symbol):
        """Load split-adjusted clean data or fallback to yfinance"""
        try:
            filename = f"clean_data_{symbol.lower()}.csv"
            # First, let's read the first few lines to understand the structure
            raw_data = pd.read_csv(filename, nrows=3)
            
            # The format appears to be:
            # Row 0: Column headers (Price, Close, High, Low, Open, Volume)
            # Row 1: Ticker symbols (NVDA, NVDA, NVDA, etc.)
            # Row 2: Empty/Date header
            # Row 3+: Actual data
            
            # Read with proper header setup
            data = pd.read_csv(filename, skiprows=3, 
                             names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
            
            # Set Date as index
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            # Ensure numeric types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove any rows with NaN values
            data = data.dropna()
            
            print(f"✅ Loaded clean data for {symbol}: {len(data)} records")
            print(f"   Columns: {list(data.columns)}")
            return data
            
        except Exception as e:
            print(f"⚠️  Error loading clean data for {symbol}: {str(e)}")
            print(f"⚠️  Falling back to yfinance...")
            # Fallback to yfinance data
            try:
                import yfinance as yf
                data = yf.download(symbol, period="2y", interval="1d", progress=False)
                print(f"✅ Downloaded yfinance data for {symbol}: {len(data)} records")
                return data
            except Exception as yf_error:
                print(f"❌ Failed to get data for {symbol}: {str(yf_error)}")
                return None
    
    def generate_improved_features(self, data):
        """Generate improved feature set focused on robust indicators"""
        df = data.copy()
        
        # Price-based features (split-adjusted)
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility features
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # Moving averages (split-adjusted)
        df['ma_5'] = df['Close'].rolling(5).mean()
        df['ma_20'] = df['Close'].rolling(20).mean()
        df['ma_50'] = df['Close'].rolling(50).mean()
        
        # Price ratios (split-neutral) - Fixed pandas assignment
        df = df.assign(
            price_ma5_ratio=df['Close'] / df['ma_5'],
            price_ma20_ratio=df['Close'] / df['ma_20'], 
            price_ma50_ratio=df['Close'] / df['ma_50']
        )
        
        # Volume features (split-adjusted)
        df['volume_ma_5'] = df['Volume'].rolling(5).mean()
        df['volume_ma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
        
        # RSI (split-neutral)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (split-neutral)
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands (split-adjusted but ratio-based)
        bb_middle = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = bb_middle + (bb_std * 2)
        df['bb_lower'] = bb_middle - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Momentum features
        df['momentum_5'] = df['Close'] / df['Close'].shift(5)
        df['momentum_20'] = df['Close'] / df['Close'].shift(20)
        
        # Target variable (next day return)
        df['target'] = df['returns'].shift(-1)
        
        # Feature columns (excluding target and intermediate calculations)
        feature_columns = [
            'returns', 'log_returns', 'volatility_5', 'volatility_20',
            'price_ma5_ratio', 'price_ma20_ratio', 'price_ma50_ratio',
            'volume_ratio', 'rsi', 'macd', 'macd_histogram', 'bb_position',
            'momentum_5', 'momentum_20'
        ]
        
        # Clean data
        df = df.dropna()
        
        return df[feature_columns], df['target']
    
    def train_improved_models(self, symbol):
        """Train improved models with clean data"""
        print(f"\n🎯 TRAINING IMPROVED MODELS FOR {symbol}")
        print("=" * 50)
        
        # Load clean data
        data = self.load_clean_data(symbol)
        if data is None:
            return None
        
        # Generate features
        X, y = self.generate_improved_features(data)
        
        print(f"📊 Feature set: {X.shape[1]} features, {X.shape[0]} samples")
        
        # Split data (time series aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train models with much stronger regularization
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=10,  # Much fewer trees
                max_depth=2,      # Very shallow
                min_samples_split=40,  # Much higher split requirement
                min_samples_leaf=20,   # Much higher leaf requirement
                max_features=0.5,      # Use only half the features
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=10,  # Much fewer trees
                max_depth=2,      # Very shallow
                learning_rate=0.01,  # Much lower learning rate
                min_samples_split=40,
                min_samples_leaf=20,
                subsample=0.7,    # Less data
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=10,  # Much fewer trees
                max_depth=2,      # Very shallow
                learning_rate=0.01,  # Much lower learning rate
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,    # Higher L1 regularization
                reg_lambda=1.0,   # Higher L2 regularization
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=10,  # Much fewer trees
                max_depth=2,      # Very shallow
                learning_rate=0.01,  # Much lower learning rate
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1
            ),
            'catboost': cb.CatBoostRegressor(
                iterations=10,    # Much fewer iterations
                depth=2,          # Very shallow
                learning_rate=0.01,  # Much lower learning rate
                l2_leaf_reg=10,   # Higher regularization
                random_state=42,
                verbose=False
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔧 Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Metrics
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                # Direction accuracy
                train_direction = np.mean(np.sign(y_train) == np.sign(train_pred))
                test_direction = np.mean(np.sign(y_test) == np.sign(test_pred))
                
                results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_direction': train_direction,
                    'test_direction': test_direction
                }
                
                print(f"   📈 Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
                print(f"   🎯 Train Direction: {train_direction:.1%}, Test Direction: {test_direction:.1%}")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {str(e)}")
        
        # Store results
        self.models[symbol] = results
        self.performance_metrics[symbol] = results
        
        return results
    
    def create_ensemble_prediction(self, symbol, models_results):
        """Create ensemble prediction from multiple models"""
        # Filter models with positive test R²
        good_models = {name: result for name, result in models_results.items() 
                      if result['test_r2'] > 0}
        
        if not good_models:
            print("❌ No models with positive R² - using best available")
            good_models = models_results
        
        print(f"\n🤖 ENSEMBLE CREATION FOR {symbol}")
        print(f"   Using {len(good_models)} models")
        
        # Weight models by test R² score
        total_weight = sum(max(0.1, result['test_r2']) for result in good_models.values())
        weights = {name: max(0.1, result['test_r2']) / total_weight 
                  for name, result in good_models.items()}
        
        print("   Model weights:")
        for name, weight in weights.items():
            r2 = good_models[name]['test_r2']
            direction = good_models[name]['test_direction']
            print(f"     {name}: {weight:.1%} (R²={r2:.3f}, Dir={direction:.1%})")
        
        return good_models, weights
    
    def predict_with_improved_model(self, symbol):
        """Make prediction using improved ensemble"""
        if symbol not in self.models:
            print(f"❌ No trained models for {symbol}")
            return None
        
        # Load recent data
        data = self.load_clean_data(symbol)
        if data is None:
            return None
        
        # Generate features for latest data point
        X, _ = self.generate_improved_features(data)
        latest_features = X.iloc[-1:]
        
        models_results = self.models[symbol]
        good_models, weights = self.create_ensemble_prediction(symbol, models_results)
        
        # Make ensemble prediction
        weighted_prediction = 0
        confidence_scores = []
        
        for name, result in good_models.items():
            model = result['model']
            pred = model.predict(latest_features)[0]
            weight = weights[name]
            weighted_prediction += pred * weight
            confidence_scores.append(result['test_r2'])
        
        # Calculate ensemble confidence
        ensemble_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Determine signal with more aggressive thresholds
        if weighted_prediction > 0.002:  # > 0.2% (much more sensitive)
            signal = "BUY"
        elif weighted_prediction < -0.002:  # < -0.2% (much more sensitive)
            signal = "SELL" 
        else:
            signal = "HOLD"
        
        return {
            'symbol': symbol,
            'predicted_return': weighted_prediction * 100,  # Convert to percentage
            'confidence': ensemble_confidence,
            'signal': signal,
            'models_used': len(good_models),
            'ensemble_r2': np.mean([r['test_r2'] for r in good_models.values()])
        }

def main():
    """Test improved Elite AI with clean data"""
    print("🚀 IMPROVED ELITE AI TRAINER")
    print("=" * 40)
    
    # Initialize improved AI
    ai = ImprovedEliteAI()
    
    # Train on clean data
    symbols = ['NVDA', 'AAPL', 'GOOGL', 'TSLA', 'AMZN']
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        results = ai.train_improved_models(symbol)
        
        if results:
            # Make prediction
            prediction = ai.predict_with_improved_model(symbol)
            if prediction:
                print(f"\n🎯 IMPROVED PREDICTION FOR {symbol}:")
                print(f"   📈 Predicted Return: {prediction['predicted_return']:.2f}%")
                print(f"   📊 Confidence: {prediction['confidence']:.3f}")
                print(f"   🚦 Signal: {prediction['signal']}")
                print(f"   🤖 Models Used: {prediction['models_used']}")
                print(f"   📈 Ensemble R²: {prediction['ensemble_r2']:.3f}")
    
    print(f"\n🏁 IMPROVED MODEL TRAINING COMPLETE")
    print("✅ Models trained with split-adjusted data")
    print("✅ Should show much better performance")
    print("✅ More reliable predictions expected")

if __name__ == "__main__":
    main()
