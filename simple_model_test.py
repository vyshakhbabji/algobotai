#!/usr/bin/env python3
"""
Simple Backtester - Just test if current models work on recent data
"""

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import json
from sklearn.metrics import mean_absolute_error

def simple_model_test():
    """Simple test of current models on recent NVDA data"""
    print("🚀 SIMPLE MODEL TEST")
    print("=" * 50)
    
    try:
        # Load models
        print("🤖 Loading models...")
        models = {
            'rf': joblib.load('fixed_data/models/random_forest_model.pkl'),
            'gb': joblib.load('fixed_data/models/gradient_boosting_model.pkl'),
            'linear': joblib.load('fixed_data/models/linear_regression_model.pkl'),
            'ridge': joblib.load('fixed_data/models/ridge_model.pkl')
        }
        print(f"✅ Loaded {len(models)} models")
        
        # Get NVDA data
        print("📡 Fetching NVDA data...")
        ticker = yf.Ticker('NVDA')
        data = ticker.history(period='3mo')  # Last 3 months
        
        print(f"📈 Retrieved {len(data)} days")
        print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Check what the models actually expect
        print(f"\n🔍 MODEL INPUT REQUIREMENTS:")
        for name, model in models.items():
            if hasattr(model, 'n_features_in_'):
                print(f"   {name.upper()}: Expects {model.n_features_in_} features")
        
        # Load feature metadata
        with open('fixed_data/preprocessed/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        expected_features = metadata['feature_columns']
        sequence_length = metadata.get('sequence_length', 10)
        input_shape = metadata.get('input_shape', [10, 38])
        
        print(f"📋 Training used: {len(expected_features)} features")
        print(f"📋 Sequence length: {sequence_length}")
        print(f"📋 Input shape: {input_shape}")
        print(f"📋 Expected flattened size: {input_shape[0] * input_shape[1]}")
        
        # Calculate basic indicators to match what we can
        df = data.copy()
        
        # Basic indicators
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Other basic features
        df['Returns'] = df['Close'].pct_change()
        df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['PC_PCT'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Bollinger Bands
        bb_middle = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_upper'] = bb_middle + (bb_std * 2)
        df['BB_middle'] = bb_middle
        df['BB_lower'] = bb_middle - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / df['BB_width']
        
        # Stochastic
        low_min = df['Low'].rolling(14).min()
        high_max = df['High'].rolling(14).max()
        df['Stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['Stoch_d'] = df['Stoch_k'].rolling(3).mean()
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # VWAP
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Additional features
        df['Price_above_SMA20'] = (df['Close'] > df['SMA_20']).astype(int)
        df['Price_above_SMA50'] = (df['Close'] > df['SMA_50']).astype(int)
        df['RSI_oversold'] = (df['RSI_14'] < 30).astype(int)
        df['RSI_overbought'] = (df['RSI_14'] > 70).astype(int)
        df['Price_momentum_5'] = df['Close'].pct_change(5) * 100
        df['Price_momentum_10'] = df['Close'].pct_change(10) * 100
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Distance_to_high'] = (df['High_20'] - df['Close']) / df['Close']
        df['Distance_to_low'] = (df['Close'] - df['Low_20']) / df['Close']
        
        # Try to match expected features
        available_features = []
        for feature in expected_features:
            if feature in df.columns:
                available_features.append(feature)
            else:
                print(f"⚠️ Missing feature: {feature}")
        
        print(f"\n📊 Available features: {len(available_features)} of {len(expected_features)}")
        
        # Create feature matrix
        if len(available_features) >= 30:  # Need at least most features
            feature_data = df[available_features].dropna()
            
            if len(feature_data) >= sequence_length + 5:  # Need enough data
                print(f"✅ Have {len(feature_data)} samples with {len(available_features)} features")
                
                # Create sequences 
                sequences = []
                targets = []
                dates_seq = []
                
                for i in range(sequence_length, len(feature_data)):
                    sequences.append(feature_data.iloc[i-sequence_length:i].values)
                    targets.append(feature_data['Close'].iloc[i] if 'Close' in feature_data.columns else data['Close'].iloc[feature_data.index[i]])
                    dates_seq.append(feature_data.index[i])
                
                sequences = np.array(sequences)
                targets = np.array(targets)
                
                if len(sequences) > 0:
                    # Use the last few for testing
                    test_size = min(5, len(sequences) // 4)
                    
                    X_test = sequences[-test_size:]
                    y_test = targets[-test_size:]
                    test_dates = dates_seq[-test_size:]
                    
                    # Flatten for sklearn models
                    X_test_flat = X_test.reshape(test_size, -1)
                    
                    print(f"\n🎯 TESTING ON LAST {test_size} DAYS:")
                    print(f"📅 Test period: {test_dates[0]} to {test_dates[-1]}")
                    print(f"📐 Input shape: {X_test_flat.shape}")
                    
                    # Test each model
                    for name, model in models.items():
                        try:
                            predictions = model.predict(X_test_flat)
                            mae = mean_absolute_error(y_test, predictions)
                            
                            # Calculate direction accuracy
                            if len(y_test) > 1:
                                actual_direction = np.sign(np.diff(y_test))
                                pred_direction = np.sign(np.diff(predictions))
                                direction_accuracy = np.mean(actual_direction == pred_direction) * 100
                            else:
                                direction_accuracy = 50
                            
                            # Total change
                            actual_change = ((y_test[-1] - y_test[0]) / y_test[0]) * 100
                            pred_change = ((predictions[-1] - predictions[0]) / predictions[0]) * 100
                            
                            print(f"\n🧠 {name.upper()} Results:")
                            print(f"   📊 MAE: ${mae:.2f}")
                            print(f"   📊 Direction Accuracy: {direction_accuracy:.1f}%")
                            print(f"   📊 Actual Change: {actual_change:+.1f}%")
                            print(f"   📊 Predicted Change: {pred_change:+.1f}%")
                            print(f"   📊 Direction Match: {'✅' if np.sign(actual_change) == np.sign(pred_change) else '❌'}")
                            
                        except Exception as e:
                            print(f"\n❌ {name.upper()}: {e}")
                
                else:
                    print("❌ Could not create sequences")
            else:
                print(f"❌ Insufficient data: {len(feature_data)} samples")
        else:
            print(f"❌ Too many missing features: {len(available_features)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    simple_model_test()
