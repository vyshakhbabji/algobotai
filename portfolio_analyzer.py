#!/usr/bin/env python3
"""
Portfolio AI Performance Analyzer
Tests AI models against your actual stock holdings
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Your actual portfolio holdings
PORTFOLIO = {
    'GOOG': {'current_value': 15243.84, 'gain': 1898.04, 'gain_pct': 28.35, 'weight': 25.4},
    'AAPL': {'current_value': 9201.71, 'gain': 104.13, 'gain_pct': 2.74, 'weight': 15.3},
    'MSFT': {'current_value': 8436.73, 'gain': 1874.12, 'gain_pct': 47.00, 'weight': 14.0},
    'NVDA': {'current_value': 10368.29, 'gain': 5737.82, 'gain_pct': 173.91, 'weight': 17.3},
    'META': {'current_value': 8126.67, 'gain': 3188.28, 'gain_pct': 94.54, 'weight': 13.5},
    'AMZN': {'current_value': 8537.61, 'gain': 1478.43, 'gain_pct': 29.50, 'weight': 14.2},
    'AVGO': {'current_value': 12003.61, 'gain': 5643.64, 'gain_pct': 201.30, 'weight': 24.9},
    'PLTR': {'current_value': 18436.28, 'gain': 16434.39, 'gain_pct': 901.54, 'weight': 38.3},
    'NFLX': {'current_value': 5747.67, 'gain': 2135.23, 'gain_pct': 111.11, 'weight': 11.9},
    'TSM': {'current_value': 4612.94, 'gain': 1003.02, 'gain_pct': 44.46, 'weight': 9.6},
    'PANW': {'current_value': 3827.61, 'gain': 99.94, 'gain_pct': 6.46, 'weight': 7.9},
    'NOW': {'current_value': 3416.38, 'gain': 90.84, 'gain_pct': 6.18, 'weight': 7.1},
    # ETFs
    'XLK': {'current_value': 9822.32, 'gain': 2101.26, 'gain_pct': 43.26, 'weight': 50.6},
    'QQQ': {'current_value': 9562.64, 'gain': 1946.59, 'gain_pct': 42.06, 'weight': 49.3},
    'BRK.B': {'current_value': 6003.13, 'gain': 41.62, 'gain_pct': 1.57, 'weight': 49.9},
    'COST': {'current_value': 6019.21, 'gain': 497.24, 'gain_pct': 14.35, 'weight': 50.0}
}

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    try:
        # Price-based indicators
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # Volatility
        df['volatility'] = df['Close'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=50).mean()
        
        # Volume analysis
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Price momentum
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Smart feature engineering
        df['price_vs_sma20'] = df['Close'] / df['sma_20'] - 1
        df['volume_price_trend'] = df['volume_ratio'] * df['momentum_5']
        df['volatility_momentum'] = df['volatility_ratio'] * abs(df['momentum_5'])
        df['rsi_momentum'] = df['rsi'] * df['momentum_5']
        
        return df
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return df

def load_models():
    """Load trained AI models"""
    models = {}
    
    # Try loading from 'better' directory first (higher quality models)
    better_dir = 'better'
    model_files = {
        'best_rf': 'best_rf_model.pkl',
        'meta_model': 'meta_model.pkl'
    }
    
    for name, filename in model_files.items():
        try:
            path = os.path.join(better_dir, filename)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
                print(f"✅ Loaded {name} model")
            else:
                print(f"❌ Model file not found: {path}")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
    
    # Fallback to fixed_data models if better models not available
    if not models:
        print("Trying fixed_data models...")
        model_dir = 'fixed_data/models'
        fallback_files = {
            'random_forest': 'random_forest_model.pkl',
            'gradient_boosting': 'gradient_boosting_model.pkl'
        }
        
        for name, filename in fallback_files.items():
            try:
                path = os.path.join(model_dir, filename)
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        models[name] = pickle.load(f)
                    print(f"✅ Loaded {name} model")
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
    
    return models

def load_preprocessors():
    """Load feature scaler and other preprocessors"""
    try:
        # Try 'better' directory first
        better_dir = 'better'
        scaler_path = os.path.join(better_dir, 'feature_scaler.pkl')
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("✅ Loaded feature scaler from better/")
            return scaler
        
        # Fallback to fixed_data
        preprocessed_dir = 'fixed_data/preprocessed'
        scaler_path = os.path.join(preprocessed_dir, 'feature_scaler.pkl')
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("✅ Loaded feature scaler from fixed_data/")
            return scaler
        else:
            print("❌ Feature scaler not found")
            return None
    except Exception as e:
        print(f"❌ Error loading preprocessors: {e}")
        return None

def create_sequences(data, sequence_length=10):
    """Create sequences for model prediction"""
    if len(data) < sequence_length:
        return None
    
    # Get the most recent sequence
    sequence = data[-sequence_length:].values
    
    # Flatten the sequence (10 days × 38 features = 380 features)
    flattened = sequence.flatten()
    
    return flattened.reshape(1, -1)

def get_stock_data(symbol, period='1y'):
    """Fetch and process stock data"""
    try:
        print(f"📈 Fetching data for {symbol}...")
        
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        if df.empty:
            print(f"❌ No data available for {symbol}")
            return None
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) < 50:  # Need enough data for analysis
            print(f"❌ Insufficient data for {symbol} ({len(df)} rows)")
            return None
        
        print(f"✅ Processed {len(df)} days of data for {symbol}")
        return df
    
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None

def make_predictions(models, scaler, data, symbol):
    """Make predictions using all available models"""
    try:
        # Expected feature columns (same as training)
        feature_columns = [
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'volatility', 'volatility_ratio',
            'volume_sma', 'volume_ratio', 'momentum_5', 'momentum_20', 'rsi', 'macd',
            'macd_signal', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_position',
            'price_vs_sma20', 'volume_price_trend', 'volatility_momentum', 'rsi_momentum',
            'Open', 'High', 'Low', 'Close', 'Volume'
        ]
        
        # Add missing columns if needed
        for col in feature_columns:
            if col not in data.columns:
                if 'sma' in col or 'ema' in col or 'bb' in col:
                    data[col] = data['Close']
                elif 'volume' in col.lower():
                    data[col] = data['Volume'] if 'Volume' in data.columns else 1000000
                else:
                    data[col] = 0
        
        # Select and order features
        feature_data = data[feature_columns].copy()
        
        # Create sequence for prediction
        sequence = create_sequences(feature_data, sequence_length=10)
        
        if sequence is None:
            print(f"❌ Cannot create sequence for {symbol} - insufficient data")
            return {}
        
        # Scale features if scaler is available
        if scaler is not None:
            try:
                sequence_scaled = scaler.transform(sequence)
            except Exception as e:
                print(f"⚠️ Scaling failed for {symbol}, using raw features: {e}")
                sequence_scaled = sequence
        else:
            sequence_scaled = sequence
        
        predictions = {}
        
        # Make predictions with each model
        for model_name, model in models.items():
            try:
                prediction = model.predict(sequence_scaled)[0]
                predictions[model_name] = prediction
                print(f"  {model_name}: {prediction:.4f}")
            except Exception as e:
                print(f"  ❌ {model_name} prediction failed: {e}")
                predictions[model_name] = 0
        
        return predictions
    
    except Exception as e:
        print(f"❌ Prediction error for {symbol}: {e}")
        return {}

def analyze_portfolio():
    """Analyze entire portfolio with AI predictions"""
    print("🤖 AI Portfolio Performance Analysis")
    print("=" * 60)
    
    # Load models and preprocessors
    models = load_models()
    scaler = load_preprocessors()
    
    if not models:
        print("❌ No models loaded. Please train models first.")
        return
    
    print(f"\n📊 Analyzing {len(PORTFOLIO)} holdings...")
    print("-" * 60)
    
    results = []
    total_value = 0
    total_gain = 0
    ai_score_sum = 0
    valid_predictions = 0
    
    for symbol, holdings in PORTFOLIO.items():
        print(f"\n🔍 Analyzing {symbol}...")
        
        # Get stock data
        data = get_stock_data(symbol)
        
        if data is None:
            print(f"❌ Skipping {symbol} - no data available")
            continue
        
        # Make AI predictions
        predictions = make_predictions(models, scaler, data, symbol)
        
        if not predictions:
            print(f"❌ No predictions for {symbol}")
            continue
        
        # Calculate AI ensemble score
        ai_scores = list(predictions.values())
        ai_score = np.mean(ai_scores) if ai_scores else 0
        ai_confidence = 1 - (np.std(ai_scores) if len(ai_scores) > 1 else 0)
        
        # Current stock info
        current_price = data['Close'].iloc[-1]
        price_change_1d = (current_price / data['Close'].iloc[-2] - 1) * 100 if len(data) > 1 else 0
        
        # Technical analysis
        latest = data.iloc[-1]
        rsi = latest['rsi']
        bb_position = latest['bb_position']
        momentum_5 = latest['momentum_5'] * 100
        
        # AI recommendation
        if ai_score > 0.1:
            ai_signal = "🟢 STRONG BUY"
        elif ai_score > 0.05:
            ai_signal = "🔵 BUY"
        elif ai_score > -0.05:
            ai_signal = "🟡 HOLD"
        elif ai_score > -0.1:
            ai_signal = "🔴 SELL"
        else:
            ai_signal = "🔴 STRONG SELL"
        
        result = {
            'symbol': symbol,
            'current_value': holdings['current_value'],
            'gain': holdings['gain'],
            'gain_pct': holdings['gain_pct'],
            'ai_score': ai_score,
            'ai_signal': ai_signal,
            'ai_confidence': ai_confidence,
            'current_price': current_price,
            'price_change_1d': price_change_1d,
            'rsi': rsi,
            'momentum_5': momentum_5,
            'bb_position': bb_position
        }
        
        results.append(result)
        
        # Display result
        print(f"  💰 Value: ${holdings['current_value']:,.2f} (+{holdings['gain_pct']:.1f}%)")
        print(f"  🤖 AI Score: {ai_score:.4f} | Signal: {ai_signal}")
        print(f"  📊 Confidence: {ai_confidence:.2f} | RSI: {rsi:.1f} | Momentum: {momentum_5:.1f}%")
        
        total_value += holdings['current_value']
        total_gain += holdings['gain']
        ai_score_sum += ai_score
        valid_predictions += 1
    
    # Portfolio summary
    print("\n" + "=" * 60)
    print("📊 PORTFOLIO SUMMARY")
    print("=" * 60)
    
    total_gain_pct = (total_gain / (total_value - total_gain)) * 100 if total_value > total_gain else 0
    avg_ai_score = ai_score_sum / valid_predictions if valid_predictions > 0 else 0
    
    print(f"💰 Total Portfolio Value: ${total_value:,.2f}")
    print(f"📈 Total Gains: ${total_gain:,.2f} (+{total_gain_pct:.1f}%)")
    print(f"🤖 Average AI Score: {avg_ai_score:.4f}")
    print(f"📊 Stocks Analyzed: {valid_predictions}/{len(PORTFOLIO)}")
    
    # Top performers by AI score
    if results:
        print("\n🏆 TOP AI RECOMMENDATIONS:")
        print("-" * 40)
        results_sorted = sorted(results, key=lambda x: x['ai_score'], reverse=True)
        
        for i, result in enumerate(results_sorted[:5]):
            print(f"{i+1}. {result['symbol']}: {result['ai_signal']} (Score: {result['ai_score']:.4f})")
    
    # Performance correlation
    if results:
        print("\n🔍 AI vs ACTUAL PERFORMANCE:")
        print("-" * 40)
        
        for result in sorted(results, key=lambda x: x['gain_pct'], reverse=True):
            ai_match = "✅" if (result['ai_score'] > 0 and result['gain_pct'] > 0) or (result['ai_score'] < 0 and result['gain_pct'] < 0) else "❌"
            print(f"{result['symbol']}: {ai_match} Actual: +{result['gain_pct']:.1f}% | AI Score: {result['ai_score']:.4f}")
    
    return results

if __name__ == "__main__":
    try:
        results = analyze_portfolio()
        print(f"\n✅ Portfolio analysis complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
