#!/usr/bin/env python3
"""
Quick AI Portfolio Test
Tests a simple AI model against your actual portfolio performance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    'XLK': {'current_value': 9822.32, 'gain': 2101.26, 'gain_pct': 43.26, 'weight': 50.6},
    'QQQ': {'current_value': 9562.64, 'gain': 1946.59, 'gain_pct': 42.06, 'weight': 49.3},
    'COST': {'current_value': 6019.21, 'gain': 497.24, 'gain_pct': 14.35, 'weight': 50.0}
}

def calculate_features(df):
    """Calculate AI features for prediction"""
    try:
        # Technical indicators
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['rsi'] = calculate_rsi(df['Close'])
        df['bb_position'] = calculate_bb_position(df['Close'])
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['volatility'] = df['Close'].rolling(window=20).std()
        
        # Price momentum
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Advanced features
        df['price_vs_sma20'] = df['Close'] / df['sma_20'] - 1
        df['volume_momentum'] = df['volume_ratio'] * df['momentum_5']
        
        # Target: future 5-day return
        df['future_return'] = df['Close'].shift(-5) / df['Close'] - 1
        
        return df
    except Exception as e:
        print(f"Error calculating features: {e}")
        return df

def calculate_rsi(prices, window=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bb_position(prices, window=20):
    """Calculate Bollinger Band position"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return (prices - lower) / (upper - lower)

def train_ai_model(data):
    """Train a quick AI model on the data"""
    try:
        # Feature columns
        feature_cols = ['sma_5', 'sma_20', 'rsi', 'bb_position', 'volume_ratio', 
                       'volatility', 'momentum_5', 'momentum_20', 'price_vs_sma20', 'volume_momentum']
        
        # Prepare data
        clean_data = data[feature_cols + ['future_return']].dropna()
        
        if len(clean_data) < 50:
            return None, None
        
        X = clean_data[feature_cols]
        y = clean_data['future_return']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Test accuracy
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"  Model trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        return model, scaler
        
    except Exception as e:
        print(f"  Error training model: {e}")
        return None, None

def make_ai_prediction(model, scaler, data):
    """Make AI prediction on current data"""
    try:
        feature_cols = ['sma_5', 'sma_20', 'rsi', 'bb_position', 'volume_ratio', 
                       'volatility', 'momentum_5', 'momentum_20', 'price_vs_sma20', 'volume_momentum']
        
        # Get latest features
        latest_features = data[feature_cols].iloc[-1:].values
        
        # Scale and predict
        latest_scaled = scaler.transform(latest_features)
        prediction = model.predict(latest_scaled)[0]
        
        return prediction
        
    except Exception as e:
        print(f"  Prediction error: {e}")
        return 0

def analyze_stock_ai(symbol):
    """Analyze a single stock with AI"""
    try:
        print(f"\n🧠 AI Analysis for {symbol}:")
        
        # Get data
        stock = yf.Ticker(symbol)
        df = stock.history(period='1y')
        
        if df.empty:
            print(f"  ❌ No data available")
            return None
        
        # Calculate features
        df = calculate_features(df)
        
        # Train AI model
        model, scaler = train_ai_model(df)
        
        if model is None:
            print(f"  ❌ Could not train model")
            return None
        
        # Make prediction
        ai_prediction = make_ai_prediction(model, scaler, df)
        
        # Convert to percentage and signal
        ai_score_pct = ai_prediction * 100
        
        if ai_score_pct > 5:
            ai_signal = "🟢 STRONG BUY"
        elif ai_score_pct > 2:
            ai_signal = "🔵 BUY"
        elif ai_score_pct > -2:
            ai_signal = "🟡 HOLD"
        elif ai_score_pct > -5:
            ai_signal = "🔴 SELL"
        else:
            ai_signal = "🔴 STRONG SELL"
        
        # Current technical info
        latest = df.iloc[-1]
        current_price = latest['Close']
        rsi = latest['rsi']
        momentum_5 = latest['momentum_5'] * 100
        
        result = {
            'symbol': symbol,
            'ai_prediction': ai_score_pct,
            'ai_signal': ai_signal,
            'current_price': current_price,
            'rsi': rsi,
            'momentum_5': momentum_5
        }
        
        print(f"  🤖 AI Prediction: {ai_score_pct:+.2f}% | Signal: {ai_signal}")
        print(f"  📊 RSI: {rsi:.1f} | 5-day momentum: {momentum_5:+.1f}%")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Analysis failed: {e}")
        return None

def main():
    """Main analysis function"""
    print("🤖 QUICK AI PORTFOLIO ANALYSIS")
    print("=" * 60)
    print("Training individual AI models for each stock...")
    
    results = []
    successful_predictions = 0
    
    for symbol, holdings in PORTFOLIO.items():
        ai_result = analyze_stock_ai(symbol)
        
        if ai_result:
            ai_result.update(holdings)
            results.append(ai_result)
            successful_predictions += 1
    
    # Summary Analysis
    print("\n" + "=" * 60)
    print("📊 AI PORTFOLIO SUMMARY")
    print("=" * 60)
    
    print(f"🎯 AI Models Trained: {successful_predictions}/{len(PORTFOLIO)}")
    
    if results:
        # Sort by actual performance
        results_by_performance = sorted(results, key=lambda x: x['gain_pct'], reverse=True)
        
        print("\n🔍 AI PREDICTIONS vs ACTUAL PERFORMANCE:")
        print("-" * 60)
        print("Stock | Actual Gain | AI Prediction | AI Signal | Match")
        print("-" * 60)
        
        correct_predictions = 0
        total_predictions = 0
        
        for r in results_by_performance:
            # Check if AI prediction direction matches actual performance
            actual_positive = r['gain_pct'] > 10  # Good performance threshold
            ai_positive = r['ai_prediction'] > 2   # AI bullish threshold
            
            match = "✅" if (actual_positive == ai_positive) else "❌"
            if actual_positive == ai_positive:
                correct_predictions += 1
            total_predictions += 1
            
            print(f"{r['symbol']:4} | {r['gain_pct']:8.1f}% | {r['ai_prediction']:10.2f}% | {r['ai_signal']:13} | {match}")
        
        # AI Accuracy
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        print(f"\n🎯 AI Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
        
        # Top AI Recommendations
        print("\n🏆 TOP AI RECOMMENDATIONS:")
        print("-" * 40)
        ai_sorted = sorted(results, key=lambda x: x['ai_prediction'], reverse=True)
        
        for i, r in enumerate(ai_sorted[:5]):
            print(f"{i+1}. {r['symbol']}: {r['ai_signal']} ({r['ai_prediction']:+.2f}%)")
        
        # Performance insights
        avg_prediction = np.mean([r['ai_prediction'] for r in results])
        print(f"\n📈 Average AI Prediction: {avg_prediction:+.2f}%")
        
        # Correlation analysis
        actual_gains = [r['gain_pct'] for r in results]
        ai_predictions = [r['ai_prediction'] for r in results]
        correlation = np.corrcoef(actual_gains, ai_predictions)[0,1]
        
        print(f"📊 Correlation (AI vs Actual): {correlation:.3f}")

if __name__ == "__main__":
    try:
        main()
        print(f"\n✅ AI Portfolio analysis complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
