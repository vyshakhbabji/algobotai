#!/usr/bin/env python3
"""
Portfolio Technical Analysis
Analyzes your portfolio using technical indicators when AI models are unavailable
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

def calculate_technical_score(df):
    """Calculate comprehensive technical analysis score"""
    try:
        latest = df.iloc[-1]
        
        # Price momentum (0-100 scale)
        momentum_1w = (latest['Close'] / df['Close'].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
        momentum_1m = (latest['Close'] / df['Close'].iloc[-20] - 1) * 100 if len(df) >= 20 else 0
        
        # Moving averages
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else latest['Close']
        sma_50 = df['Close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else latest['Close']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1] if len(gain) >= 14 else 50
        
        # Bollinger Bands
        bb_middle = df['Close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else latest['Close']
        bb_std = df['Close'].rolling(window=20).std().iloc[-1] if len(df) >= 20 else 0
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        bb_position = (latest['Close'] - bb_lower) / (bb_upper - bb_lower) if bb_std > 0 else 0.5
        
        # Volume analysis
        avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else latest['Volume']
        volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1
        
        # Calculate technical score (0-100)
        score = 0
        
        # Momentum signals (40 points max)
        if momentum_1w > 5: score += 10
        elif momentum_1w > 2: score += 5
        elif momentum_1w < -5: score -= 10
        elif momentum_1w < -2: score -= 5
        
        if momentum_1m > 10: score += 15
        elif momentum_1m > 5: score += 8
        elif momentum_1m < -10: score -= 15
        elif momentum_1m < -5: score -= 8
        
        # Price vs moving averages (20 points max)
        if latest['Close'] > sma_20 > sma_50: score += 10
        elif latest['Close'] > sma_20: score += 5
        elif latest['Close'] < sma_20 < sma_50: score -= 10
        elif latest['Close'] < sma_20: score -= 5
        
        # RSI signals (20 points max)
        if 30 <= rsi <= 70: score += 10  # Normal range
        elif rsi > 80: score -= 10  # Overbought
        elif rsi < 20: score += 5   # Oversold (potential bounce)
        elif rsi > 70: score -= 5   # Slight overbought
        elif rsi < 30: score += 3   # Slight oversold
        
        # Bollinger Bands (10 points max)
        if 0.2 <= bb_position <= 0.8: score += 5  # Normal range
        elif bb_position > 0.9: score -= 5  # Near upper band
        elif bb_position < 0.1: score += 3  # Near lower band
        
        # Volume confirmation (10 points max)
        if volume_ratio > 1.5: score += 5  # High volume
        elif volume_ratio > 1.2: score += 3
        elif volume_ratio < 0.8: score -= 3  # Low volume
        
        # Normalize to 0-100 range
        score = max(0, min(100, score + 50))
        
        return {
            'technical_score': score,
            'momentum_1w': momentum_1w,
            'momentum_1m': momentum_1m,
            'rsi': rsi,
            'bb_position': bb_position,
            'volume_ratio': volume_ratio,
            'price_vs_sma20': (latest['Close'] / sma_20 - 1) * 100 if sma_20 > 0 else 0
        }
        
    except Exception as e:
        print(f"Error calculating technical score: {e}")
        return {
            'technical_score': 50,
            'momentum_1w': 0,
            'momentum_1m': 0,
            'rsi': 50,
            'bb_position': 0.5,
            'volume_ratio': 1,
            'price_vs_sma20': 0
        }

def get_stock_data(symbol, period='6mo'):
    """Fetch stock data"""
    try:
        print(f"📈 Analyzing {symbol}...")
        
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        if df.empty:
            print(f"❌ No data for {symbol}")
            return None
        
        print(f"✅ Got {len(df)} days of data")
        return df
    
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None

def analyze_portfolio():
    """Analyze portfolio using technical analysis"""
    print("📊 PORTFOLIO TECHNICAL ANALYSIS")
    print("=" * 60)
    
    results = []
    total_value = 0
    total_gain = 0
    
    for symbol, holdings in PORTFOLIO.items():
        print(f"\n🔍 {symbol}:")
        
        # Get stock data
        data = get_stock_data(symbol)
        if data is None:
            continue
        
        # Calculate technical analysis
        tech_analysis = calculate_technical_score(data)
        
        # Current price info
        current_price = data['Close'].iloc[-1]
        price_change_1d = (current_price / data['Close'].iloc[-2] - 1) * 100 if len(data) > 1 else 0
        
        # Technical signal
        score = tech_analysis['technical_score']
        if score >= 75:
            signal = "🟢 STRONG BUY"
        elif score >= 60:
            signal = "🔵 BUY"
        elif score >= 40:
            signal = "🟡 HOLD"
        elif score >= 25:
            signal = "🔴 SELL"
        else:
            signal = "🔴 STRONG SELL"
        
        result = {
            'symbol': symbol,
            'current_value': holdings['current_value'],
            'gain_pct': holdings['gain_pct'],
            'technical_score': score,
            'signal': signal,
            'current_price': current_price,
            'price_change_1d': price_change_1d,
            **tech_analysis
        }
        
        results.append(result)
        
        # Display analysis
        print(f"  💰 Value: ${holdings['current_value']:,.2f} (+{holdings['gain_pct']:.1f}%)")
        print(f"  📊 Technical Score: {score:.1f}/100 | Signal: {signal}")
        print(f"  📈 Momentum: 1W: {tech_analysis['momentum_1w']:.1f}% | 1M: {tech_analysis['momentum_1m']:.1f}%")
        print(f"  🎯 RSI: {tech_analysis['rsi']:.1f} | vs SMA20: {tech_analysis['price_vs_sma20']:.1f}%")
        
        total_value += holdings['current_value']
        total_gain += holdings['gain']
    
    # Portfolio Summary
    print("\n" + "=" * 60)
    print("📈 PORTFOLIO SUMMARY")
    print("=" * 60)
    
    total_gain_pct = (total_gain / (total_value - total_gain)) * 100 if total_value > total_gain else 0
    avg_tech_score = np.mean([r['technical_score'] for r in results]) if results else 0
    
    print(f"💰 Total Value: ${total_value:,.2f}")
    print(f"📈 Total Gains: ${total_gain:,.2f} (+{total_gain_pct:.1f}%)")
    print(f"📊 Avg Technical Score: {avg_tech_score:.1f}/100")
    print(f"🎯 Stocks Analyzed: {len(results)}/{len(PORTFOLIO)}")
    
    # Top technical recommendations
    if results:
        print("\n🏆 TOP TECHNICAL SIGNALS:")
        print("-" * 40)
        results_sorted = sorted(results, key=lambda x: x['technical_score'], reverse=True)
        
        for i, r in enumerate(results_sorted[:5]):
            print(f"{i+1}. {r['symbol']}: {r['signal']} (Score: {r['technical_score']:.1f})")
    
    # Performance vs Technical Analysis
    if results:
        print("\n🔍 ACTUAL vs TECHNICAL ANALYSIS:")
        print("-" * 50)
        
        for r in sorted(results, key=lambda x: x['gain_pct'], reverse=True):
            # Check if technical analysis would have been bullish on winners
            tech_match = "✅" if (r['technical_score'] > 60 and r['gain_pct'] > 10) or (r['technical_score'] < 40 and r['gain_pct'] < 0) else "❌"
            print(f"{r['symbol']}: {tech_match} Gain: +{r['gain_pct']:.1f}% | Tech Score: {r['technical_score']:.1f}")
    
    return results

if __name__ == "__main__":
    try:
        results = analyze_portfolio()
        print(f"\n✅ Technical analysis complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
