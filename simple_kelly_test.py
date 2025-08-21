#!/usr/bin/env python3
"""
SIMPLE KELLY TEST - Copy exact working logic but add Kelly sizing
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

def calculate_indicators(df):
    """Calculate technical indicators"""
    d = df.copy()
    
    # Moving averages
    d['MA5'] = d['Close'].rolling(5).mean()
    d['MA10'] = d['Close'].rolling(10).mean()
    d['MA20'] = d['Close'].rolling(20).mean()
    
    # RSI
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    d['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = d['High'] - d['Low']
    high_close = np.abs(d['High'] - d['Close'].shift())
    low_close = np.abs(d['Low'] - d['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    d['ATR'] = tr.rolling(14).mean()
    
    # Volume
    d['Volume_MA'] = d['Volume'].rolling(20).mean()
    d['Volume_Ratio'] = d['Volume'] / d['Volume_MA']
    
    return d

def generate_trading_signal(symbol, all_data, current_date):
    """EXACT copy of working signal generation"""
    
    # CRITICAL: Only use data up to current_date (no future leakage)
    historical_data = all_data[symbol].loc[:current_date]
    
    if len(historical_data) < 30:
        return {'signal': 'HOLD', 'strength': 0.0, 'price': 0}
    
    # Calculate indicators
    df = calculate_indicators(historical_data)
    
    if len(df) < 20:
        return {'signal': 'HOLD', 'strength': 0.0, 'price': 0}
    
    # Get current values
    latest = df.iloc[-1]
    current_price = float(latest['Close'])
    
    # Technical signal components (EXACT working logic)
    buy_strength = 0.0
    sell_strength = 0.0
    
    # RSI signals
    rsi = latest['RSI'] if not pd.isna(latest['RSI']) else 50
    if rsi < 35:  # Oversold
        buy_strength += 0.25
    elif rsi > 65:  # Overbought
        sell_strength += 0.25
    
    # Moving average signals
    ma5 = latest['MA5'] if not pd.isna(latest['MA5']) else current_price
    ma10 = latest['MA10'] if not pd.isna(latest['MA10']) else current_price
    ma20 = latest['MA20'] if not pd.isna(latest['MA20']) else current_price
    
    # Trend signals
    if current_price > ma5 > ma20:  # Strong uptrend
        buy_strength += 0.3
    elif current_price < ma5 < ma20:  # Strong downtrend
        sell_strength += 0.3
    
    # Momentum signals
    if len(df) >= 10:
        momentum_5d = (current_price - df['Close'].iloc[-6]) / df['Close'].iloc[-6]
        momentum_10d = (current_price - df['Close'].iloc[-11]) / df['Close'].iloc[-11]
        
        if momentum_5d > 0.03 and momentum_10d > 0.05:  # Strong momentum
            buy_strength += 0.25
        elif momentum_5d < -0.03 and momentum_10d < -0.05:
            sell_strength += 0.25
    
    # Volume confirmation
    volume_ratio = latest['Volume_Ratio'] if not pd.isna(latest['Volume_Ratio']) else 1.0
    if volume_ratio > 1.3:  # Above average volume
        buy_strength *= 1.1
        sell_strength *= 1.1
    
    # Determine final signal (EXACT working logic)
    signal = 'HOLD'
    strength = 0.0
    threshold = 0.35  # EXACT threshold from working system
    
    if buy_strength > threshold and buy_strength > sell_strength:
        signal = 'BUY'
        strength = min(1.0, buy_strength)
    elif sell_strength > threshold and sell_strength > buy_strength:
        signal = 'SELL'
        strength = min(1.0, sell_strength)
    
    return {
        'signal': signal,
        'strength': strength,
        'price': current_price,
        'rsi': rsi,
        'buy_strength': buy_strength,
        'sell_strength': sell_strength,
        'volume_ratio': volume_ratio
    }

def calculate_kelly_position_size(signal_strength, base_kelly=0.08):
    """Calculate Kelly position size"""
    strength_multiplier = min(signal_strength / 0.35, 2.5)
    kelly_multiplier = 1.5
    position_size = base_kelly * strength_multiplier * kelly_multiplier
    return max(0.02, min(position_size, 0.15))

def simple_kelly_test():
    """Simple test with exact working logic"""
    
    print("🚀 SIMPLE KELLY TEST")
    print("Using EXACT working signal logic + Kelly sizing")
    print("="*60)
    
    # Use exact same symbols as working system
    symbols = ['PLTR', 'NVDA', 'TSLA', 'AVGO', 'AMD', 'INTC', 'MU', 'META', 'LRCX', 
              'AAPL', 'GOOGL', 'AMZN', 'BAC', 'ORCL', 'RBLX', 'SHOP', 'U', 'NET', 
              'GS', 'NKE', 'NFLX', 'DIS', 'MS', 'SBUX', 'COIN']
    
    # Download data
    print("📥 Downloading data...")
    train_start = "2023-11-21"
    test_end = "2025-08-20"
    
    all_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=train_start, end=test_end)
            if len(df) > 50:
                all_data[symbol] = df
                print(f"   ✅ {symbol}: {len(df)} days")
        except Exception as e:
            print(f"   ❌ {symbol}: Error")
    
    print(f"✅ Downloaded {len(all_data)} symbols")
    
    # Test signals on first trading day (same as working system)
    test_date = "2025-05-22"
    print(f"\\n🎯 Testing signals for {test_date} (first trading day):")
    
    signals_generated = []
    
    for symbol in all_data.keys():
        try:
            signal_data = generate_trading_signal(symbol, all_data, test_date)
            
            if signal_data['signal'] != 'HOLD':
                kelly_size = calculate_kelly_position_size(signal_data['strength'])
                
                print(f"\\n{symbol}: {signal_data['signal']}")
                print(f"   Strength: {signal_data['strength']:.3f}")
                print(f"   Kelly Size: {kelly_size:.1%}")
                print(f"   Price: ${signal_data['price']:.2f}")
                print(f"   RSI: {signal_data['rsi']:.1f}")
                print(f"   Buy/Sell: {signal_data['buy_strength']:.3f}/{signal_data['sell_strength']:.3f}")
                
                signals_generated.append({
                    'symbol': symbol,
                    'signal': signal_data['signal'],
                    'strength': signal_data['strength'],
                    'kelly_size': kelly_size,
                    'price': signal_data['price']
                })
        
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {str(e)}")
    
    print(f"\\n📊 SUMMARY:")
    print(f"   Signals Generated: {len(signals_generated)}")
    
    if signals_generated:
        buy_signals = [s for s in signals_generated if s['signal'] == 'BUY']
        sell_signals = [s for s in signals_generated if s['signal'] == 'SELL']
        
        print(f"   Buy Signals: {len(buy_signals)}")
        print(f"   Sell Signals: {len(sell_signals)}")
        
        if buy_signals:
            total_kelly = sum(s['kelly_size'] for s in buy_signals)
            print(f"   Total Kelly Deployment: {total_kelly:.1%}")
    else:
        print("   ❌ NO SIGNALS GENERATED!")
        print("   This explains why Kelly system had 0 trades")
    
    # Compare with working system results
    print(f"\\n🔍 Comparing with working system...")
    try:
        with open('forward_test_results_20250820_232218.json', 'r') as f:
            working_results = json.load(f)
        
        first_day_trades = [t for t in working_results['trade_history'] if t['date'] == test_date]
        print(f"   Working system trades on {test_date}: {len(first_day_trades)}")
        
        for trade in first_day_trades:
            print(f"   {trade['action']}: {trade['symbol']} (strength: {trade['strength']:.3f})")
            
    except Exception as e:
        print(f"   ❌ Could not compare: {e}")

if __name__ == "__main__":
    simple_kelly_test()
