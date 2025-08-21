#!/usr/bin/env python3
"""
Debug script to identify exactly where the pandas Series ambiguity error is occurring
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def debug_series_ambiguity():
    print("🔍 DEBUGGING PANDAS SERIES AMBIGUITY")
    print("=" * 50)
    
    # Download sample data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print("📊 Downloading AAPL data...")
    data = yf.download('AAPL', start=start_date, end=end_date, progress=False)
    
    print(f"✅ Data shape: {data.shape}")
    print(f"📈 Columns: {list(data.columns)}")
    
    close_prices = data['Close']
    volumes = data['Volume']
    
    print(f"\n🔍 Testing basic operations...")
    
    # Test simple calculations
    try:
        print("Test 1: Basic price calculation")
        if len(close_prices) >= 21:
            result = (close_prices.iloc[-1] - close_prices.iloc[-21]) / close_prices.iloc[-21]
            print(f"   ✅ Momentum 21d: {result}")
            print(f"   ✅ Type: {type(result)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    try:
        print("\nTest 2: Volatility calculation")
        if len(close_prices) >= 21:
            vol = close_prices.iloc[-21:].std() / close_prices.iloc[-21:].mean()
            print(f"   ✅ Volatility: {vol}")
            print(f"   ✅ Type: {type(vol)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    try:
        print("\nTest 3: Volume ratio calculation")
        if len(volumes) >= 42:
            ratio = volumes.iloc[-21:].mean() / volumes.iloc[-42:-21].mean()
            print(f"   ✅ Volume ratio: {ratio}")
            print(f"   ✅ Type: {type(ratio)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    try:
        print("\nTest 4: Trend consistency calculation")
        if len(close_prices) >= 10:
            recent_returns = close_prices.iloc[-10:].pct_change().dropna()
            print(f"   📊 Recent returns type: {type(recent_returns)}")
            print(f"   📊 Recent returns shape: {recent_returns.shape}")
            
            positive_days = (recent_returns > 0).sum()
            print(f"   📊 Positive days type: {type(positive_days)}")
            print(f"   📊 Positive days value: {positive_days}")
            
            trend_consistency = positive_days / len(recent_returns) if len(recent_returns) > 0 else 0.5
            print(f"   ✅ Trend consistency: {trend_consistency}")
            print(f"   ✅ Type: {type(trend_consistency)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    try:
        print("\nTest 5: Label creation test")
        forward_days = 21
        if len(close_prices) >= forward_days + 30:
            for i in range(30, min(35, len(close_prices) - forward_days)):  # Just test a few
                current_price = close_prices.iloc[i]
                future_price = close_prices.iloc[i + forward_days]
                forward_return = (future_price - current_price) / current_price
                
                print(f"   Index {i}: current={current_price}, future={future_price}, return={forward_return}")
                print(f"   Types: current={type(current_price)}, future={type(future_price)}, return={type(forward_return)}")
                
                # This is likely where the error occurs
                if forward_return > 0.03:
                    label = 1
                else:
                    label = 0
                print(f"   ✅ Label: {label}")
                break  # Only test first iteration
    except Exception as e:
        print(f"   ❌ Error in label creation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_series_ambiguity()
