#!/usr/bin/env python3
"""
Simple NVDA Current Analysis
Focus on current market conditions and AI model reliability
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_nvda_current():
    """Current NVDA analysis without complex AI models"""
    print("🔍 NVDA CURRENT MARKET ANALYSIS")
    print("=" * 40)
    
    # Get current data
    nvda = yf.Ticker("NVDA")
    info = nvda.info
    
    current_price = info.get('currentPrice', 0)
    market_cap = info.get('marketCap', 0)
    pe_ratio = info.get('trailingPE', 0)
    
    print(f"💰 Current Price: ${current_price:.2f}")
    print(f"📈 Market Cap: ${market_cap/1e12:.2f}T")
    print(f"📊 P/E Ratio: {pe_ratio:.1f}")
    
    # Get price history
    data = yf.download("NVDA", period="1y", interval="1d")
    
    # Check for split in data
    print(f"\n📊 SPLIT VERIFICATION:")
    print(f"   Data points: {len(data)}")
    min_price = float(data['Close'].min())
    max_price = float(data['Close'].max())
    print(f"   Price range: ${min_price:.2f} - ${max_price:.2f}")
    
    # Look for major price changes that could indicate split
    daily_changes = data['Close'].pct_change()
    large_changes = daily_changes[abs(daily_changes) > 0.5]  # 50%+ changes
    
    print(f"\n🔍 LARGE PRICE MOVEMENTS (>50%):")
    if len(large_changes) > 0:
        for date, change in large_changes.items():
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            print(f"   {date_str}: {change:.1%}")
    else:
        print("   No major single-day moves detected")
    
    # Recent performance
    recent_30d = data['Close'].iloc[-30:] if len(data) >= 30 else data['Close']
    recent_return = ((recent_30d.iloc[-1] - recent_30d.iloc[0]) / recent_30d.iloc[0]) * 100
    
    ytd_data = data[data.index >= '2025-01-01'] if len(data) > 0 else pd.Series()
    if len(ytd_data) > 0:
        ytd_return = ((ytd_data['Close'].iloc[-1] - ytd_data['Close'].iloc[0]) / ytd_data['Close'].iloc[0]) * 100
    else:
        ytd_return = 0
    
    print(f"\n📈 RECENT PERFORMANCE:")
    print(f"   30-day return: {recent_return:+.1f}%")
    print(f"   YTD 2025 return: {ytd_return:+.1f}%")
    
    # Simple technical indicators
    ma_20 = float(data['Close'].rolling(20).mean().iloc[-1])
    ma_50 = float(data['Close'].rolling(50).mean().iloc[-1]) 
    
    print(f"\n📊 TECHNICAL INDICATORS:")
    print(f"   20-day MA: ${ma_20:.2f} {'🟢' if current_price > ma_20 else '🔴'}")
    print(f"   50-day MA: ${ma_50:.2f} {'🟢' if current_price > ma_50 else '🔴'}")
    
    # Volatility
    volatility = daily_changes.std() * np.sqrt(252) * 100
    print(f"   Volatility: {volatility:.1f}%")
    
    # Position recommendation based on simple factors
    print(f"\n🎯 SIMPLE POSITION ASSESSMENT:")
    
    bullish_factors = 0
    if current_price > ma_20:
        bullish_factors += 1
        print(f"   ✅ Above 20-day MA")
    else:
        print(f"   ❌ Below 20-day MA")
        
    if current_price > ma_50:
        bullish_factors += 1
        print(f"   ✅ Above 50-day MA")
    else:
        print(f"   ❌ Below 50-day MA")
        
    if recent_return > 0:
        bullish_factors += 1
        print(f"   ✅ Positive 30-day momentum")
    else:
        print(f"   ❌ Negative 30-day momentum")
    
    if pe_ratio < 70:  # Reasonable for growth stock
        bullish_factors += 1
        print(f"   ✅ Reasonable P/E for growth stock")
    else:
        print(f"   ❌ High P/E ratio")
    
    print(f"\n📊 BULLISH FACTORS: {bullish_factors}/4")
    
    if bullish_factors >= 3:
        recommendation = "🟢 HOLD/BUY"
    elif bullish_factors >= 2:
        recommendation = "🟡 CAUTIOUS HOLD" 
    else:
        recommendation = "🔴 CONSIDER REDUCING"
        
    print(f"🎯 SIMPLE RECOMMENDATION: {recommendation}")
    
    print(f"\n💡 ABOUT ELITE AI CONFUSION:")
    print(f"   • If Elite AI shows SELL with low confidence")
    print(f"   • It's likely due to training data issues")
    print(f"   • Stock splits can break ML models")
    print(f"   • Trust fundamental analysis over AI")

if __name__ == "__main__":
    analyze_nvda_current()
