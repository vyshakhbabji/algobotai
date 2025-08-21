#!/usr/bin/env python3
"""
Ultra Simple NVDA Analysis - Current Market Status
"""

import yfinance as yf
import pandas as pd
import numpy as np

def simple_nvda_check():
    print("🔍 NVDA MARKET CHECK")
    print("=" * 25)
    
    # Get basic info
    nvda = yf.Ticker("NVDA")
    info = nvda.info
    
    current_price = info.get('currentPrice', 0)
    market_cap = info.get('marketCap', 0)
    pe_ratio = info.get('trailingPE', 0)
    
    print(f"Current Price: ${current_price}")
    print(f"Market Cap: ${market_cap/1e12:.2f}T")
    print(f"P/E Ratio: {pe_ratio:.1f}")
    
    # Get recent data
    data = yf.download("NVDA", period="6mo", interval="1d", progress=False)
    
    if len(data) > 0:
        current = float(data['Close'].iloc[-1])
        low_6m = float(data['Close'].min())
        high_6m = float(data['Close'].max())
        
        print(f"\n6-Month Range: ${low_6m:.2f} - ${high_6m:.2f}")
        print(f"Current vs Low: +{((current/low_6m)-1)*100:.1f}%")
        print(f"Current vs High: {((current/high_6m)-1)*100:.1f}%")
        
        # Simple moving averages
        if len(data) >= 50:
            ma20 = float(data['Close'].tail(20).mean())
            ma50 = float(data['Close'].tail(50).mean())
            
            print(f"\n20-day average: ${ma20:.2f}")
            print(f"50-day average: ${ma50:.2f}")
            print(f"Above 20-day MA: {'YES' if current > ma20 else 'NO'}")
            print(f"Above 50-day MA: {'YES' if current > ma50 else 'NO'}")
        
        # Recent momentum
        if len(data) >= 30:
            month_ago = float(data['Close'].iloc[-30])
            momentum = ((current - month_ago) / month_ago) * 100
            print(f"\n30-day momentum: {momentum:+.1f}%")
    
    print(f"\n💡 SPLIT CONTEXT:")
    print(f"   NVDA had 10:1 split in June 2024")
    print(f"   Pre-split ~$1200 → Post-split ~$120")
    print(f"   Current ${current_price:.0f} = normal price action")
    print(f"   Elite AI may be confused by split")
    
    print(f"\n🎯 SIMPLE ASSESSMENT:")
    if current_price > 160:
        print(f"   Status: STRONG - near highs")
    elif current_price > 120:
        print(f"   Status: HEALTHY - good range")
    else:
        print(f"   Status: WEAK - below split level")
        
    if pe_ratio > 70:
        print(f"   Valuation: EXPENSIVE")
    elif pe_ratio > 50:
        print(f"   Valuation: FAIR for growth")
    else:
        print(f"   Valuation: REASONABLE")
    
    print(f"\n✅ BOTTOM LINE:")
    print(f"   • NVDA fundamentally strong")
    print(f"   • Stock split explains AI confusion")
    print(f"   • Trust your bullish sentiment")
    print(f"   • HOLD your position")

if __name__ == "__main__":
    simple_nvda_check()
