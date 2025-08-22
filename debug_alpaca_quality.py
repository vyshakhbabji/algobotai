"""
🔍 DEBUG ALPACA STOCK QUALITY CHECKER
Let's see why all stocks are failing the quality checks
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import alpaca_trade_api as tradeapi

# Suppress warnings
warnings.filterwarnings('ignore')

class AlpacaDebugger:
    def __init__(self):
        # Load Alpaca credentials
        config_path = Path(__file__).parent / "alpaca_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize Alpaca API
        alpaca_config = config['alpaca']
        self.api = tradeapi.REST(
            alpaca_config['api_key'],
            alpaca_config['secret_key'],
            alpaca_config['base_url'],
            api_version='v2'
        )
        
        print("🔍 Alpaca Debugger initialized")
    
    def fetch_and_analyze(self, symbol: str, start_date: str, end_date: str):
        """Fetch and analyze a single stock"""
        try:
            print(f"\n📊 Analyzing {symbol}:")
            
            # Get bars from Alpaca
            bars = self.api.get_bars(
                symbol,
                timeframe='1Day',
                start=start_date,
                end=end_date,
                adjustment='all'
            ).df
            
            if bars.empty:
                print("   ❌ No data returned from Alpaca")
                return
            
            # Rename columns
            bars = bars.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Remove timezone
            bars.index = bars.index.tz_localize(None)
            
            print(f"   📈 Total days: {len(bars)}")
            print(f"   📅 Date range: {bars.index[0]} to {bars.index[-1]}")
            
            # Check recent data (last 30 days)
            recent_data = bars.tail(30)
            print(f"   📅 Recent 30 days: {len(recent_data)} rows")
            
            # Price analysis
            avg_price = recent_data['Close'].mean()
            min_price = recent_data['Close'].min()
            max_price = recent_data['Close'].max()
            print(f"   💰 Price: avg=${avg_price:.2f}, min=${min_price:.2f}, max=${max_price:.2f}")
            
            # Volume analysis
            avg_volume = recent_data['Volume'].mean()
            avg_dollar_volume = (recent_data['Close'] * recent_data['Volume']).mean()
            print(f"   📊 Volume: avg={avg_volume:,.0f} shares, avg_dollar=${avg_dollar_volume:,.0f}")
            
            # Volatility analysis
            returns = recent_data['Close'].pct_change().dropna()
            volatility = returns.std()
            print(f"   📈 Volatility: {volatility:.4f} ({volatility:.2%})")
            
            # Data completeness
            missing_days = recent_data['Close'].isna().sum()
            print(f"   🔍 Missing days in last 30: {missing_days}")
            
            # Check against ADJUSTED filters
            print(f"\n   🎯 Quality Check Results (ADJUSTED):")
            
            # Min days check (60 for 3-month period)
            days_check = len(bars) >= 60
            print(f"   ✅ Min 60 days: {days_check} ({len(bars)} days)")
            
            # Price range check ($5-$500)
            price_check = 5.0 <= avg_price <= 500.0
            print(f"   {'✅' if price_check else '❌'} Price range $5-$500: {price_check} (${avg_price:.2f})")
            
            # Volume check ($1M+ daily)
            volume_check = avg_dollar_volume >= 1000000
            print(f"   {'✅' if volume_check else '❌'} Min $1M volume: {volume_check} (${avg_dollar_volume:,.0f})")
            
            # Volatility check (1%-15% - ADJUSTED)
            vol_check = 0.01 <= volatility <= 0.15
            print(f"   {'✅' if vol_check else '❌'} Volatility 1%-15%: {vol_check} ({volatility:.2%})")
            
            # Missing data check (max 3 missing)
            missing_check = missing_days <= 3
            print(f"   {'✅' if missing_check else '❌'} Max 3 missing days: {missing_check} ({missing_days} missing)")
            
            # Overall pass
            overall_pass = days_check and price_check and volume_check and vol_check and missing_check
            print(f"   {'🎉' if overall_pass else '❌'} OVERALL PASS: {overall_pass}")
            
            return overall_pass
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

def main():
    debugger = AlpacaDebugger()
    
    # Test a few representative stocks
    test_symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'PLTR']
    
    end_date = "2024-12-06"
    start_date = "2024-09-06"  # 3 months back
    
    print(f"🔍 Debugging Alpaca stock quality for period: {start_date} to {end_date}")
    
    passed_count = 0
    for symbol in test_symbols:
        passed = debugger.fetch_and_analyze(symbol, start_date, end_date)
        if passed:
            passed_count += 1
    
    print(f"\n📊 SUMMARY:")
    print(f"   Tested: {len(test_symbols)} stocks")
    print(f"   Passed: {passed_count} stocks")
    print(f"   Failed: {len(test_symbols) - passed_count} stocks")
    
    if passed_count == 0:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   - Reduce minimum volume requirement")
        print(f"   - Adjust volatility range")
        print(f"   - Check date range (need more history?)")

if __name__ == "__main__":
    main()
