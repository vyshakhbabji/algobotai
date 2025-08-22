"""
🔍 DEBUG VERSION - Check Daily Trading Activity
Let's see exactly what's happening each day
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

# Alpaca import
import alpaca_trade_api as tradeapi

warnings.filterwarnings('ignore')

def quick_debug_test():
    """Quick debug test to see daily activity"""
    
    # Load Alpaca credentials
    config_path = Path(__file__).parent / "alpaca_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    api = tradeapi.REST(
        config['alpaca']['api_key'],
        config['alpaca']['secret_key'],
        config['alpaca']['base_url'],
        api_version='v2'
    )
    
    print("🔍 DEBUG: Daily Trading Activity Check")
    print("=" * 50)
    
    # Get some data for AAPL
    symbol = "AAPL"
    training_start = "2024-01-01"
    trading_start = "2025-06-02"
    trading_end = "2025-06-10"  # Just first week
    
    print(f"📊 Testing {symbol} from {trading_start} to {trading_end}")
    
    try:
        # Get data
        bars = api.get_bars(symbol, timeframe="1Day", start=training_start, end=trading_end, limit=5000)
        data = bars.df.reset_index()
        data['date'] = data['timestamp'].dt.date
        data = data.set_index('date')
        
        # Rename columns to match yfinance format
        data = data.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        print(f"✅ Got {len(data)} days of data")
        print(f"📅 Data range: {data.index[0]} to {data.index[-1]}")
        
        # Show trading period data
        trading_dates = pd.bdate_range(start=trading_start, end=trading_end)
        print(f"\n📈 Trading dates to simulate: {len(trading_dates)}")
        
        for i, date in enumerate(trading_dates):
            date_obj = date.date()
            print(f"Day {i+1}: {date_obj}", end=" ")
            
            # Check if we have data for this date
            try:
                current_data = data.loc[:date_obj]
                if len(current_data) > 0:
                    latest_price = current_data['Close'].iloc[-1]
                    print(f"✅ Price: ${latest_price:.2f} (Data points: {len(current_data)})")
                else:
                    print("❌ No data")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Check specific date alignment issue
        print(f"\n🔍 Checking for date alignment issues:")
        test_date = pd.to_datetime("2025-06-02").date()
        print(f"Looking for date: {test_date}")
        print(f"Available dates around that time:")
        
        # Show dates in data around trading start
        trading_period_data = data.loc["2025-05-30":"2025-06-10"]
        print(f"Data available: {list(trading_period_data.index)}")
        
        if len(trading_period_data) == 0:
            print("❌ NO DATA in trading period - this is the problem!")
        else:
            print("✅ Data is available in trading period")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    quick_debug_test()
