"""
🎯 SIMPLE TEST WITH WORKING ALPACA DATES
Use the successful approach but with proper historical dates
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import traceback

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score

# Test Alpaca connection first
def test_alpaca_data():
    try:
        # Load Alpaca credentials
        config_path = Path(__file__).parent / "alpaca_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(
            config['alpaca']['api_key'],
            config['alpaca']['secret_key'],
            config['alpaca']['base_url'],
            api_version='v2'
        )
        
        # Test what dates work - including up to August 20, 2025
        test_dates = [
            ("2024-01-01", "2025-06-01"),  # Training period (17 months)
            ("2025-06-02", "2025-08-20"),  # Trading period (2.5 months until yesterday)
            ("2024-01-01", "2025-08-20"),  # Full period test
            ("2025-08-01", "2025-08-20"),  # Recent August data
        ]
        
        print("🔍 Testing Alpaca data availability:")
        
        for start, end in test_dates:
            try:
                bars = api.get_bars(
                    "AAPL",
                    timeframe="1Day",
                    start=start,
                    end=end,
                    limit=10
                )
                data = bars.df
                print(f"   ✅ {start} to {end}: {len(data)} days available")
                
            except Exception as e:
                print(f"   ❌ {start} to {end}: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Alpaca test failed: {e}")
        return False

if __name__ == "__main__":
    test_alpaca_data()
