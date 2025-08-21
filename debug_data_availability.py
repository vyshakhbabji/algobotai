#!/usr/bin/env python3
"""
Debug the forward test setup
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def check_data_availability():
    """Check what data is available for NVDA and other stocks"""
    
    # Test period
    start_date = "2024-05-13"
    end_date = "2024-08-13"
    
    # Lookback for training (2 years before start)
    lookback_start = pd.to_datetime(start_date) - timedelta(days=730)
    lookback_start_str = lookback_start.strftime('%Y-%m-%d')
    
    print(f"Test period: {start_date} to {end_date}")
    print(f"Training data period: {lookback_start_str} to {start_date}")
    print("=" * 60)
    
    # Check NVDA data
    print("\nChecking NVDA data:")
    try:
        nvda_train = yf.download('NVDA', start=lookback_start_str, end=start_date, progress=False)
        nvda_test = yf.download('NVDA', start=start_date, end=end_date, progress=False)
        
        print(f"Training data: {len(nvda_train)} days ({nvda_train.index[0]} to {nvda_train.index[-1]})")
        print(f"Test data: {len(nvda_test)} days ({nvda_test.index[0]} to {nvda_test.index[-1]})")
        
        if len(nvda_test) > 0:
            start_price = float(nvda_test['Close'].iloc[0])
            end_price = float(nvda_test['Close'].iloc[-1])
            print(f"Start price: ${start_price:.2f}")
            print(f"End price: ${end_price:.2f}")
            print(f"Return: {(end_price/start_price - 1):.2%}")
            
    except Exception as e:
        print(f"Error getting NVDA data: {e}")
    
    # Check a few other symbols from the scan
    symbols = ['NFLX', 'PM', 'T', 'AVGO', 'COST']
    print(f"\nChecking other symbols: {symbols}")
    
    for symbol in symbols:
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if len(data) > 0:
                start_price = float(data['Close'].iloc[0])
                end_price = float(data['Close'].iloc[-1])
                ret = (end_price / start_price - 1)
                print(f"{symbol}: {len(data)} days, return: {ret:.2%}")
        except Exception as e:
            print(f"{symbol}: Error - {e}")

if __name__ == '__main__':
    check_data_availability()
