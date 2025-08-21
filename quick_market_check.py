#!/usr/bin/env python3
"""
Quick market check for current AI recommendations
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def check_current_market():
    # Get current prices for the recommended stocks
    symbols = ['TSLA', 'PLTR', 'AAPL']
    print('=== CURRENT MARKET ANALYSIS ===')
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print()
    
    total_budget = 5355.12
    recommendations = {
        'TSLA': 37.0,
        'PLTR': 65.0, 
        'AAPL': 55.0
    }
    
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period='5d')
            current_price = hist['Close'].iloc[-1]
            change_1d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100
            
            shares = recommendations[symbol]
            position_value = shares * current_price
            position_pct = (position_value / total_budget) * 100
            
            print(f'{symbol}: ${current_price:.2f} ({change_1d:+.2f}% today)')
            print(f'  Recommended: {shares} shares = ${position_value:.2f} ({position_pct:.1f}% of portfolio)')
            print()
            
        except Exception as e:
            print(f'{symbol}: Error getting data - {e}')
            print()

if __name__ == "__main__":
    check_current_market()
