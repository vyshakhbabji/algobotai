#!/usr/bin/env python3
"""
Check available data history for stocks in our universe
"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def check_stock_data_availability():
    """Check how much historical data is available for each stock"""
    
    stocks = [
        # TECH GIANTS
        "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN",
        # VOLATILE GROWTH  
        "TSLA", "META", "NFLX", "CRM", "UBER",
        # TRADITIONAL VALUE
        "JPM", "WMT", "JNJ", "PG", "KO",
        # EMERGING/SPECULATIVE
        "PLTR", "COIN", "SNOW", "AMD", "INTC",
        # ENERGY/MATERIALS
        "XOM", "CVX", "CAT", "BA", "GE"
    ]
    
    print("📊 STOCK DATA AVAILABILITY CHECK")
    print("=" * 50)
    
    results = []
    
    for symbol in stocks:
        try:
            # Try to get maximum available data
            ticker = yf.Ticker(symbol)
            
            # Get 5 years of data to check availability
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5*365)  # 5 years
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                first_date = data.index[0]
                last_date = data.index[-1]
                total_days = (last_date - first_date).days
                years_available = total_days / 365.25
                
                results.append({
                    'symbol': symbol,
                    'first_date': first_date,
                    'last_date': last_date,
                    'total_days': total_days,
                    'years_available': years_available,
                    'data_points': len(data)
                })
                
                print(f"✅ {symbol:6}: {years_available:.1f} years ({len(data):,} days) - {first_date.date()} to {last_date.date()}")
            else:
                print(f"❌ {symbol:6}: No data available")
                
        except Exception as e:
            print(f"❌ {symbol:6}: Error - {str(e)}")
    
    # Summary statistics
    if results:
        avg_years = sum(r['years_available'] for r in results) / len(results)
        min_years = min(r['years_available'] for r in results)
        max_years = max(r['years_available'] for r in results)
        
        print(f"\n📈 SUMMARY:")
        print(f"   Average data available: {avg_years:.1f} years")
        print(f"   Minimum data available: {min_years:.1f} years")
        print(f"   Maximum data available: {max_years:.1f} years")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if min_years >= 3:
            print(f"   ✅ Can safely use 2-3 years for backtesting")
        elif min_years >= 1:
            print(f"   ⚠️  Limited to 1 year for some stocks")
        else:
            print(f"   ❌ Some stocks have very limited data")
        
        print(f"\n⚙️  CURRENT OPTIMIZER SETTINGS:")
        print(f"   📅 Using: ~6 months (170 days)")
        print(f"   📊 Available: Up to {max_years:.1f} years")
        print(f"   🎯 Recommendation: Use 1-2 years for better results")
        
    return results

if __name__ == "__main__":
    results = check_stock_data_availability()
