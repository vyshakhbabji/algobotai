#!/usr/bin/env python3
"""
Data Availability Checker
Check how many days of data we have for each stock
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Your stock universe
STOCK_UNIVERSE = [
    'GOOG', 'AAPL', 'MSFT', 'NVDA', 'META', 'AMZN', 'AVGO', 'PLTR', 
    'NFLX', 'TSM', 'PANW', 'NOW', 'XLK', 'QQQ', 'COST'
]

def check_data_availability():
    """Check data availability for all stocks"""
    print("📊 DATA AVAILABILITY CHECK")
    print("=" * 70)
    print(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("-" * 70)
    
    results = []
    
    for symbol in STOCK_UNIVERSE:
        try:
            print(f"📈 Checking {symbol}...")
            
            # Get maximum available data
            stock = yf.Ticker(symbol)
            df_max = stock.history(period='max')
            
            # Get different time periods
            df_5y = stock.history(period='5y')
            df_2y = stock.history(period='2y')
            df_1y = stock.history(period='1y')
            df_6m = stock.history(period='6mo')
            df_3m = stock.history(period='3mo')
            df_1m = stock.history(period='1mo')
            
            if not df_max.empty:
                start_date = df_max.index[0].strftime('%Y-%m-%d')
                end_date = df_max.index[-1].strftime('%Y-%m-%d')
                total_days = len(df_max)
                
                result = {
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_days': total_days,
                    'max_days': len(df_max),
                    'days_5y': len(df_5y),
                    'days_2y': len(df_2y),
                    'days_1y': len(df_1y),
                    'days_6m': len(df_6m),
                    'days_3m': len(df_3m),
                    'days_1m': len(df_1m)
                }
                
                results.append(result)
                
                print(f"  📅 Data Range: {start_date} to {end_date}")
                print(f"  📊 Total Days: {total_days}")
                print(f"  📈 Recent availability:")
                print(f"    • 5 years: {len(df_5y)} days")
                print(f"    • 2 years: {len(df_2y)} days") 
                print(f"    • 1 year: {len(df_1y)} days")
                print(f"    • 6 months: {len(df_6m)} days")
                print(f"    • 3 months: {len(df_3m)} days")
                print(f"    • 1 month: {len(df_1m)} days")
                
            else:
                print(f"  ❌ No data available")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    # Summary table
    if results:
        print("=" * 70)
        print("📋 SUMMARY TABLE")
        print("=" * 70)
        print(f"{'Stock':<6} {'Start Date':<12} {'End Date':<12} {'Total':<6} {'1Y':<4} {'6M':<4} {'3M':<4} {'1M':<4}")
        print("-" * 70)
        
        for r in results:
            print(f"{r['symbol']:<6} {r['start_date']:<12} {r['end_date']:<12} "
                  f"{r['total_days']:<6} {r['days_1y']:<4} {r['days_6m']:<4} "
                  f"{r['days_3m']:<4} {r['days_1m']:<4}")
        
        # Overall statistics
        print("\n" + "=" * 70)
        print("📊 OVERALL STATISTICS")
        print("=" * 70)
        
        avg_total = sum(r['total_days'] for r in results) / len(results)
        avg_1y = sum(r['days_1y'] for r in results) / len(results)
        avg_6m = sum(r['days_6m'] for r in results) / len(results)
        avg_3m = sum(r['days_3m'] for r in results) / len(results)
        
        min_total = min(r['total_days'] for r in results)
        max_total = max(r['total_days'] for r in results)
        
        print(f"📈 Average total days: {avg_total:.0f}")
        print(f"📈 Average 1-year data: {avg_1y:.0f} days")
        print(f"📈 Average 6-month data: {avg_6m:.0f} days")
        print(f"📈 Average 3-month data: {avg_3m:.0f} days")
        print(f"📈 Range: {min_total} to {max_total} days")
        
        # Data quality assessment
        print(f"\n🎯 DATA QUALITY ASSESSMENT:")
        stocks_with_5y = sum(1 for r in results if r['days_5y'] > 1000)
        stocks_with_2y = sum(1 for r in results if r['days_2y'] > 400)
        stocks_with_1y = sum(1 for r in results if r['days_1y'] > 200)
        
        print(f"  • Stocks with 5+ years data: {stocks_with_5y}/{len(results)}")
        print(f"  • Stocks with 2+ years data: {stocks_with_2y}/{len(results)}")
        print(f"  • Stocks with 1+ year data: {stocks_with_1y}/{len(results)}")
        
        # Training recommendations
        print(f"\n🤖 AI TRAINING RECOMMENDATIONS:")
        if avg_1y >= 200:
            print("  ✅ Sufficient data for 1-year model training")
        else:
            print("  ⚠️ Limited data for 1-year training")
        
        if avg_6m >= 100:
            print("  ✅ Good data for 6-month backtesting")
        else:
            print("  ⚠️ Limited data for backtesting")
        
        if avg_3m >= 50:
            print("  ✅ Adequate data for 3-month validation")
        else:
            print("  ⚠️ Limited data for validation")
        
        # Find the stock with most/least data
        most_data = max(results, key=lambda x: x['total_days'])
        least_data = min(results, key=lambda x: x['total_days'])
        
        print(f"\n📊 DATA EXTREMES:")
        print(f"  🏆 Most data: {most_data['symbol']} ({most_data['total_days']} days)")
        print(f"  📉 Least data: {least_data['symbol']} ({least_data['total_days']} days)")

if __name__ == "__main__":
    try:
        check_data_availability()
        print(f"\n✅ Data availability check complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Check interrupted")
    except Exception as e:
        print(f"\n❌ Check failed: {e}")
        import traceback
        traceback.print_exc()
