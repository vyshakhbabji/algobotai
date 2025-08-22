#!/usr/bin/env python3
"""
Alpaca vs YFinance Data Structure Comparison
Test and compare data formats between the two sources
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def load_alpaca_config():
    """Load Alpaca configuration"""
    with open('alpaca_config.json', 'r') as f:
        config = json.load(f)
    return config

def fetch_yfinance_data(symbol, period="1mo"):
    """Fetch data from yfinance"""
    print(f"📊 Fetching {symbol} from YFinance...")
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval="1h")
        print(f"  ✅ YFinance: Got {len(data)} rows")
        return data
    except Exception as e:
        print(f"  ❌ YFinance error: {e}")
        return None

def fetch_alpaca_data(symbol, days_back=30):
    """Fetch data from Alpaca"""
    print(f"📊 Fetching {symbol} from Alpaca...")
    try:
        config = load_alpaca_config()
        client = StockHistoricalDataClient(
            api_key=config['alpaca']['api_key'],
            secret_key=config['alpaca']['secret_key']
        )
        
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days_back)
        
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Hour,
            start=start_date,
            end=end_date
        )
        
        bars = client.get_stock_bars(request)
        df = bars.df
        
        if not df.empty:
            df = df.reset_index()
            print(f"  ✅ Alpaca: Got {len(df)} rows")
            return df
        else:
            print(f"  ❌ Alpaca: No data returned")
            return None
            
    except Exception as e:
        print(f"  ❌ Alpaca error: {e}")
        return None

def compare_data_structures(symbol="AAPL"):
    """Compare data structures between YFinance and Alpaca"""
    print(f"\n🔍 COMPARING DATA STRUCTURES FOR {symbol}")
    print("="*60)
    
    # Fetch from both sources
    yf_data = fetch_yfinance_data(symbol)
    alpaca_data = fetch_alpaca_data(symbol)
    
    if yf_data is None or alpaca_data is None:
        print("❌ Cannot compare - missing data from one source")
        return None, None
    
    print(f"\n📊 DATA STRUCTURE COMPARISON:")
    print(f"YFinance columns: {list(yf_data.columns)}")
    print(f"Alpaca columns: {list(alpaca_data.columns)}")
    
    print(f"\n📈 SAMPLE DATA:")
    print(f"YFinance sample:")
    print(yf_data.head(2))
    print(f"\nAlpaca sample:")
    print(alpaca_data.head(2))
    
    print(f"\n📊 DATA TYPES:")
    print(f"YFinance dtypes:")
    print(yf_data.dtypes)
    print(f"\nAlpaca dtypes:")
    print(alpaca_data.dtypes)
    
    # Check for overlapping time periods
    if 'timestamp' in alpaca_data.columns:
        alpaca_data['timestamp'] = pd.to_datetime(alpaca_data['timestamp'])
        alpaca_start = alpaca_data['timestamp'].min()
        alpaca_end = alpaca_data['timestamp'].max()
    else:
        alpaca_start = alpaca_data.index.min()
        alpaca_end = alpaca_data.index.max()
    
    yf_start = yf_data.index.min()
    yf_end = yf_data.index.max()
    
    print(f"\n⏰ TIME RANGES:")
    print(f"YFinance: {yf_start} to {yf_end}")
    print(f"Alpaca: {alpaca_start} to {alpaca_end}")
    
    # Compare price values for overlapping times
    print(f"\n💰 PRICE COMPARISON:")
    if not yf_data.empty and not alpaca_data.empty:
        yf_close = yf_data['Close'].iloc[-1] if 'Close' in yf_data.columns else None
        alpaca_close = alpaca_data['close'].iloc[-1] if 'close' in alpaca_data.columns else None
        
        if yf_close and alpaca_close:
            price_diff = abs(yf_close - alpaca_close)
            price_diff_pct = (price_diff / yf_close) * 100
            print(f"YFinance latest close: ${yf_close:.2f}")
            print(f"Alpaca latest close: ${alpaca_close:.2f}")
            print(f"Difference: ${price_diff:.2f} ({price_diff_pct:.2f}%)")
    
    return yf_data, alpaca_data

def create_unified_format(yf_data, alpaca_data):
    """Create a unified format for both data sources"""
    print(f"\n🔄 CREATING UNIFIED FORMAT:")
    
    # Standardize YFinance data
    if yf_data is not None:
        yf_unified = yf_data.copy()
        yf_unified.columns = [col.lower() for col in yf_unified.columns]
        yf_unified = yf_unified.reset_index()
        yf_unified.rename(columns={'datetime': 'timestamp'}, inplace=True)
        if 'date' in yf_unified.columns:
            yf_unified.rename(columns={'date': 'timestamp'}, inplace=True)
        yf_unified['source'] = 'yfinance'
        print(f"  ✅ YFinance unified: {len(yf_unified)} rows")
    
    # Standardize Alpaca data
    if alpaca_data is not None:
        alpaca_unified = alpaca_data.copy()
        if 'symbol' in alpaca_unified.columns:
            alpaca_unified = alpaca_unified.drop('symbol', axis=1)
        alpaca_unified['source'] = 'alpaca'
        print(f"  ✅ Alpaca unified: {len(alpaca_unified)} rows")
    
    # Show unified formats
    if yf_data is not None:
        print(f"\nYFinance unified columns: {list(yf_unified.columns)}")
    if alpaca_data is not None:
        print(f"Alpaca unified columns: {list(alpaca_unified.columns)}")
    
    return yf_unified if yf_data is not None else None, alpaca_unified if alpaca_data is not None else None

def test_trading_logic_compatibility():
    """Test if our trading logic works with both data sources"""
    print(f"\n🧪 TESTING TRADING LOGIC COMPATIBILITY:")
    print("="*60)
    
    # Test with multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    results = {}
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}...")
        yf_data, alpaca_data = compare_data_structures(symbol)
        
        if yf_data is not None and alpaca_data is not None:
            yf_unified, alpaca_unified = create_unified_format(yf_data, alpaca_data)
            
            # Test basic calculations
            yf_returns = yf_unified['close'].pct_change().dropna()
            alpaca_returns = alpaca_unified['close'].pct_change().dropna()
            
            results[symbol] = {
                'yf_data_points': len(yf_unified),
                'alpaca_data_points': len(alpaca_unified),
                'yf_avg_return': yf_returns.mean() if len(yf_returns) > 0 else 0,
                'alpaca_avg_return': alpaca_returns.mean() if len(alpaca_returns) > 0 else 0,
                'yf_volatility': yf_returns.std() if len(yf_returns) > 0 else 0,
                'alpaca_volatility': alpaca_returns.std() if len(alpaca_returns) > 0 else 0
            }
    
    # Display results
    print(f"\n📈 TRADING METRICS COMPARISON:")
    for symbol, metrics in results.items():
        print(f"\n{symbol}:")
        print(f"  Data points - YF: {metrics['yf_data_points']}, Alpaca: {metrics['alpaca_data_points']}")
        print(f"  Avg Return - YF: {metrics['yf_avg_return']:.4f}, Alpaca: {metrics['alpaca_avg_return']:.4f}")
        print(f"  Volatility - YF: {metrics['yf_volatility']:.4f}, Alpaca: {metrics['alpaca_volatility']:.4f}")
    
    return results

def recommend_data_source():
    """Provide recommendation on which data source to use"""
    print(f"\n🎯 RECOMMENDATION:")
    print("="*60)
    
    print(f"✅ USE ALPACA DATA for backtesting because:")
    print(f"   1. Same data source as live trading (consistency)")
    print(f"   2. Real-time data matches historical data structure")
    print(f"   3. No discrepancies between backtest and live trading")
    print(f"   4. Official broker data (more reliable)")
    print(f"   5. Same API, same data format, same timezone handling")
    
    print(f"\n⚠️  YFinance issues:")
    print(f"   1. Different data provider (potential price differences)")
    print(f"   2. Different timezone handling")
    print(f"   3. Potential column name/structure differences")
    print(f"   4. Could lead to backtest vs live trading discrepancies")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Create Alpaca-only backtesting system")
    print(f"   2. Use same data format for backtest and live trading")
    print(f"   3. Validate strategy with consistent data source")

def main():
    """Main comparison function"""
    print("🔍 ALPACA vs YFINANCE DATA COMPARISON")
    print("="*60)
    
    # Compare data structures
    yf_data, alpaca_data = compare_data_structures("AAPL")
    
    if yf_data is not None and alpaca_data is not None:
        # Create unified formats
        yf_unified, alpaca_unified = create_unified_format(yf_data, alpaca_data)
        
        # Test trading logic compatibility
        results = test_trading_logic_compatibility()
    
    # Provide recommendation
    recommend_data_source()
    
    print(f"\n✅ Analysis complete!")
    print(f"💡 Recommendation: Use Alpaca data for all backtesting")

if __name__ == "__main__":
    main()
