"""
🔍 SIGNAL GENERATION TEST
Test signal generation for specific dates
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def test_signal_for_date(symbol: str, test_date: str = "2024-06-01"):
    """Test signal generation for a specific date"""
    print(f"🔍 Testing signal for {symbol} on {test_date}")
    
    try:
        # Fetch data with proper date range
        ticker = yf.Ticker(symbol)
        # Get data from 2024 to include our test period
        data = ticker.history(start="2024-01-01", end="2024-12-31")
        
        if data.empty:
            print(f"❌ No data for {symbol}")
            return
        
        # Ensure timezone consistency
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Get data up to test date
        test_date_dt = pd.to_datetime(test_date)
        recent_data = data[data.index <= test_date_dt]
        
        if len(recent_data) < 50:
            print(f"❌ Insufficient data for {symbol} up to {test_date} (have {len(recent_data)} days)")
            return None, 0
        
        print(f"✅ Have {len(recent_data)} days of data up to {test_date}")
        
        # Calculate technical indicators
        df = recent_data.copy()
        
        # Price and volume features
        df['returns'] = df['Close'].pct_change()
        df['volume_ma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
        
        # Moving averages
        df['ma_5'] = df['Close'].rolling(5).mean()
        df['ma_20'] = df['Close'].rolling(20).mean()
        df['ma_50'] = df['Close'].rolling(50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Get latest values
        latest = df.iloc[-1]
        
        print(f"📊 Technical Analysis for {symbol} on {test_date}:")
        print(f"   Price: ${latest['Close']:.2f}")
        print(f"   RSI: {latest['rsi']:.1f}")
        print(f"   MA5: ${latest['ma_5']:.2f}")
        print(f"   MA20: ${latest['ma_20']:.2f}")
        print(f"   Volume Ratio: {latest['volume_ratio']:.2f}")
        print(f"   BB Position: {latest['bb_position']:.3f}")
        print(f"   MACD Histogram: {latest['macd_histogram']:.4f}")
        
        # Calculate signal conditions
        rsi = latest['rsi']
        price = latest['Close']
        ma_5 = latest['ma_5']
        ma_20 = latest['ma_20']
        volume_ratio = latest['volume_ratio']
        bb_position = latest['bb_position']
        macd_hist = latest['macd_histogram']
        
        price_vs_ma5 = (price - ma_5) / ma_5
        price_vs_ma20 = (price - ma_20) / ma_20
        
        print(f"\n🎯 Signal Conditions:")
        print(f"   Price vs MA5: {price_vs_ma5*100:+.2f}%")
        print(f"   Price vs MA20: {price_vs_ma20*100:+.2f}%")
        
        # Strong buy conditions
        oversold_support = rsi < 35 and price_vs_ma5 > -0.03
        momentum_buy = (price_vs_ma5 > 0.02 and price_vs_ma20 > 0.01 and 
                       volume_ratio > 1.2 and macd_hist > 0)
        
        strong_buy = oversold_support or momentum_buy
        
        print(f"   Oversold + Support: {oversold_support}")
        print(f"   Momentum Buy: {momentum_buy}")
        print(f"   Strong Buy: {strong_buy}")
        
        # Strong sell conditions
        overbought_resistance = rsi > 75 and price_vs_ma5 < 0.02
        momentum_sell = (price_vs_ma5 < -0.02 and price_vs_ma20 < -0.01 and macd_hist < 0)
        
        strong_sell = overbought_resistance or momentum_sell
        
        print(f"   Overbought + Resistance: {overbought_resistance}")
        print(f"   Momentum Sell: {momentum_sell}")
        print(f"   Strong Sell: {strong_sell}")
        
        # Generate signal
        if strong_buy:
            strength = min(0.8, (1.2 + price_vs_ma5 + (80-rsi)/100 + (volume_ratio-1)))
            strength = max(0.4, strength)
            signal = 'BUY'
        elif strong_sell:
            strength = min(0.8, (1.2 - price_vs_ma5 + (rsi-20)/100))
            strength = max(0.4, strength)
            signal = 'SELL'
        else:
            strength = 0.0
            signal = 'HOLD'
        
        print(f"\n🚀 Final Signal:")
        print(f"   Action: {signal}")
        print(f"   Strength: {strength:.3f}")
        print(f"   Threshold: 0.400 (min for trading)")
        
        if strength >= 0.4:
            print(f"   ✅ Signal strong enough for trading!")
        else:
            print(f"   ⚠️ Signal too weak for trading")
            
        return signal, strength
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def test_multiple_dates():
    """Test signals across multiple dates in our test period"""
    print("🔍 Testing signals across multiple dates")
    print("=" * 60)
    
    symbols = ['AAPL', 'NVDA', 'TSLA']
    test_dates = ['2024-05-20', '2024-06-01', '2024-06-15', '2024-07-01', '2024-07-15', '2024-08-01']
    
    total_signals = 0
    trade_signals = 0
    
    for symbol in symbols:
        print(f"\n📊 {symbol} Analysis:")
        for test_date in test_dates:
            signal, strength = test_signal_for_date(symbol, test_date)
            if signal is not None:
                total_signals += 1
                if strength >= 0.4:
                    trade_signals += 1
                print(f"   {test_date}: {signal} ({strength:.3f})")
            else:
                print(f"   {test_date}: ERROR")
    
    print(f"\n📈 Summary:")
    print(f"   Total signals generated: {total_signals}")
    print(f"   Tradeable signals: {trade_signals}")
    print(f"   Trade rate: {trade_signals/total_signals*100:.1f}%" if total_signals > 0 else "   No signals")

if __name__ == "__main__":
    # Test a specific known volatile period
    print("🔍 SIGNAL GENERATION DEBUGGING")
    print("Testing signal generation for specific dates")
    print("=" * 50)
    
    # Test one specific case first
    test_signal_for_date("AAPL", "2024-06-01")
    
    print("\n" + "="*50)
    
    # Test multiple dates
    test_multiple_dates()
