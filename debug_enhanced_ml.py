"""
🔧 DEBUG: Enhanced ML System
Testing individual components to find the issue
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def test_data_fetching():
    """Test if we can fetch data properly"""
    print("🔍 Testing data fetching...")
    
    symbol = "AAPL"
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1y")
        
        if data.empty:
            print(f"❌ No data for {symbol}")
            return False
        
        print(f"✅ Fetched {len(data)} days of data for {symbol}")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"   Last close: ${data['Close'].iloc[-1]:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return False

def test_technical_indicators():
    """Test technical indicator calculations"""
    print("\n🔍 Testing technical indicators...")
    
    try:
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="6m")
        
        # Calculate basic indicators
        data['ma_5'] = data['Close'].rolling(5).mean()
        data['ma_20'] = data['Close'].rolling(20).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume ratio
        data['volume_ma_20'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_ma_20']
        
        latest = data.iloc[-1]
        print(f"✅ Technical indicators calculated successfully")
        print(f"   RSI: {latest['rsi']:.1f}")
        print(f"   Price vs MA5: {((latest['Close'] - latest['ma_5']) / latest['ma_5'] * 100):+.1f}%")
        print(f"   Price vs MA20: {((latest['Close'] - latest['ma_20']) / latest['ma_20'] * 100):+.1f}%")
        print(f"   Volume Ratio: {latest['volume_ratio']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error calculating indicators: {e}")
        return False

def test_signal_generation():
    """Test base signal generation"""
    print("\n🔍 Testing signal generation...")
    
    try:
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="6m")
        
        # Calculate all indicators
        data['ma_5'] = data['Close'].rolling(5).mean()
        data['ma_20'] = data['Close'].rolling(20).mean()
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        data['volume_ma_20'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_ma_20']
        
        # Bollinger Bands
        data['bb_middle'] = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Generate signal for latest data
        latest = data.iloc[-1]
        
        rsi = latest['rsi']
        price = latest['Close']
        ma_5 = latest['ma_5']
        ma_20 = latest['ma_20']
        volume_ratio = latest['volume_ratio']
        bb_position = latest['bb_position']
        macd_hist = latest['macd_histogram']
        
        price_vs_ma5 = (price - ma_5) / ma_5
        price_vs_ma20 = (price - ma_20) / ma_20
        
        print(f"📊 Signal Analysis for AAPL:")
        print(f"   RSI: {rsi:.1f}")
        print(f"   Price vs MA5: {price_vs_ma5*100:+.2f}%")
        print(f"   Price vs MA20: {price_vs_ma20*100:+.2f}%")
        print(f"   Volume Ratio: {volume_ratio:.2f}")
        print(f"   MACD Histogram: {macd_hist:.4f}")
        
        # Signal conditions
        strong_buy = (
            (rsi < 35 and price_vs_ma5 > -0.03) or  # Oversold with support
            (price_vs_ma5 > 0.02 and price_vs_ma20 > 0.01 and volume_ratio > 1.2 and macd_hist > 0)
        )
        
        strong_sell = (
            (rsi > 75 and price_vs_ma5 < 0.02) or  # Overbought with resistance
            (price_vs_ma5 < -0.02 and price_vs_ma20 < -0.01 and macd_hist < 0)
        )
        
        if strong_buy:
            strength = min(0.8, (1.2 + price_vs_ma5 + (80-rsi)/100 + (volume_ratio-1)))
            signal = {'signal': 'BUY', 'strength': max(0.4, strength), 'price': price}
        elif strong_sell:
            strength = min(0.8, (1.2 - price_vs_ma5 + (rsi-20)/100))
            signal = {'signal': 'SELL', 'strength': max(0.4, strength), 'price': price}
        else:
            signal = {'signal': 'HOLD', 'strength': 0.0, 'price': price}
        
        print(f"\n🎯 Generated Signal:")
        print(f"   Action: {signal['signal']}")
        print(f"   Strength: {signal['strength']:.3f}")
        print(f"   Price: ${signal['price']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating signal: {e}")
        return False

def test_ml_training():
    """Test ML model training"""
    print("\n🔍 Testing ML model training...")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import RobustScaler
        import lightgbm as lgb
        
        # Create some sample data
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.rand(n_samples) * 0.7 + 0.3  # Signal strength 0.3-1.0
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = lgb.LGBMRegressor(n_estimators=50, max_depth=6, random_state=42, verbose=-1)
        model.fit(X_scaled, y)
        
        # Test prediction
        pred = model.predict(X_scaled[:1])[0]
        
        print(f"✅ ML model training successful")
        print(f"   Sample prediction: {pred:.3f}")
        print(f"   Expected range: 0.3-1.0")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training ML model: {e}")
        return False

def test_date_range():
    """Test specific date range for backtest"""
    print("\n🔍 Testing backtest date range...")
    
    try:
        start_date = "2024-05-20"
        end_date = "2024-08-20"
        
        # Create date range
        test_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [d for d in test_dates if d.weekday() < 5]
        
        print(f"✅ Date range test successful")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Total days: {len(test_dates)}")
        print(f"   Trading days: {len(trading_days)}")
        print(f"   First trading day: {trading_days[0].date()}")
        print(f"   Last trading day: {trading_days[-1].date()}")
        
        # Test data availability for this period
        ticker = yf.Ticker("AAPL")
        data = ticker.history(start="2024-01-01", end="2024-12-31")
        
        if not data.empty:
            # Check if we have data for our test period
            data.index = data.index.tz_localize(None)
            period_data = data[(data.index >= pd.to_datetime(start_date)) & 
                             (data.index <= pd.to_datetime(end_date))]
            
            print(f"   Data points in period: {len(period_data)}")
            if len(period_data) > 0:
                print(f"   First data point: {period_data.index[0].date()}")
                print(f"   Last data point: {period_data.index[-1].date()}")
            else:
                print("   ⚠️ No data available for test period")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing date range: {e}")
        return False

def main():
    print("🔧 ENHANCED ML SYSTEM DEBUGGING")
    print("Testing individual components")
    print("=" * 50)
    
    tests = [
        test_data_fetching,
        test_technical_indicators,
        test_signal_generation,
        test_ml_training,
        test_date_range
    ]
    
    results = []
    for test in tests:
        success = test()
        results.append(success)
    
    print(f"\n📊 Debug Summary:")
    print(f"   Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✅ All components working - issue may be in integration")
    else:
        print("❌ Found issues that need fixing")

if __name__ == "__main__":
    main()
