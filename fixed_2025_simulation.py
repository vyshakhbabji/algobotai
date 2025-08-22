"""
🎯 FIXED OPTIMAL 2025 SIMULATION
Proper daily trading simulation with correct data alignment and portfolio management
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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score

# Alpaca import
import alpaca_trade_api as tradeapi

# Suppress warnings
warnings.filterwarnings('ignore')

def create_simple_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create simple technical features"""
    df = data.copy()
    
    # Simple moving averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    
    # Momentum
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # Volatility
    df['volatility'] = df['close'].rolling(10).std()
    
    # Volume ratio
    df['volume_ma'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Target: next day return
    df['target'] = df['close'].shift(-1) / df['close'] - 1
    df['target_binary'] = (df['target'] > 0).astype(int)
    
    return df

def train_simple_model(data: pd.DataFrame) -> Optional[Dict]:
    """Train a simple model like our successful backtester"""
    try:
        # Create features
        df = create_simple_features(data)
        
        # Feature columns
        feature_cols = ['sma_5', 'sma_20', 'momentum_5', 'momentum_10', 'volatility', 'volume_ratio']
        
        # Drop NaN values
        df_clean = df[feature_cols + ['target_binary']].dropna()
        
        if len(df_clean) < 100:  # Need minimum data
            return None
        
        # Use last 30 days for validation
        X = df_clean[feature_cols]
        y = df_clean['target_binary']
        
        split_idx = len(df_clean) - 30
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Validate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        if accuracy < 0.52:  # 52% threshold
            return None
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'accuracy': accuracy
        }
        
    except Exception as e:
        return None

def generate_simple_signal(model_info: Dict, data: pd.DataFrame) -> str:
    """Generate simple BUY/HOLD signal"""
    try:
        # Create features for latest data point
        df = create_simple_features(data)
        latest_features = df[model_info['feature_cols']].iloc[-1:].values
        
        if np.isnan(latest_features).any():
            return 'HOLD'
        
        # Scale and predict
        latest_scaled = model_info['scaler'].transform(latest_features)
        prediction = model_info['model'].predict(latest_scaled)[0]
        confidence = model_info['model'].predict_proba(latest_scaled)[0].max()
        
        # Simple threshold
        if prediction == 1 and confidence > 0.6:
            return 'BUY'
        else:
            return 'HOLD'
            
    except Exception as e:
        return 'HOLD'

def run_fixed_simulation():
    """Run a fixed simulation with proper data handling"""
    
    print("🚀 FIXED OPTIMAL 2025 SIMULATION")
    print("=" * 50)
    
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
    
    # Optimal dates
    training_start = "2024-01-01"
    trading_start = "2025-06-02"
    trading_end = "2025-08-20"
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
    
    print(f"📊 Training: {training_start} to {trading_start}")
    print(f"📈 Trading: {trading_start} to {trading_end}")
    print(f"🎯 Testing with {len(symbols)} symbols")
    
    # Get data and train models
    trained_models = {}
    
    print(f"\n🤖 Training Models:")
    for symbol in symbols:
        try:
            # Get training data
            bars = api.get_bars(symbol, timeframe="1Day", start=training_start, end=trading_start, limit=5000)
            data = bars.df.reset_index()
            data['date'] = data['timestamp'].dt.date
            data = data.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
            
            if len(data) < 300:
                print(f"   ❌ {symbol}: Insufficient data ({len(data)} days)")
                continue
            
            # Train model
            model_info = train_simple_model(data)
            if model_info is None:
                print(f"   ❌ {symbol}: Model failed validation")
                continue
            
            trained_models[symbol] = model_info
            print(f"   ✅ {symbol}: {model_info['accuracy']:.1%} accuracy")
            
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {e}")
    
    print(f"\n✅ {len(trained_models)} models trained successfully")
    
    if not trained_models:
        print("❌ No valid models - stopping simulation")
        return
    
    # Run simple trading simulation
    print(f"\n💼 Running Trading Simulation:")
    
    # Initialize portfolio
    initial_capital = 100000
    cash = initial_capital
    positions = {}
    trades = []
    
    # Get trading dates
    trading_dates = pd.bdate_range(start=trading_start, end=trading_end)
    print(f"   📈 {len(trading_dates)} trading days")
    
    # Get all trading data upfront
    trading_data = {}
    for symbol in trained_models.keys():
        try:
            bars = api.get_bars(symbol, timeframe="1Day", start=training_start, end=trading_end, limit=5000)
            data = bars.df.reset_index()
            data['date'] = data['timestamp'].dt.date
            data = data.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
            trading_data[symbol] = data
        except Exception as e:
            print(f"   ❌ Failed to get trading data for {symbol}: {e}")
    
    # Daily simulation
    for i, date in enumerate(trading_dates):
        date_obj = date.date()
        
        if i % 10 == 0:
            print(f"   Day {i+1:2d}/{len(trading_dates)}: {date_obj}")
        
        # Generate signals for this date
        daily_signals = {}
        for symbol in trained_models.keys():
            if symbol not in trading_data:
                continue
            
            # Get data up to current date
            current_data = trading_data[symbol][trading_data[symbol].index <= date_obj]
            
            if len(current_data) < 50:  # Need minimum data
                continue
            
            # Generate signal
            signal = generate_simple_signal(trained_models[symbol], current_data)
            if signal == 'BUY':
                daily_signals[symbol] = current_data['close'].iloc[-1]
        
        # Execute trades (simple logic)
        max_positions = 3
        position_size = 0.2  # 20% per position
        
        # Buy signals (if we have cash and room)
        if daily_signals and len(positions) < max_positions:
            for symbol, price in list(daily_signals.items())[:max_positions - len(positions)]:
                if cash > 1000:  # Need minimum cash
                    position_value = cash * position_size
                    shares = int(position_value / price)
                    
                    if shares > 0:
                        cost = shares * price
                        cash -= cost
                        positions[symbol] = {'shares': shares, 'entry_price': price}
                        
                        trades.append({
                            'date': str(date_obj),
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'value': cost
                        })
    
    # Calculate final portfolio value
    final_value = cash
    for symbol, position in positions.items():
        if symbol in trading_data:
            # Get final price
            final_data = trading_data[symbol][trading_data[symbol].index <= trading_dates[-1].date()]
            if len(final_data) > 0:
                final_price = final_data['close'].iloc[-1]
                final_value += position['shares'] * final_price
    
    total_return = (final_value - initial_capital) / initial_capital
    
    print(f"\n🎯 FIXED SIMULATION RESULTS:")
    print(f"   📈 Total Return: {total_return:.2%}")
    print(f"   💰 Final Value: ${final_value:,.2f}")
    print(f"   📊 Total Trades: {len(trades)}")
    print(f"   💼 Final Positions: {len(positions)}")
    print(f"   💵 Final Cash: ${cash:,.2f}")
    
    if trades:
        print(f"\n📋 Recent Trades:")
        for trade in trades[-5:]:
            print(f"   {trade['date']}: {trade['action']} {trade['shares']} {trade['symbol']} @ ${trade['price']:.2f}")

if __name__ == "__main__":
    run_fixed_simulation()
