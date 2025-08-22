"""
🎯 ALPACA VERSION OF PROVEN TRADING LOGIC
Uses the exact same trading simulation as comprehensive_backtester.py but with Alpaca data
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

class AlpacaProvenBacktester:
    """Uses proven trading logic from comprehensive_backtester.py with Alpaca data"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        
        # Load Alpaca credentials
        config_path = Path(__file__).parent / "alpaca_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.api = tradeapi.REST(
            config['alpaca']['api_key'],
            config['alpaca']['secret_key'],
            config['alpaca']['base_url'],
            api_version='v2'
        )
        
        # Configuration (same as successful backtester)
        self.config = {
            'min_model_accuracy': 0.52,
            'max_position_size': 0.12,  # 12% per position
            'max_positions': 8,
            'min_signal_strength': 0.35,
            'min_conviction': 0.5
        }
    
    def fetch_alpaca_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Alpaca - same format as yfinance"""
        try:
            bars = self.api.get_bars(symbol, timeframe="1Day", start=start_date, end=end_date, limit=5000)
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
            
            return data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"   ❌ Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features (same as working backtester)"""
        df = data.copy()
        
        # Technical indicators
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['rsi'] = self.calculate_rsi(df['Close'])
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['volatility'] = df['Close'].rolling(10).std()
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Target
        df['target'] = df['Close'].shift(-1) / df['Close'] - 1
        df['target_binary'] = (df['target'] > 0).astype(int)
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train_model(self, data: pd.DataFrame) -> Optional[Dict]:
        """Train model (same as working backtester)"""
        try:
            df = self.create_features(data)
            feature_cols = ['sma_5', 'sma_20', 'rsi', 'momentum_5', 'volatility', 'volume_ratio']
            df_clean = df[feature_cols + ['target_binary']].dropna()
            
            if len(df_clean) < 100:
                return None
            
            # Split for validation
            X = df_clean[feature_cols]
            y = df_clean['target_binary']
            split_idx = len(df_clean) - 30
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train model
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Validate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy < self.config['min_model_accuracy']:
                return None
            
            return {
                'model': model,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'accuracy': accuracy
            }
            
        except:
            return None
    
    def generate_signal(self, model_info: Dict, data: pd.DataFrame) -> Dict:
        """Generate signal using model"""
        try:
            df = self.create_features(data)
            latest_features = df[model_info['feature_cols']].iloc[-1:].values
            
            if np.isnan(latest_features).any():
                return {'signal': 'HOLD', 'strength': 0.0, 'conviction': 0.0}
            
            latest_scaled = model_info['scaler'].transform(latest_features)
            prediction = model_info['model'].predict(latest_scaled)[0]
            probabilities = model_info['model'].predict_proba(latest_scaled)[0]
            
            conviction = max(probabilities)
            strength = conviction if prediction == 1 else -conviction
            
            if abs(strength) < self.config['min_signal_strength']:
                signal = 'HOLD'
            elif strength > 0:
                signal = 'BUY'
            else:
                signal = 'SELL'
            
            return {
                'signal': signal,
                'strength': abs(strength),
                'conviction': conviction,
                'price': data['Close'].iloc[-1]
            }
            
        except:
            return {'signal': 'HOLD', 'strength': 0.0, 'conviction': 0.0}
    
    def calculate_portfolio_value(self, portfolio: Dict, universe_data: Dict, date) -> float:
        """Calculate portfolio value - EXACT SAME as comprehensive_backtester.py"""
        total_value = portfolio['cash']
        
        for symbol, shares in portfolio['positions'].items():
            if shares > 0 and symbol in universe_data:
                try:
                    # Get price for this date (handle both string and date objects)
                    if isinstance(date, str):
                        date_key = pd.to_datetime(date).date()
                    else:
                        date_key = date
                    
                    symbol_data = universe_data[symbol].loc[:date_key]
                    if len(symbol_data) > 0:
                        current_price = float(symbol_data['Close'].iloc[-1])
                        total_value += shares * current_price
                except:
                    continue
        
        return total_value
    
    def execute_trades(self, signals: Dict, portfolio: Dict, date: str) -> List[Dict]:
        """Execute trades - EXACT SAME logic as comprehensive_backtester.py"""
        trades = []
        
        for symbol, signal_data in signals.items():
            if (signal_data['signal'] == 'BUY' and 
                signal_data['strength'] > self.config['min_signal_strength'] and
                signal_data['conviction'] > self.config['min_conviction']):
                
                # Calculate position size
                position_value = portfolio['cash'] * self.config['max_position_size']
                shares = int(position_value / signal_data['price'])
                
                if shares > 0 and portfolio['cash'] > position_value:
                    portfolio['cash'] -= position_value
                    portfolio['positions'][symbol] = portfolio['positions'].get(symbol, 0) + shares
                    
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': signal_data['price'],
                        'value': position_value,
                        'strength': signal_data['strength'],
                        'conviction': signal_data['conviction']
                    })
            
            elif (signal_data['signal'] == 'SELL' and
                  symbol in portfolio['positions'] and
                  portfolio['positions'][symbol] > 0):
                
                shares = portfolio['positions'][symbol]
                trade_value = shares * signal_data['price']
                
                portfolio['cash'] += trade_value
                portfolio['positions'][symbol] = 0
                
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares,
                    'price': signal_data['price'],
                    'value': trade_value,
                    'strength': signal_data['strength'],
                    'conviction': signal_data['conviction']
                })
        
        return trades
    
    def run_proven_backtest(self, training_start: str, trading_start: str, trading_end: str):
        """Run backtest using PROVEN trading logic"""
        
        print("🚀 ALPACA PROVEN BACKTEST")
        print(f"📊 Training: {training_start} to {trading_start}")
        print(f"📈 Trading: {trading_start} to {trading_end}")
        print("-" * 50)
        
        # Test symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'PYPL', 'COIN']
        
        # Phase 1: Get data and train models
        print("\n🤖 Phase 1: Model Training")
        universe_data = {}
        trained_models = {}
        
        for symbol in symbols:
            # Get training data
            training_data = self.fetch_alpaca_data(symbol, training_start, trading_start)
            if len(training_data) < 200:
                print(f"   ❌ {symbol}: Insufficient data")
                continue
            
            # Train model
            model_info = self.train_model(training_data)
            if model_info is None:
                print(f"   ❌ {symbol}: Model failed")
                continue
            
            # Get full data for trading simulation
            full_data = self.fetch_alpaca_data(symbol, training_start, trading_end)
            if len(full_data) < 200:
                continue
            
            universe_data[symbol] = full_data
            trained_models[symbol] = model_info
            print(f"   ✅ {symbol}: {model_info['accuracy']:.1%} accuracy")
        
        print(f"\n✅ {len(trained_models)} models ready")
        
        # Phase 2: Trading simulation - EXACT SAME as comprehensive_backtester.py
        print(f"\n💼 Phase 2: Trading Simulation")
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {}
        }
        
        all_trades = []
        daily_values = []
        
        # Get trading dates
        trading_dates = pd.bdate_range(start=trading_start, end=trading_end)
        
        for i, date in enumerate(trading_dates):
            date_str = date.strftime('%Y-%m-%d')
            
            if i % 10 == 0:
                print(f"   Day {i+1:2d}/{len(trading_dates)}: {date_str}")
            
            # Generate signals for each symbol
            daily_signals = {}
            for symbol in trained_models.keys():
                if symbol in universe_data:
                    # Get data up to current date (fix date conversion)
                    current_data = universe_data[symbol].loc[:date.date()]  # Convert to date object
                    if len(current_data) >= 50:
                        signal_info = self.generate_signal(trained_models[symbol], current_data)
                        if signal_info['signal'] != 'HOLD':
                            daily_signals[symbol] = signal_info
            
            # Execute trades using PROVEN logic
            trades_today = self.execute_trades(daily_signals, portfolio, date_str)
            all_trades.extend(trades_today)
            
            # Calculate portfolio value using PROVEN method
            portfolio_value = self.calculate_portfolio_value(portfolio, universe_data, date.date())
            
            daily_values.append({
                'date': date_str,
                'portfolio_value': portfolio_value,
                'cash': portfolio['cash'],
                'positions': len([p for p in portfolio['positions'].values() if p > 0])
            })
        
        # Calculate results
        final_value = daily_values[-1]['portfolio_value'] if daily_values else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        print(f"\n🎯 PROVEN ALPACA RESULTS:")
        print(f"   📈 Total Return: {total_return:.2%}")
        print(f"   💰 Final Value: ${final_value:,.2f}")
        print(f"   📊 Total Trades: {len(all_trades)}")
        print(f"   💼 Final Positions: {len([p for p in portfolio['positions'].values() if p > 0])}")
        
        if all_trades:
            print(f"\n📋 Recent Trades:")
            for trade in all_trades[-5:]:
                print(f"   {trade['date']}: {trade['action']} {trade['shares']} {trade['symbol']} @ ${trade['price']:.2f}")

def main():
    backtester = AlpacaProvenBacktester()
    
    # Use your optimal dates
    backtester.run_proven_backtest(
        training_start="2024-01-01",
        trading_start="2025-06-02", 
        trading_end="2025-08-20"
    )

if __name__ == "__main__":
    main()
