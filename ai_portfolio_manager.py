#!/usr/bin/env python3
"""
AI Portfolio Manager
Complete system to manage $10k investment with AI-driven buy/hold/sell decisions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Your stock universe (removed BRK.B as requested)
STOCK_UNIVERSE = [
    'GOOG', 'AAPL', 'MSFT', 'NVDA', 'META', 'AMZN', 'AVGO', 'PLTR', 
    'NFLX', 'TSM', 'PANW', 'NOW', 'XLK', 'QQQ', 'COST'
]

INITIAL_CAPITAL = 10000  # $10,000 starting capital

class AIPortfolioManager:
    def __init__(self, capital=10000):
        self.initial_capital = capital
        self.current_capital = capital
        self.positions = {}  # {symbol: shares}
        self.cash = capital
        self.portfolio_history = []
        self.trades = []
        
    def calculate_features(self, df):
        """Calculate AI features"""
        try:
            # Technical indicators
            df['sma_5'] = df['Close'].rolling(window=5).mean()
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['ema_12'] = df['Close'].ewm(span=12).mean()
            df['ema_26'] = df['Close'].ewm(span=26).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_middle = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['bb_upper'] = bb_middle + (bb_std * 2)
            df['bb_lower'] = bb_middle - (bb_std * 2)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume and momentum
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
            df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
            df['volatility'] = df['Close'].rolling(window=20).std()
            
            # Advanced features
            df['price_vs_sma20'] = df['Close'] / df['sma_20'] - 1
            df['volume_momentum'] = df['volume_ratio'] * df['momentum_5']
            df['rsi_momentum'] = df['rsi'] * df['momentum_5']
            
            # Target: future 5-day return
            df['future_return'] = df['Close'].shift(-5) / df['Close'] - 1
            
            return df
        except Exception as e:
            print(f"Error calculating features: {e}")
            return df
    
    def train_model(self, symbol, period='1y'):
        """Train AI model for a specific stock"""
        try:
            print(f"Training model for {symbol}...")
            
            # Get data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                return None, None, None
            
            # Calculate features
            df = self.calculate_features(df)
            
            # Feature columns
            feature_cols = ['sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'bb_position', 
                           'volume_ratio', 'momentum_5', 'momentum_20', 'volatility',
                           'price_vs_sma20', 'volume_momentum', 'rsi_momentum']
            
            # Prepare training data
            clean_data = df[feature_cols + ['future_return']].dropna()
            
            if len(clean_data) < 50:
                return None, None, None
            
            X = clean_data[feature_cols]
            y = clean_data['future_return']
            
            # Split: use 80% for training, 20% for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_train_scaled, y_train)
            
            # Validation score
            val_score = model.score(X_val_scaled, y_val)
            
            return model, scaler, val_score
            
        except Exception as e:
            print(f"Error training {symbol}: {e}")
            return None, None, None
    
    def get_ai_signal(self, symbol, model, scaler, current_data):
        """Get AI signal strength for a stock"""
        try:
            feature_cols = ['sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'bb_position', 
                           'volume_ratio', 'momentum_5', 'momentum_20', 'volatility',
                           'price_vs_sma20', 'volume_momentum', 'rsi_momentum']
            
            # Get latest features
            latest_features = current_data[feature_cols].iloc[-1:].values
            
            # Scale and predict
            latest_scaled = scaler.transform(latest_features)
            prediction = model.predict(latest_scaled)[0]
            
            # Convert to strength score (0-100)
            strength = max(0, min(100, (prediction + 0.1) * 500))  # Normalize to 0-100
            
            return strength, prediction
            
        except Exception as e:
            print(f"Error getting signal for {symbol}: {e}")
            return 50, 0  # Neutral
    
    def backtest_portfolio(self, start_date, end_date):
        """Backtest the AI portfolio management system"""
        print(f"🤖 AI Portfolio Backtest: {start_date} to {end_date}")
        print("=" * 60)
        
        # Train models on historical data (1 year before start_date)
        train_start = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365)
        train_period = f"{train_start.strftime('%Y-%m-%d')}/{start_date}"
        
        models = {}
        scalers = {}
        
        print("📚 Training AI models...")
        for symbol in STOCK_UNIVERSE:
            model, scaler, score = self.train_model(symbol, period='2y')  # 2 years for training
            if model is not None:
                models[symbol] = model
                scalers[symbol] = scaler
                print(f"  {symbol}: R² = {score:.3f}")
        
        print(f"\n✅ Trained {len(models)} models successfully")
        
        # Get test period data
        test_start = datetime.strptime(start_date, '%Y-%m-%d')
        test_end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Monthly rebalancing
        current_date = test_start
        monthly_results = []
        
        while current_date <= test_end:
            print(f"\n📅 Rebalancing on {current_date.strftime('%Y-%m-%d')}")
            
            # Get current data for all stocks
            stock_data = {}
            stock_prices = {}
            
            for symbol in STOCK_UNIVERSE:
                if symbol in models:
                    try:
                        stock = yf.Ticker(symbol)
                        # Get data up to current date
                        df = stock.history(start=train_start, end=current_date + timedelta(days=1))
                        
                        if not df.empty:
                            df = self.calculate_features(df)
                            stock_data[symbol] = df
                            stock_prices[symbol] = df['Close'].iloc[-1]
                    except:
                        continue
            
            # Get AI signals for all stocks
            signals = {}
            for symbol in stock_data:
                if symbol in models:
                    strength, prediction = self.get_ai_signal(symbol, models[symbol], scalers[symbol], stock_data[symbol])
                    signals[symbol] = {
                        'strength': strength,
                        'prediction': prediction,
                        'price': stock_prices[symbol]
                    }
            
            # Portfolio allocation based on AI strength
            self.rebalance_portfolio(signals, current_date)
            
            # Calculate portfolio value
            portfolio_value = self.calculate_portfolio_value(stock_prices)
            monthly_results.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions': self.positions.copy()
            })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return monthly_results
    
    def rebalance_portfolio(self, signals, date):
        """Rebalance portfolio based on AI signals"""
        if not signals:
            return
        
        # Sort stocks by AI strength (highest first)
        sorted_signals = sorted(signals.items(), key=lambda x: x[1]['strength'], reverse=True)
        
        # Current portfolio value
        current_value = self.cash
        for symbol, shares in self.positions.items():
            if symbol in signals:
                current_value += shares * signals[symbol]['price']
        
        # Sell all current positions first
        for symbol, shares in list(self.positions.items()):
            if symbol in signals and shares > 0:
                sell_value = shares * signals[symbol]['price']
                self.cash += sell_value
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares,
                    'price': signals[symbol]['price'],
                    'value': sell_value
                })
        
        self.positions = {}
        
        # Buy top performers based on strength
        # Only buy stocks with strength > 60 (strong buy threshold)
        buy_candidates = [(symbol, data) for symbol, data in sorted_signals if data['strength'] > 60]
        
        if buy_candidates:
            # Divide cash equally among top candidates (max 5 stocks for diversification)
            num_positions = min(5, len(buy_candidates))
            cash_per_stock = self.cash / num_positions
            
            for i in range(num_positions):
                symbol, data = buy_candidates[i]
                price = data['price']
                shares = int(cash_per_stock / price)  # Buy full shares only
                
                if shares > 0:
                    cost = shares * price
                    self.positions[symbol] = shares
                    self.cash -= cost
                    
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': price,
                        'value': cost
                    })
                    
                    print(f"  🟢 BUY {shares} shares of {symbol} at ${price:.2f} (Strength: {data['strength']:.1f})")
    
    def calculate_portfolio_value(self, current_prices):
        """Calculate current portfolio value"""
        total_value = self.cash
        
        for symbol, shares in self.positions.items():
            if symbol in current_prices:
                total_value += shares * current_prices[symbol]
        
        return total_value

def run_backtest():
    """Run the complete backtest"""
    # 3-month backtest period (May 2025 to August 2025)
    start_date = "2025-05-01"
    end_date = "2025-08-01"
    
    manager = AIPortfolioManager(capital=10000)
    results = manager.backtest_portfolio(start_date, end_date)
    
    # Calculate performance
    initial_value = results[0]['portfolio_value']
    final_value = results[-1]['portfolio_value']
    total_return = (final_value / initial_value - 1) * 100
    
    print("\n" + "=" * 60)
    print("📊 BACKTEST RESULTS")
    print("=" * 60)
    print(f"💰 Initial Capital: ${initial_value:,.2f}")
    print(f"💰 Final Value: ${final_value:,.2f}")
    print(f"📈 Total Return: {total_return:+.2f}%")
    print(f"💸 Cash Remaining: ${manager.cash:,.2f}")
    
    # Current positions
    if manager.positions:
        print(f"\n🎯 Final Positions:")
        for symbol, shares in manager.positions.items():
            print(f"  {symbol}: {shares} shares")
    
    # Trade summary
    if manager.trades:
        print(f"\n📋 Trade Summary ({len(manager.trades)} trades):")
        buy_trades = [t for t in manager.trades if t['action'] == 'BUY']
        sell_trades = [t for t in manager.trades if t['action'] == 'SELL']
        print(f"  Buys: {len(buy_trades)}")
        print(f"  Sells: {len(sell_trades)}")
    
    return results, manager

if __name__ == "__main__":
    try:
        results, manager = run_backtest()
        print(f"\n✅ Backtest complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Backtest interrupted")
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
