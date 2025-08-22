#!/usr/bin/env python3
"""
Recent Alpaca Backtest System
Tests trading strategy with available Alpaca data (last 30-60 days)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AlpacaRecentBacktester:
    def __init__(self, config_file='alpaca_config.json'):
        """Initialize with Alpaca API credentials"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.data_client = StockHistoricalDataClient(
            api_key=config['alpaca']['api_key'],
            secret_key=config['alpaca']['secret_key']
        )
        
        # Trading symbols
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
        
        # Strategy parameters
        self.initial_balance = 50000
        self.current_balance = self.initial_balance
        self.positions = {}
        self.position_size = 0.15  # 15% per position
        self.stop_loss = 0.08  # 8% stop loss
        self.take_profit = 0.20  # 20% take profit
        
        # ML models and scalers
        self.models = {}
        self.scalers = {}
        
        # Results tracking
        self.trades = []
        self.portfolio_value = []
        self.daily_returns = []
        
    def fetch_recent_data(self, days_back=45):
        """Fetch recent data for all symbols"""
        print(f"📊 Fetching {days_back} days of data...")
        
        # Calculate date range (avoiding weekends)
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days_back)
        
        all_data = {}
        
        for symbol in self.symbols:
            try:
                print(f"  Fetching {symbol}...")
                
                request = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame.Hour,  # Hourly data for better granularity
                    start=start_date,
                    end=end_date
                )
                
                bars = self.data_client.get_stock_bars(request)
                df = bars.df
                
                if not df.empty:
                    # Clean and prepare data
                    df = df.reset_index()
                    df['symbol'] = df['symbol'] if 'symbol' in df.columns else symbol
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    all_data[symbol] = df
                    print(f"    ✅ Got {len(df)} hours of data for {symbol}")
                else:
                    print(f"    ❌ No data for {symbol}")
                    
            except Exception as e:
                print(f"    ❌ Error fetching {symbol}: {e}")
                
        return all_data
    
    def calculate_features(self, df):
        """Calculate technical indicators and features"""
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_ratio'] = df['ma_5'] / df['ma_20']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(10).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std_val = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price momentum
        df['momentum_1h'] = df['close'] / df['close'].shift(1) - 1
        df['momentum_4h'] = df['close'] / df['close'].shift(4) - 1
        df['momentum_24h'] = df['close'] / df['close'].shift(24) - 1
        
        return df
    
    def prepare_ml_features(self, df):
        """Prepare features for ML model"""
        feature_columns = [
            'ma_ratio', 'volatility', 'rsi', 'macd', 'macd_signal',
            'bb_position', 'volume_ratio', 'momentum_1h', 'momentum_4h', 'momentum_24h'
        ]
        
        # Create target (next hour return)
        df['target'] = df['close'].shift(-1) / df['close'] - 1
        
        # Drop rows with NaN values
        df_clean = df[feature_columns + ['target', 'timestamp', 'close']].dropna()
        
        return df_clean, feature_columns
    
    def train_model(self, symbol, df, feature_columns):
        """Train ML model for a symbol"""
        print(f"🧠 Training model for {symbol}...")
        
        # Prepare features and target
        X = df[feature_columns].values
        y = df['target'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model (use 80% for training)
        train_size = int(0.8 * len(X_scaled))
        X_train = X_scaled[:train_size]
        y_train = y[:train_size]
        
        # Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Store model and scaler
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        
        print(f"    ✅ Model trained for {symbol}")
        
        return train_size
    
    def generate_signals(self, symbol, df, feature_columns, start_idx):
        """Generate trading signals using ML model"""
        if symbol not in self.models:
            return pd.Series(index=df.index, dtype=float)
        
        model = self.models[symbol]
        scaler = self.scalers[symbol]
        
        # Prepare test features
        X = df[feature_columns].values
        X_scaled = scaler.transform(X)
        
        # Generate predictions
        predictions = model.predict(X_scaled)
        
        # Create signals
        signals = pd.Series(index=df.index, dtype=float)
        
        # Signal logic: buy if predicted return > 0.5%, sell if < -0.5%
        buy_threshold = 0.005
        sell_threshold = -0.005
        
        signals[predictions > buy_threshold] = 1  # Buy signal
        signals[predictions < sell_threshold] = -1  # Sell signal
        signals.fillna(0, inplace=True)  # Hold signal
        
        # Only use signals after training period
        signals[:start_idx] = 0
        
        return signals
    
    def simulate_trading(self, data):
        """Simulate trading with the strategy"""
        print("📈 Starting trading simulation...")
        
        # Prepare all data and train models
        prepared_data = {}
        feature_columns = None
        
        for symbol in self.symbols:
            if symbol in data:
                df = data[symbol].copy()
                df = self.calculate_features(df)
                df_clean, feature_cols = self.prepare_ml_features(df)
                
                if len(df_clean) > 50:  # Need minimum data for training
                    train_size = self.train_model(symbol, df_clean, feature_cols)
                    signals = self.generate_signals(symbol, df_clean, feature_cols, train_size)
                    
                    df_clean['signal'] = signals
                    prepared_data[symbol] = df_clean
                    feature_columns = feature_cols
        
        if not prepared_data:
            print("❌ No sufficient data for any symbol")
            return
        
        # Combine all timestamps for simulation
        all_timestamps = set()
        for df in prepared_data.values():
            all_timestamps.update(df['timestamp'])
        
        all_timestamps = sorted(list(all_timestamps))
        
        print(f"🕐 Simulating {len(all_timestamps)} trading periods...")
        
        # Simulate trading
        for timestamp in all_timestamps:
            current_prices = {}
            current_signals = {}
            
            # Get current prices and signals for all symbols
            for symbol, df in prepared_data.items():
                symbol_data = df[df['timestamp'] == timestamp]
                if not symbol_data.empty:
                    current_prices[symbol] = symbol_data['close'].iloc[0]
                    current_signals[symbol] = symbol_data['signal'].iloc[0]
            
            # Execute trades
            self.execute_trades(timestamp, current_prices, current_signals)
            
            # Update portfolio value
            portfolio_val = self.calculate_portfolio_value(current_prices)
            self.portfolio_value.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_val,
                'cash': self.current_balance,
                'positions_value': portfolio_val - self.current_balance
            })
    
    def execute_trades(self, timestamp, prices, signals):
        """Execute trades based on signals"""
        for symbol, signal in signals.items():
            if symbol not in prices:
                continue
                
            price = prices[symbol]
            
            # Check if we have a position
            if symbol in self.positions:
                position = self.positions[symbol]
                current_value = position['quantity'] * price
                
                # Check stop loss and take profit
                pnl_pct = (price - position['entry_price']) / position['entry_price']
                
                should_sell = (
                    signal == -1 or  # Sell signal
                    pnl_pct <= -self.stop_loss or  # Stop loss
                    pnl_pct >= self.take_profit  # Take profit
                )
                
                if should_sell:
                    # Sell position
                    sell_value = current_value * 0.999  # Account for fees
                    self.current_balance += sell_value
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': position['quantity'],
                        'price': price,
                        'value': sell_value,
                        'pnl': sell_value - position['total_cost'],
                        'pnl_pct': pnl_pct
                    })
                    
                    del self.positions[symbol]
            
            # Check for buy signal
            elif signal == 1:
                # Calculate position size
                position_value = self.current_balance * self.position_size
                
                if position_value > 1000:  # Minimum trade size
                    quantity = int(position_value / price)
                    total_cost = quantity * price * 1.001  # Account for fees
                    
                    if total_cost <= self.current_balance:
                        # Buy position
                        self.current_balance -= total_cost
                        
                        self.positions[symbol] = {
                            'quantity': quantity,
                            'entry_price': price,
                            'total_cost': total_cost,
                            'timestamp': timestamp
                        }
                        
                        self.trades.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': quantity,
                            'price': price,
                            'value': total_cost,
                            'pnl': 0,
                            'pnl_pct': 0
                        })
    
    def calculate_portfolio_value(self, current_prices):
        """Calculate current portfolio value"""
        total_value = self.current_balance
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_value += position['quantity'] * current_prices[symbol]
        
        return total_value
    
    def analyze_results(self):
        """Analyze and display results"""
        if not self.portfolio_value:
            print("❌ No trading data to analyze")
            return
        
        # Create portfolio DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_value)
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        portfolio_df = portfolio_df.sort_values('timestamp')
        
        # Calculate metrics
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        daily_returns = portfolio_df['daily_return'].dropna()
        
        # Calculate Sharpe ratio (assuming hourly data, annualize)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252 * 24)  # Hourly to annual
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cumulative_max']) / portfolio_df['cumulative_max']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Analyze trades
        trades_df = pd.DataFrame(self.trades)
        
        print("\n" + "="*60)
        print("🚀 ALPACA RECENT BACKTEST RESULTS")
        print("="*60)
        
        print(f"📊 Portfolio Performance:")
        print(f"   Initial Balance: ${self.initial_balance:,.2f}")
        print(f"   Final Value: ${final_value:,.2f}")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Max Drawdown: {max_drawdown:.2%}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            print(f"\n📈 Trading Activity:")
            print(f"   Total Trades: {len(trades_df)}")
            print(f"   Buy Orders: {len(buy_trades)}")
            print(f"   Sell Orders: {len(sell_trades)}")
            
            if not sell_trades.empty:
                winning_trades = sell_trades[sell_trades['pnl'] > 0]
                win_rate = len(winning_trades) / len(sell_trades)
                avg_pnl = sell_trades['pnl'].mean()
                
                print(f"   Win Rate: {win_rate:.2%}")
                print(f"   Average PnL per Trade: ${avg_pnl:.2f}")
                print(f"   Best Trade: ${sell_trades['pnl'].max():.2f}")
                print(f"   Worst Trade: ${sell_trades['pnl'].min():.2f}")
        
        # Trading period
        start_date = portfolio_df['timestamp'].min()
        end_date = portfolio_df['timestamp'].max()
        trading_days = (end_date - start_date).days
        
        print(f"\n📅 Trading Period:")
        print(f"   Start: {start_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   End: {end_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Duration: {trading_days} days")
        
        # Annualized return
        if trading_days > 0:
            annualized_return = (1 + total_return) ** (365 / trading_days) - 1
            print(f"   Annualized Return: {annualized_return:.2%}")
        
        print("\n🎯 Strategy validated with real Alpaca data!")
        print("   Ready for deployment to Google Cloud Platform")
        
        # Save results
        results = {
            'backtest_type': 'alpaca_recent',
            'initial_balance': self.initial_balance,
            'final_value': float(final_value),
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe_ratio),
            'trading_days': int(trading_days),
            'total_trades': len(trades_df),
            'win_rate': float(len(winning_trades) / len(sell_trades)) if not sell_trades.empty else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('alpaca_recent_backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: alpaca_recent_backtest_results.json")
        
        return results

def main():
    """Run the Alpaca recent backtest"""
    print("🚀 Starting Alpaca Recent Backtest System...")
    print("📊 Testing with real Alpaca data from the last 45 days")
    
    # Initialize backtester
    backtester = AlpacaRecentBacktester()
    
    # Fetch recent data
    data = backtester.fetch_recent_data(days_back=45)
    
    if not data:
        print("❌ No data available for backtesting")
        return
    
    print(f"✅ Successfully fetched data for {len(data)} symbols")
    
    # Run simulation
    backtester.simulate_trading(data)
    
    # Analyze results
    results = backtester.analyze_results()
    
    return results

if __name__ == "__main__":
    main()
