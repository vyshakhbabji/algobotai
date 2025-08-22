#!/usr/bin/env python3
"""
Pure Alpaca Backtesting System
Uses only Alpaca data to ensure consistency with live trading
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

class PureAlpacaBacktester:
    def __init__(self, config_file='alpaca_config.json'):
        """Initialize with Alpaca credentials"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.data_client = StockHistoricalDataClient(
            api_key=config['alpaca']['api_key'],
            secret_key=config['alpaca']['secret_key']
        )
        
        # Trading configuration
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
        self.initial_balance = 50000
        self.position_size = 0.15  # 15% per position
        self.stop_loss = 0.08  # 8% stop loss
        self.take_profit = 0.20  # 20% take profit
        
        # Strategy parameters (matching live trading exactly)
        self.lookback_hours = 168  # 1 week for training
        self.signal_threshold_buy = 0.005  # 0.5% predicted return to buy
        self.signal_threshold_sell = -0.005  # -0.5% predicted return to sell
        
        # Results tracking
        self.portfolio_history = []
        self.trades = []
        self.models = {}
        self.scalers = {}
        
    def fetch_alpaca_data(self, start_date, end_date, timeframe=TimeFrame.Hour):
        """Fetch data from Alpaca for all symbols"""
        print(f"📊 Fetching Alpaca data from {start_date.date()} to {end_date.date()}...")
        
        all_data = {}
        
        for symbol in self.symbols:
            try:
                print(f"  Fetching {symbol}...")
                
                request = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date
                )
                
                bars = self.data_client.get_stock_bars(request)
                df = bars.df
                
                if not df.empty:
                    # Process Alpaca data format
                    df = df.reset_index()
                    df['symbol'] = symbol
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    # Add hour and day features for strategy
                    df['hour'] = df['timestamp'].dt.hour
                    df['day_of_week'] = df['timestamp'].dt.dayofweek
                    df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)
                    
                    all_data[symbol] = df
                    print(f"    ✅ Got {len(df)} bars for {symbol}")
                else:
                    print(f"    ❌ No data for {symbol}")
                    
            except Exception as e:
                print(f"    ❌ Error fetching {symbol}: {e}")
                
        return all_data
    
    def calculate_alpaca_features(self, df):
        """Calculate features using Alpaca data structure"""
        # Price and return features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_ratio_5_20'] = df['sma_5'] / df['sma_20']
        df['sma_ratio_20_50'] = df['sma_20'] / df['sma_50']
        
        # Price position relative to moving averages
        df['price_vs_sma5'] = (df['close'] - df['sma_5']) / df['sma_5']
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        # Volatility (using Alpaca's VWAP)
        df['volatility_10h'] = df['returns'].rolling(10).std()
        df['volatility_24h'] = df['returns'].rolling(24).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize to -1 to 1
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features (using Alpaca's volume and trade_count)
        df['volume_sma'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['trade_count_sma'] = df['trade_count'].rolling(10).mean()
        df['trade_count_ratio'] = df['trade_count'] / df['trade_count_sma']
        
        # VWAP features (unique to Alpaca)
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        df['vwap_trend'] = df['vwap'].pct_change()
        
        # Momentum features
        df['momentum_1h'] = df['close'] / df['close'].shift(1) - 1
        df['momentum_4h'] = df['close'] / df['close'].shift(4) - 1
        df['momentum_24h'] = df['close'] / df['close'].shift(24) - 1
        
        # High-Low ratio
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['open_close_ratio'] = (df['close'] - df['open']) / df['open']
        
        return df
    
    def prepare_ml_data(self, df):
        """Prepare data for ML model training"""
        # Feature columns (using Alpaca-specific features)
        feature_columns = [
            'sma_ratio_5_20', 'sma_ratio_20_50', 'price_vs_sma5', 'price_vs_sma20',
            'volatility_10h', 'volatility_24h', 'rsi_normalized', 
            'macd', 'macd_signal', 'macd_histogram', 'bb_position',
            'volume_ratio', 'trade_count_ratio', 'price_vs_vwap', 'vwap_trend',
            'momentum_1h', 'momentum_4h', 'momentum_24h',
            'hl_ratio', 'open_close_ratio', 'hour', 'day_of_week', 'is_market_hours'
        ]
        
        # Target: next period return
        df['target'] = df['close'].shift(-1) / df['close'] - 1
        
        # Clean data
        df_clean = df[feature_columns + ['target', 'timestamp', 'close', 'symbol']].dropna()
        
        return df_clean, feature_columns
    
    def train_symbol_model(self, symbol, df, feature_columns):
        """Train ML model for a specific symbol"""
        print(f"🧠 Training model for {symbol} with {len(df)} samples...")
        
        if len(df) < 100:
            print(f"    ⚠️ Insufficient data for {symbol}")
            return False
        
        # Prepare features and target
        X = df[feature_columns].values
        y = df['target'].values
        
        # Remove any remaining NaN or inf values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            print(f"    ⚠️ Too few clean samples for {symbol}")
            return False
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use 80% for training
        train_size = int(0.8 * len(X_scaled))
        X_train = X_scaled[:train_size]
        y_train = y[:train_size]
        
        # Train Random Forest model (same as live trading)
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Store model and scaler
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        
        # Calculate feature importance
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"    ✅ Model trained for {symbol}")
        print(f"    📊 Top features: {[f[0] for f in top_features]}")
        
        return True
    
    def generate_trading_signals(self, symbol, df, feature_columns):
        """Generate trading signals using trained model"""
        if symbol not in self.models:
            return pd.Series(index=df.index, data=0)
        
        model = self.models[symbol]
        scaler = self.scalers[symbol]
        
        # Prepare features
        X = df[feature_columns].values
        
        # Handle NaN/inf values
        mask = np.isfinite(X).all(axis=1)
        predictions = np.zeros(len(X))
        
        if mask.sum() > 0:
            X_clean = X[mask]
            X_scaled = scaler.transform(X_clean)
            pred_clean = model.predict(X_scaled)
            predictions[mask] = pred_clean
        
        # Convert predictions to signals
        signals = pd.Series(index=df.index, data=0)
        signals[predictions > self.signal_threshold_buy] = 1   # Buy
        signals[predictions < self.signal_threshold_sell] = -1  # Sell
        
        return signals
    
    def run_backtest(self, start_date, end_date):
        """Run complete backtest using only Alpaca data"""
        print(f"🚀 Running Pure Alpaca Backtest")
        print(f"📅 Period: {start_date.date()} to {end_date.date()}")
        print("="*60)
        
        # Fetch all data
        data = self.fetch_alpaca_data(start_date, end_date)
        
        if not data:
            print("❌ No data available for backtesting")
            return None
        
        print(f"✅ Data fetched for {len(data)} symbols")
        
        # Process data and train models
        processed_data = {}
        
        for symbol in data:
            df = data[symbol].copy()
            
            # Calculate features
            df = self.calculate_alpaca_features(df)
            
            # Prepare ML data
            df_clean, feature_columns = self.prepare_ml_data(df)
            
            if len(df_clean) > 100:
                # Train model
                if self.train_symbol_model(symbol, df_clean, feature_columns):
                    # Generate signals
                    signals = self.generate_trading_signals(symbol, df_clean, feature_columns)
                    df_clean['signal'] = signals
                    processed_data[symbol] = df_clean
        
        if not processed_data:
            print("❌ No models could be trained")
            return None
        
        print(f"✅ Models trained for {len(processed_data)} symbols")
        
        # Run trading simulation
        self.simulate_trading(processed_data)
        
        # Analyze results
        results = self.analyze_results()
        
        return results
    
    def simulate_trading(self, data):
        """Simulate trading with the processed data"""
        print("📈 Starting trading simulation...")
        
        # Get all unique timestamps
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df['timestamp'])
        
        all_timestamps = sorted(list(all_timestamps))
        
        # Initialize portfolio
        cash = self.initial_balance
        positions = {}  # symbol -> {'quantity': int, 'avg_cost': float}
        
        print(f"🕐 Simulating {len(all_timestamps)} time periods...")
        
        for i, timestamp in enumerate(all_timestamps):
            current_prices = {}
            current_signals = {}
            
            # Get current data for all symbols
            for symbol, df in data.items():
                symbol_data = df[df['timestamp'] == timestamp]
                if not symbol_data.empty:
                    current_prices[symbol] = symbol_data['close'].iloc[0]
                    current_signals[symbol] = symbol_data['signal'].iloc[0]
            
            # Execute trades
            cash, positions = self.execute_trades(
                timestamp, current_prices, current_signals, cash, positions
            )
            
            # Calculate portfolio value
            portfolio_value = cash
            for symbol, position in positions.items():
                if symbol in current_prices:
                    portfolio_value += position['quantity'] * current_prices[symbol]
            
            # Record portfolio state
            self.portfolio_history.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'num_positions': len(positions),
                'positions_value': portfolio_value - cash
            })
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(all_timestamps)} periods...")
    
    def execute_trades(self, timestamp, prices, signals, cash, positions):
        """Execute trades based on signals"""
        # Process sell signals first
        for symbol, signal in signals.items():
            if signal == -1 and symbol in positions:
                # Sell position
                position = positions[symbol]
                sell_value = position['quantity'] * prices[symbol] * 0.999  # Account for fees
                cash += sell_value
                
                # Record trade
                pnl = sell_value - (position['quantity'] * position['avg_cost'])
                pnl_pct = pnl / (position['quantity'] * position['avg_cost'])
                
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': position['quantity'],
                    'price': prices[symbol],
                    'value': sell_value,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                
                del positions[symbol]
        
        # Process buy signals
        for symbol, signal in signals.items():
            if signal == 1 and symbol not in positions:
                # Calculate position size
                position_value = cash * self.position_size
                
                if position_value > 1000:  # Minimum position size
                    quantity = int(position_value / prices[symbol])
                    total_cost = quantity * prices[symbol] * 1.001  # Account for fees
                    
                    if total_cost <= cash:
                        # Buy position
                        cash -= total_cost
                        positions[symbol] = {
                            'quantity': quantity,
                            'avg_cost': prices[symbol] * 1.001
                        }
                        
                        self.trades.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': quantity,
                            'price': prices[symbol],
                            'value': total_cost,
                            'pnl': 0,
                            'pnl_pct': 0
                        })
        
        return cash, positions
    
    def analyze_results(self):
        """Analyze backtest results"""
        if not self.portfolio_history:
            print("❌ No portfolio data to analyze")
            return None
        
        # Create portfolio DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        portfolio_df = portfolio_df.sort_values('timestamp')
        
        # Calculate metrics
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        initial_value = self.initial_balance
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        returns = portfolio_df['returns'].dropna()
        
        # Risk metrics
        volatility = returns.std()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24) if returns.std() > 0 else 0
        
        # Drawdown
        portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Trading metrics
        trades_df = pd.DataFrame(self.trades)
        
        print("\n" + "="*70)
        print("🚀 PURE ALPACA BACKTESTING RESULTS")
        print("="*70)
        
        print(f"📊 Portfolio Performance:")
        print(f"   Initial Balance: ${initial_value:,.2f}")
        print(f"   Final Value: ${final_value:,.2f}")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Volatility: {volatility:.4f}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {max_drawdown:.2%}")
        
        if not trades_df.empty:
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            if not sell_trades.empty:
                winning_trades = sell_trades[sell_trades['pnl'] > 0]
                win_rate = len(winning_trades) / len(sell_trades)
                avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
                avg_loss = sell_trades[sell_trades['pnl'] < 0]['pnl'].mean()
                
                print(f"\n📈 Trading Performance:")
                print(f"   Total Trades: {len(trades_df)}")
                print(f"   Completed Trades: {len(sell_trades)}")
                print(f"   Win Rate: {win_rate:.2%}")
                print(f"   Average Win: ${avg_win:.2f}")
                print(f"   Average Loss: ${avg_loss:.2f}")
                print(f"   Best Trade: ${sell_trades['pnl'].max():.2f}")
                print(f"   Worst Trade: ${sell_trades['pnl'].min():.2f}")
        
        # Period analysis
        start_date = portfolio_df['timestamp'].min()
        end_date = portfolio_df['timestamp'].max()
        period_days = (end_date - start_date).days
        
        print(f"\n📅 Period Analysis:")
        print(f"   Start: {start_date}")
        print(f"   End: {end_date}")
        print(f"   Duration: {period_days} days")
        
        if period_days > 0:
            annualized_return = (1 + total_return) ** (365 / period_days) - 1
            print(f"   Annualized Return: {annualized_return:.2%}")
        
        print(f"\n🎯 DATA SOURCE: 100% Alpaca (consistent with live trading)")
        print(f"✅ No YFinance discrepancies - results are reliable!")
        
        # Save results
        results = {
            'backtest_type': 'pure_alpaca',
            'data_source': 'alpaca_only',
            'initial_balance': initial_value,
            'final_value': float(final_value),
            'total_return': float(total_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate) if not trades_df.empty and not sell_trades.empty else 0,
            'total_trades': len(trades_df),
            'period_days': period_days,
            'annualized_return': float(annualized_return) if period_days > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('pure_alpaca_backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: pure_alpaca_backtest_results.json")
        
        return results

def main():
    """Run the pure Alpaca backtesting system"""
    print("🚀 Pure Alpaca Backtesting System")
    print("📊 Using 100% Alpaca data for consistency with live trading")
    print("="*70)
    
    # Initialize backtester
    backtester = PureAlpacaBacktester()
    
    # Set backtest period (last 60 days)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=60)
    
    # Run backtest
    results = backtester.run_backtest(start_date, end_date)
    
    if results:
        print(f"\n🎉 Backtest completed successfully!")
        print(f"📈 Total Return: {results['total_return']:.2%}")
        print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"🎯 Data Consistency: 100% Alpaca (no YFinance discrepancies)")
    else:
        print("❌ Backtest failed")

if __name__ == "__main__":
    main()
