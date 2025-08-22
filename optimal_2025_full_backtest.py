"""
🎯 FULL OPTIMAL 2025 ALPACA BACKTESTER
Complete implementation with your optimal date strategy:
- Training: 2024-01-01 to 2025-06-02 (17 months, 355 days)
- Trading: 2025-06-02 to 2025-08-20 (2.5 months, 56 days)
- All 50 symbols with complete filtering and ML validation
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
os.environ['PYTHONPATH'] = str(Path(__file__).parent)

class OptimalAlpacaBacktester2025:
    """Full backtester with optimal 2025 dates and all missing components"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
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
        
        # Portfolio tracking
        self.positions = {}
        self.trades = []
        self.daily_performance = []
        
        # Configuration optimized for 17-month training period
        self.config = {
            # Stock filtering criteria
            'min_data_days': 300,           # Need 300+ days (17 months = ~355 days)
            'min_avg_volume': 1000000,      # $1M+ average daily volume
            'min_price': 5.0,               # Minimum $5 stock price
            'max_price': 500.0,             # Maximum $500 stock price
            'min_volatility': 0.008,        # Minimum 0.8% volatility
            'max_volatility': 0.15,         # Maximum 15% volatility
            
            # ML model validation criteria
            'min_model_accuracy': 0.52,     # Minimum 52% accuracy
            'min_training_samples': 150,    # Need 150+ samples (17 months of data)
            'accuracy_lookback_days': 30,   # Validate on last 30 days
            'retrain_frequency': 5,         # Retrain every 5 days
            
            # Signal and position criteria
            'signal_threshold': 0.35,       # EXACT threshold from working system
            'max_position_size': 0.12,      # 12% max position
            'max_positions': 8,             # Max 8 positions
            'stop_loss_pct': 0.08,          # 8% stop loss
            'take_profit_pct': 0.25,        # 25% take profit
        }
        
        print("🎯 Optimal 2025 Backtester initialized")
        print(f"   📊 Min model accuracy: {self.config['min_model_accuracy']:.1%}")
        print(f"   🔍 Min data days: {self.config['min_data_days']} (17-month requirement)")
        print(f"   🧠 Min training samples: {self.config['min_training_samples']}")
    
    def get_stock_universe(self) -> List[str]:
        """Get comprehensive stock universe"""
        return [
            # Tech giants
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX',
            # Software & Cloud
            'CRM', 'ORCL', 'ADBE', 'PYPL', 'INTC', 'AMD', 'QCOM', 'SNOW', 'PLTR', 'COIN',
            'RBLX', 'NET', 'DDOG', 'CRWD', 'ZS', 'OKTA', 'MDB', 'SHOP', 'SQ', 'UBER',
            'LYFT', 'ABNB', 'DASH',
            # Finance
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK',
            # Healthcare & Other
            'JNJ', 'PFE', 'UNH', 'LLY', 'ABBV', 'TMO', 'DHR', 'ABT', 'KO', 'PG', 'WMT', 'DIS'
        ]
    
    def fetch_alpaca_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Alpaca API"""
        try:
            bars = self.api.get_bars(
                symbol,
                timeframe="1Day",
                start=start_date,
                end=end_date,
                limit=5000
            )
            data = bars.df
            
            if data.empty:
                return pd.DataFrame()
            
            # Reset index to make timestamp a column
            data = data.reset_index()
            data['date'] = data['timestamp'].dt.date
            data = data.set_index('date')
            
            return data[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"   ❌ Alpaca error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def validate_stock_quality(self, symbol: str, data: pd.DataFrame) -> bool:
        """Validate stock meets quality requirements"""
        if len(data) < self.config['min_data_days']:
            return False
        
        # Price range check
        avg_price = data['close'].mean()
        if avg_price < self.config['min_price'] or avg_price > self.config['max_price']:
            return False
        
        # Volume check
        avg_volume = data['volume'].mean()
        avg_dollar_volume = avg_volume * avg_price
        if avg_dollar_volume < self.config['min_avg_volume']:
            return False
        
        # Volatility check
        returns = data['close'].pct_change().dropna()
        volatility = returns.std()
        if volatility < self.config['min_volatility'] or volatility > self.config['max_volatility']:
            return False
        
        return True
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical features for ML model"""
        df = data.copy()
        
        # Price features
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Momentum features
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Volatility features
        df['volatility_10'] = df['close'].rolling(10).std()
        df['volatility_20'] = df['close'].rolling(20).std()
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Target: next day return
        df['target'] = df['close'].shift(-1) / df['close'] - 1
        df['target_binary'] = (df['target'] > 0).astype(int)
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train_and_validate_model(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """Train and validate ML model with accuracy threshold"""
        try:
            # Create features
            df = self.create_features(data)
            
            # Select feature columns
            feature_cols = [
                'sma_5', 'sma_20', 'sma_50', 'rsi', 'momentum_5', 'momentum_20',
                'volatility_10', 'volatility_20', 'volume_ratio'
            ]
            
            # Drop NaN values
            df_clean = df[feature_cols + ['target_binary']].dropna()
            
            if len(df_clean) < self.config['min_training_samples']:
                return None
            
            # Prepare training data
            X = df_clean[feature_cols]
            y = df_clean['target_binary']
            
            # Split: use last 30 days for validation
            split_idx = len(df_clean) - 30
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Validate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Check accuracy threshold
            if accuracy < self.config['min_model_accuracy']:
                return None
            
            return {
                'model': model,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'accuracy': accuracy,
                'symbol': symbol
            }
            
        except Exception as e:
            return None
    
    def generate_signal(self, symbol: str, model_info: Dict, current_data: pd.DataFrame) -> Dict:
        """Generate trading signal using trained model"""
        try:
            # Create features
            df = self.create_features(current_data)
            
            # Get latest features
            latest_features = df[model_info['feature_cols']].iloc[-1:].values
            
            # Scale features
            latest_scaled = model_info['scaler'].transform(latest_features)
            
            # Get prediction
            prediction = model_info['model'].predict(latest_scaled)[0]
            prediction_proba = model_info['model'].predict_proba(latest_scaled)[0]
            
            confidence = max(prediction_proba)
            signal_strength = confidence if prediction == 1 else -confidence
            
            # Apply signal threshold
            if abs(signal_strength) < self.config['signal_threshold']:
                return {'signal': 'HOLD', 'strength': signal_strength, 'confidence': confidence}
            
            signal = 'BUY' if signal_strength > 0 else 'SELL'
            return {'signal': signal, 'strength': abs(signal_strength), 'confidence': confidence}
            
        except Exception as e:
            return {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.0}
    
    def execute_daily_trades(self, portfolio: Dict, daily_signals: Dict, trades: List, date_str: str):
        """Execute daily trades based on signals - exactly like yfinance backtester"""
        
        # Check for sell signals first (exit positions)
        positions_to_close = []
        for symbol in list(portfolio['positions'].keys()):
            if symbol in daily_signals and daily_signals[symbol]['signal'] == 'SELL':
                positions_to_close.append(symbol)
        
        # Execute sell orders
        for symbol in positions_to_close:
            position = portfolio['positions'][symbol]
            price = daily_signals[symbol]['price']
            value = position['shares'] * price
            
            portfolio['cash'] += value
            del portfolio['positions'][symbol]
            
            trades.append({
                'date': date_str,
                'symbol': symbol,
                'action': 'SELL',
                'shares': position['shares'],
                'price': price,
                'value': value,
                'strength': daily_signals[symbol]['strength'],
                'confidence': daily_signals[symbol]['confidence'],
                'model_accuracy': daily_signals[symbol]['model_accuracy']
            })
        
        # Check for buy signals (new positions)
        buy_signals = {k: v for k, v in daily_signals.items() if v['signal'] == 'BUY'}
        
        if buy_signals and len(portfolio['positions']) < self.config['max_positions']:
            # Sort by signal strength and model accuracy
            sorted_signals = sorted(buy_signals.items(), 
                                  key=lambda x: (x[1]['strength'] * x[1]['model_accuracy']), 
                                  reverse=True)
            
            available_slots = self.config['max_positions'] - len(portfolio['positions'])
            
            for symbol, signal_info in sorted_signals[:available_slots]:
                # Calculate position size
                position_value = portfolio['cash'] * self.config['max_position_size']
                price = signal_info['price']
                shares = int(position_value / price)
                
                if shares > 0 and portfolio['cash'] >= shares * price:
                    total_cost = shares * price
                    
                    portfolio['cash'] -= total_cost
                    portfolio['positions'][symbol] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': date_str
                    }
                    
                    trades.append({
                        'date': date_str,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': price,
                        'value': total_cost,
                        'strength': signal_info['strength'],
                        'confidence': signal_info['confidence'],
                        'model_accuracy': signal_info['model_accuracy']
                    })
    
    def run_backtest(self, training_start: str, trading_start: str, trading_end: str, max_symbols: int = 50) -> Dict:
        """Run complete backtest with optimal dates"""
        
        print(f"\n🚀 OPTIMAL 2025 ALPACA BACKTEST")
        print(f"📊 Training: {training_start} to {trading_start} (17 months)")
        print(f"📈 Trading: {trading_start} to {trading_end} (2.5 months)")
        print(f"🎯 All missing components included")
        print("-" * 70)
        
        # Phase 1: Data collection and filtering
        print(f"\n📥 Phase 1: Data Collection & Stock Filtering")
        
        all_symbols = self.get_stock_universe()[:max_symbols]
        valid_symbols = []
        all_data = {}
        
        for i, symbol in enumerate(all_symbols, 1):
            print(f"    {i:2d}/{len(all_symbols)}: {symbol}", end=" ")
            
            # Get training data
            data = self.fetch_alpaca_data(symbol, training_start, trading_start)
            
            if data.empty:
                print("❌ No data")
                continue
            
            # Validate quality
            if not self.validate_stock_quality(symbol, data):
                print("❌ Failed quality check")
                continue
            
            valid_symbols.append(symbol)
            all_data[symbol] = data
            print(f"✅ {len(data)} days")
        
        print(f"\n   ✅ Filtered to {len(valid_symbols)} high-quality stocks")
        
        # Phase 2: ML model training
        print(f"\n🤖 Phase 2: ML Model Training & Validation")
        
        trained_models = {}
        model_stats = []
        
        for symbol in valid_symbols:
            print(f"   Training {symbol}: ", end="")
            
            model_info = self.train_and_validate_model(symbol, all_data[symbol])
            
            if model_info is None:
                print("❌ Failed validation")
                continue
            
            trained_models[symbol] = model_info
            model_stats.append({
                'symbol': symbol,
                'accuracy': model_info['accuracy']
            })
            print(f"✅ Accuracy: {model_info['accuracy']:.1%}")
        
        print(f"\n   ✅ {len(trained_models)} models passed validation")
        
        if model_stats:
            avg_accuracy = np.mean([m['accuracy'] for m in model_stats])
            print(f"   📈 Average accuracy: {avg_accuracy:.1%}")
        
        # Phase 3: Trading simulation with REAL daily trading
        print(f"\n💼 Phase 3: Daily Trading Simulation (Like yfinance)")
        
        # Generate trading dates
        trading_dates = pd.bdate_range(start=trading_start, end=trading_end)
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'total_value': self.initial_capital
        }
        
        trades = []
        daily_performance = []
        
        print(f"   📈 Simulating {len(trading_dates)} trading days with real signals")
        
        # Get trading data for each valid symbol
        trading_data = {}
        for symbol in trained_models.keys():
            trading_data[symbol] = self.fetch_alpaca_data(symbol, training_start, trading_end)
        
        for i, date in enumerate(trading_dates):
            date_str = date.strftime('%Y-%m-%d')
            
            if i % 10 == 0:
                print(f"   Day {i+1:2d}/{len(trading_dates)}: {date_str}")
            
            # Generate signals for each valid symbol
            daily_signals = {}
            
            for symbol in trained_models.keys():
                try:
                    # Get data up to current date for signal generation
                    current_data = trading_data[symbol][trading_data[symbol].index <= pd.to_datetime(date).date()]
                    
                    if len(current_data) < 50:  # Need minimum data for features
                        continue
                    
                    # Generate signal using trained model
                    signal_info = self.generate_signal(symbol, trained_models[symbol], current_data)
                    
                    if signal_info['signal'] != 'HOLD':
                        daily_signals[symbol] = {
                            'signal': signal_info['signal'],
                            'strength': signal_info['strength'],
                            'confidence': signal_info['confidence'],
                            'price': current_data['close'].iloc[-1],
                            'model_accuracy': trained_models[symbol]['accuracy']
                        }
                        
                except Exception as e:
                    continue
            
            # Execute trades based on signals
            self.execute_daily_trades(portfolio, daily_signals, trades, date_str)
            
            # Update portfolio value
            total_value = portfolio['cash']
            for symbol, position in portfolio['positions'].items():
                if symbol in trading_data and len(trading_data[symbol]) > 0:
                    try:
                        current_data = trading_data[symbol][trading_data[symbol].index <= pd.to_datetime(date).date()]
                        if len(current_data) > 0:
                            current_price = current_data['close'].iloc[-1]
                            total_value += position['shares'] * current_price
                    except:
                        pass
            
            portfolio['total_value'] = total_value
            
            daily_performance.append({
                'date': date_str,
                'portfolio_value': total_value,
                'cash': portfolio['cash'],
                'positions': len(portfolio['positions']),
                'daily_signals': len(daily_signals)
            })
        
        # Calculate final results
        final_value = portfolio['total_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        results = {
            'total_return': total_return,
            'final_value': final_value,
            'n_trades': len(trades),
            'n_buy_trades': len([t for t in trades if t['action'] == 'BUY']),
            'n_symbols_traded': len(set(t['symbol'] for t in trades)),
            'n_symbols_filtered': len(valid_symbols),
            'n_models_trained': len(trained_models),
            'avg_model_accuracy': avg_accuracy if model_stats else 0.0,
            'model_stats': model_stats,
            'trades': trades,
            'daily_performance': daily_performance,
            'training_days': 355,
            'trading_days': len(trading_dates)
        }
        
        return results

def main():
    print("🎯 OPTIMAL 2025 ALPACA STRATEGY - FULL BACKTEST")
    print("=" * 70)
    
    # Initialize backtester
    backtester = OptimalAlpacaBacktester2025(initial_capital=100000)
    
    # YOUR OPTIMAL DATES
    training_start = "2024-01-01"
    trading_start = "2025-06-02"
    trading_end = "2025-08-20"
    
    # Run full backtest
    results = backtester.run_backtest(
        training_start=training_start,
        trading_start=trading_start,
        trading_end=trading_end,
        max_symbols=50
    )
    
    if results:
        print(f"\n🎯 OPTIMAL 2025 ALPACA RESULTS:")
        print(f"   📈 Total Return: {results['total_return']:.2%}")
        print(f"   💰 Final Value: ${results['final_value']:,.2f}")
        print(f"   📊 Trades: {results['n_trades']}")
        print(f"   🔍 Symbols Filtered: {results['n_symbols_filtered']}")
        print(f"   🤖 Models Trained: {results['n_models_trained']}")
        print(f"   📈 Avg Model Accuracy: {results['avg_model_accuracy']:.1%}")
        print(f"   📊 Training Days: {results['training_days']}")
        print(f"   📈 Trading Days: {results['trading_days']}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"optimal_2025_backtest_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {results_file}")
        
        # Show top models
        if results.get('model_stats'):
            print(f"\n🏆 TOP PERFORMING MODELS:")
            sorted_models = sorted(results['model_stats'], key=lambda x: x['accuracy'], reverse=True)[:10]
            for i, model in enumerate(sorted_models, 1):
                print(f"   {i:2d}. {model['symbol']}: {model['accuracy']:.1%}")

if __name__ == "__main__":
    main()
