"""
🚀 ALPACA HISTORICAL BACKTEST SYSTEM
3-Month Trading Simulation with Real Alpaca Data

Features:
- Uses real Alpaca historical data (2+ years)
- Simulates 3-month trading period
- Real-time signal generation
- Comprehensive performance analysis
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional
import traceback

# Alpaca imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ML imports
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')

class AlpacaHistoricalBacktester:
    """Historical backtesting with real Alpaca data"""
    
    def __init__(self, config_file: str = "alpaca_config.json"):
        self.config = self._load_config(config_file)
        self._setup_alpaca_client()
        
        # Trading parameters
        self.initial_cash = 50000
        self.max_positions = 8
        self.position_size_pct = 0.15
        self.min_signal_strength = 0.4
        
        # Risk management
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.15
        
        # Trading universe
        self.trading_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
        
        print("🚀 ALPACA HISTORICAL BACKTESTER INITIALIZED")
        print(f"🎯 Trading Universe: {', '.join(self.trading_symbols)}")
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load config: {e}")
    
    def _setup_alpaca_client(self):
        """Setup Alpaca data client"""
        alpaca_config = self.config['alpaca']
        
        self.data_client = StockHistoricalDataClient(
            api_key=alpaca_config['api_key'],
            secret_key=alpaca_config['secret_key']
        )
        
        print("✅ Alpaca data client initialized")
    
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical data from Alpaca"""
        try:
            print(f"📊 Fetching {symbol} data from {start_date.date()} to {end_date.date()}")
            
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,  # Daily bars for backtesting
                start=start_date,
                end=end_date
            )
            
            bars = self.data_client.get_stock_bars(request_params)
            
            if symbol not in bars:
                print(f"⚠️ No data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for bar in bars[symbol]:
                data.append({
                    'Date': bar.timestamp.date(),
                    'Open': float(bar.open),
                    'High': float(bar.high),
                    'Low': float(bar.low),
                    'Close': float(bar.close),
                    'Volume': int(bar.volume)
                })
            
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            print(f"✅ Fetched {len(df)} days of {symbol} data")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced trading features"""
        if df.empty or len(df) < 50:
            return df
        
        df = df.copy()
        
        # Basic features
        df['returns'] = df['Close'].pct_change()
        df['rsi'] = self._calculate_rsi(df['Close'], 14)
        
        # Moving averages
        for period in [10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume analysis
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
        
        # Bollinger Bands
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['sma_20'] + (bb_std * 2)
        df['bb_lower'] = df['sma_20'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train_ml_model(self, symbol: str, df: pd.DataFrame, current_date: datetime) -> Dict:
        """Train ML model with data up to current date"""
        try:
            # Use only data up to current date for training
            train_df = df[df['Date'] <= current_date].copy()
            
            if len(train_df) < 100:
                return {'model': None, 'accuracy': 0.0}
            
            # Calculate features
            df_featured = self.calculate_features(train_df)
            if df_featured.empty or len(df_featured) < 50:
                return {'model': None, 'accuracy': 0.0}
            
            # Feature selection
            feature_cols = [
                'rsi', 'macd', 'macd_histogram', 'bb_position',
                'volume_ratio', 'volatility_ratio', 'price_sma_20_ratio', 
                'momentum_5', 'momentum_10'
            ]
            
            available_features = [col for col in feature_cols if col in df_featured.columns]
            if len(available_features) < 5:
                return {'model': None, 'accuracy': 0.0}
            
            X = df_featured[available_features].fillna(0)
            
            # Create target (predict if price will go up in next 5 days)
            forward_returns = df_featured['Close'].shift(-5) / df_featured['Close'] - 1
            y = (forward_returns > 0.02).astype(int)  # 2% threshold
            
            # Remove last 5 rows (no future data available)
            X = X.iloc[:-5]
            y = y.iloc[:-5]
            
            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Clean data
            train_mask = ~(y_train.isna() | X_train.isna().any(axis=1))
            test_mask = ~(y_test.isna() | X_test.isna().any(axis=1))
            
            if train_mask.sum() < 30 or test_mask.sum() < 5:
                return {'model': None, 'accuracy': 0.0}
            
            X_train_clean = X_train[train_mask]
            y_train_clean = y_train[train_mask]
            X_test_clean = X_test[test_mask]
            y_test_clean = y_test[test_mask]
            
            # Train model
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train_clean, y_train_clean)
            
            # Calculate accuracy
            pred = model.predict(X_test_clean)
            accuracy = np.mean(pred == y_test_clean)
            
            return {
                'model': model,
                'feature_cols': available_features,
                'accuracy': accuracy,
                'training_samples': len(X_train_clean)
            }
            
        except Exception as e:
            print(f"❌ ML training failed for {symbol}: {e}")
            return {'model': None, 'accuracy': 0.0}
    
    def generate_trading_signal(self, symbol: str, df: pd.DataFrame, current_date: datetime, 
                               model_info: Dict) -> Dict:
        """Generate trading signal for current date"""
        try:
            # Get data up to current date
            current_df = df[df['Date'] <= current_date].copy()
            
            if len(current_df) < 50:
                return {'signal': 0.0, 'confidence': 0.0, 'reason': 'Insufficient data'}
            
            # Calculate features
            df_featured = self.calculate_features(current_df)
            if df_featured.empty:
                return {'signal': 0.0, 'confidence': 0.0, 'reason': 'Feature calculation failed'}
            
            latest = df_featured.iloc[-1]
            
            # Technical analysis signal
            rsi = latest.get('rsi', 50)
            macd_hist = latest.get('macd_histogram', 0)
            bb_position = latest.get('bb_position', 0.5)
            volume_ratio = latest.get('volume_ratio', 1.0)
            momentum_5 = latest.get('momentum_5', 0)
            momentum_10 = latest.get('momentum_10', 0)
            
            tech_signal = 0.0
            
            # Enhanced technical analysis
            # RSI component
            if rsi < 30:
                tech_signal += 0.4
            elif rsi < 40:
                tech_signal += 0.2
            elif rsi > 70:
                tech_signal -= 0.4
            elif rsi > 60:
                tech_signal -= 0.2
            
            # MACD component
            if macd_hist > 0 and latest.get('macd', 0) > latest.get('macd_signal', 0):
                tech_signal += 0.3
            elif macd_hist < 0 and latest.get('macd', 0) < latest.get('macd_signal', 0):
                tech_signal -= 0.3
            
            # Momentum component
            if momentum_5 > 0.03 and momentum_10 > 0.01:
                tech_signal += 0.2
            elif momentum_5 < -0.03 and momentum_10 < -0.01:
                tech_signal -= 0.2
            
            # Volume confirmation
            if volume_ratio > 1.3:
                tech_signal *= 1.2
            elif volume_ratio < 0.8:
                tech_signal *= 0.8
            
            # Bollinger Band position
            if bb_position < 0.2:
                tech_signal += 0.15
            elif bb_position > 0.8:
                tech_signal -= 0.15
            
            # Price trend
            price_sma_ratio = latest.get('price_sma_20_ratio', 1.0)
            if price_sma_ratio > 1.02:
                tech_signal += 0.1
            elif price_sma_ratio < 0.98:
                tech_signal -= 0.1
            
            # ML signal (if model available)
            ml_signal = 0.0
            ml_confidence = 0.0
            
            if model_info and model_info['model'] is not None:
                try:
                    X = latest[model_info['feature_cols']].fillna(0).values.reshape(1, -1)
                    ml_prob = model_info['model'].predict_proba(X)[0][1]
                    ml_signal = (ml_prob - 0.5) * 2  # Convert to -1 to 1 range
                    ml_confidence = model_info['accuracy']
                except:
                    ml_signal = 0.0
                    ml_confidence = 0.0
            
            # Combine signals
            if ml_confidence > 0.65:
                final_signal = 0.7 * ml_signal + 0.3 * tech_signal
                confidence = ml_confidence
            else:
                final_signal = tech_signal
                confidence = 0.6
            
            final_signal = np.clip(final_signal, -1.0, 1.0)
            
            return {
                'signal': final_signal,
                'confidence': confidence,
                'tech_signal': tech_signal,
                'ml_signal': ml_signal,
                'ml_confidence': ml_confidence,
                'reason': f'RSI:{rsi:.1f}, MACD:{macd_hist:.3f}, Mom5:{momentum_5:.3f}, Vol:{volume_ratio:.1f}'
            }
            
        except Exception as e:
            print(f"❌ Signal generation failed for {symbol}: {e}")
            return {'signal': 0.0, 'confidence': 0.0, 'reason': f'Error: {e}'}
    
    def run_3month_backtest(self, start_date: str = "2024-05-20", end_date: str = "2024-08-20") -> Dict:
        """Run 3-month backtesting with real Alpaca data"""
        
        print("🚀 STARTING 3-MONTH ALPACA BACKTEST")
        print("=" * 50)
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"💰 Initial Capital: ${self.initial_cash:,}")
        print(f"🎯 Trading Symbols: {', '.join(self.trading_symbols)}")
        
        # Convert dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get 2+ years of historical data for training (start from 2 years before)
        data_start = start_dt - timedelta(days=750)  # ~2+ years of data
        
        # Fetch data for all symbols
        print("\n📊 Fetching historical data...")
        stock_data = {}
        for symbol in self.trading_symbols:
            df = self.get_historical_data(symbol, data_start, end_dt)
            if not df.empty and len(df) > 200:  # Need sufficient data
                stock_data[symbol] = df
        
        if not stock_data:
            print("❌ No data available for backtesting")
            return {}
        
        print(f"✅ Data loaded for {len(stock_data)} symbols")
        
        # Initialize portfolio
        cash = self.initial_cash
        positions = {}
        portfolio_values = []
        trades = []
        ml_models = {}
        
        # Get trading dates (business days only)
        trading_dates = pd.bdate_range(start_dt, end_dt)
        
        print(f"\n🎯 Running backtest for {len(trading_dates)} trading days...")
        
        # Track performance
        daily_returns = []
        max_drawdown = 0
        peak_value = self.initial_cash
        
        for i, current_date in enumerate(trading_dates):
            try:
                current_date_dt = current_date.to_pydatetime()
                
                # Update ML models every 20 trading days (~1 month)
                if i % 20 == 0:
                    print(f"\n🧠 Updating ML models for date {current_date.date()}...")
                    for symbol in stock_data.keys():
                        model_info = self.train_ml_model(symbol, stock_data[symbol], current_date_dt)
                        if model_info['model'] is not None:
                            ml_models[symbol] = model_info
                            print(f"   ✅ {symbol}: Accuracy {model_info['accuracy']:.2f}")
                
                # Calculate current portfolio value
                portfolio_value = cash
                for symbol, quantity in positions.items():
                    if symbol in stock_data:
                        # Get price for current date
                        symbol_data = stock_data[symbol]
                        price_data = symbol_data[symbol_data['Date'] <= current_date_dt]
                        if not price_data.empty:
                            current_price = price_data['Close'].iloc[-1]
                            portfolio_value += quantity * current_price
                
                portfolio_values.append({
                    'date': current_date,
                    'value': portfolio_value,
                    'cash': cash,
                    'positions': len(positions)
                })
                
                # Track daily returns and drawdown
                if len(portfolio_values) > 1:
                    prev_value = portfolio_values[-2]['value']
                    daily_return = (portfolio_value / prev_value - 1) * 100
                    daily_returns.append(daily_return)
                    
                    # Update drawdown
                    if portfolio_value > peak_value:
                        peak_value = portfolio_value
                    else:
                        current_drawdown = (peak_value - portfolio_value) / peak_value * 100
                        max_drawdown = max(max_drawdown, current_drawdown)
                
                # Generate signals and trade
                for symbol in stock_data.keys():
                    try:
                        # Check if we have price data for this date
                        symbol_data = stock_data[symbol]
                        available_data = symbol_data[symbol_data['Date'] <= current_date_dt]
                        
                        if len(available_data) < 50:
                            continue
                        
                        current_price = available_data['Close'].iloc[-1]
                        
                        # Generate signal
                        model_info = ml_models.get(symbol, {'model': None, 'accuracy': 0.0})
                        signal_info = self.generate_trading_signal(symbol, symbol_data, current_date_dt, model_info)
                        
                        signal = signal_info['signal']
                        confidence = signal_info['confidence']
                        
                        current_position = positions.get(symbol, 0)
                        position_value = current_position * current_price
                        
                        # Buy logic
                        if (signal > self.min_signal_strength and 
                            confidence > 0.6 and
                            current_position == 0 and 
                            len(positions) < self.max_positions):
                            
                            position_size = portfolio_value * self.position_size_pct
                            if position_size <= cash:
                                quantity = int(position_size / current_price)
                                if quantity > 0:
                                    cost = quantity * current_price
                                    positions[symbol] = quantity
                                    cash -= cost
                                    
                                    trades.append({
                                        'date': current_date,
                                        'symbol': symbol,
                                        'action': 'buy',
                                        'quantity': quantity,
                                        'price': current_price,
                                        'value': cost,
                                        'signal': signal,
                                        'confidence': confidence,
                                        'reason': signal_info['reason']
                                    })
                        
                        # Sell logic
                        elif ((signal < -self.min_signal_strength and confidence > 0.6) or 
                              signal < -0.6) and current_position > 0:
                            
                            proceeds = current_position * current_price
                            cash += proceeds
                            
                            # Calculate P&L for this trade
                            buy_trades = [t for t in trades if t['symbol'] == symbol and t['action'] == 'buy']
                            if buy_trades:
                                avg_buy_price = np.mean([t['price'] for t in buy_trades[-3:]])  # Last 3 buys
                                pnl_pct = (current_price / avg_buy_price - 1) * 100
                            else:
                                pnl_pct = 0
                            
                            del positions[symbol]
                            
                            trades.append({
                                'date': current_date,
                                'symbol': symbol,
                                'action': 'sell',
                                'quantity': current_position,
                                'price': current_price,
                                'value': proceeds,
                                'signal': signal,
                                'confidence': confidence,
                                'pnl_pct': pnl_pct,
                                'reason': signal_info['reason']
                            })
                    
                    except Exception as e:
                        continue
                
                # Progress update
                if i % 20 == 0:
                    progress = (i / len(trading_dates)) * 100
                    print(f"📊 Progress: {progress:.1f}% | Portfolio: ${portfolio_value:,.0f} | Positions: {len(positions)}")
            
            except Exception as e:
                print(f"❌ Error on {current_date.date()}: {e}")
                continue
        
        # Final calculations
        final_value = portfolio_values[-1]['value'] if portfolio_values else self.initial_cash
        total_return = (final_value / self.initial_cash - 1) * 100
        
        # Annualized return
        days = len(trading_dates)
        annualized_return = ((final_value / self.initial_cash) ** (252 / days) - 1) * 100
        
        # Performance metrics
        winning_trades = [t for t in trades if t['action'] == 'sell' and t.get('pnl_pct', 0) > 0]
        losing_trades = [t for t in trades if t['action'] == 'sell' and t.get('pnl_pct', 0) < 0]
        
        win_rate = len(winning_trades) / len([t for t in trades if t['action'] == 'sell']) * 100 if trades else 0
        
        avg_daily_return = np.mean(daily_returns) if daily_returns else 0
        volatility = np.std(daily_returns) if daily_returns else 0
        sharpe_ratio = (avg_daily_return / volatility * np.sqrt(252)) if volatility > 0 else 0
        
        results = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_value': self.initial_cash,
            'final_value': final_value,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(trades),
            'buy_trades': len([t for t in trades if t['action'] == 'buy']),
            'sell_trades': len([t for t in trades if t['action'] == 'sell']),
            'win_rate_pct': win_rate,
            'avg_daily_return_pct': avg_daily_return,
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe_ratio,
            'trading_days': days,
            'final_positions': len(positions),
            'symbols_traded': len(set([t['symbol'] for t in trades])),
            'portfolio_history': portfolio_values,
            'trade_history': trades,
            'ml_models_trained': len([m for m in ml_models.values() if m['model'] is not None])
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print comprehensive backtest results"""
        
        print("\n" + "=" * 60)
        print("🏆 ALPACA 3-MONTH BACKTEST RESULTS")
        print("=" * 60)
        
        print(f"📅 Period: {results['start_date']} to {results['end_date']}")
        print(f"⏰ Trading Days: {results['trading_days']}")
        print()
        
        print("💰 FINANCIAL PERFORMANCE:")
        print(f"   Initial Capital: ${results['initial_value']:,.2f}")
        print(f"   Final Value: ${results['final_value']:,.2f}")
        print(f"   Total Return: {results['total_return_pct']:+.1f}%")
        print(f"   Annualized Return: {results['annualized_return_pct']:+.1f}%")
        print(f"   Max Drawdown: {results['max_drawdown_pct']:.1f}%")
        print()
        
        print("📊 TRADING STATISTICS:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Buy Orders: {results['buy_trades']}")
        print(f"   Sell Orders: {results['sell_trades']}")
        print(f"   Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"   Symbols Traded: {results['symbols_traded']}")
        print()
        
        print("📈 RISK METRICS:")
        print(f"   Daily Volatility: {results['volatility_pct']:.2f}%")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Avg Daily Return: {results['avg_daily_return_pct']:+.3f}%")
        print()
        
        print("🧠 ML PERFORMANCE:")
        print(f"   Models Trained: {results['ml_models_trained']}")
        print(f"   Final Positions: {results['final_positions']}")
        
        # Performance comparison
        baseline_return = 57.6  # Our previous best
        improvement = results['annualized_return_pct'] - baseline_return
        
        print(f"\n🎯 PERFORMANCE vs BASELINE:")
        print(f"   Baseline (Previous): {baseline_return:.1f}%")
        print(f"   Alpaca Real Data: {results['annualized_return_pct']:.1f}%")
        print(f"   Improvement: {improvement:+.1f}%")
        
        if improvement > 5:
            print(f"   🎉 EXCELLENT! {improvement:.1f}% boost with real data!")
        elif improvement > 0:
            print(f"   ✅ Good improvement: {improvement:.1f}% boost")
        else:
            print(f"   ⚠️ Performance below baseline by {abs(improvement):.1f}%")

def main():
    """Run Alpaca historical backtest"""
    
    print("🚀 ALPACA HISTORICAL BACKTESTING SYSTEM")
    print("3-Month Trading with Real Market Data")
    print("=" * 50)
    
    try:
        # Create backtester
        backtester = AlpacaHistoricalBacktester()
        
        # Run 3-month backtest
        print("🎯 Running 3-month backtest with real Alpaca data...")
        results = backtester.run_3month_backtest()
        
        if results:
            # Print results
            backtester.print_results(results)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"alpaca_3month_backtest_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {results_file}")
            
            # Quick recommendation
            annualized_return = results['annualized_return_pct']
            if annualized_return > 60:
                print(f"\n🚀 READY FOR CLOUD DEPLOYMENT!")
                print(f"   System validated with {annualized_return:.1f}% annual returns")
                print(f"   Proceed with GCP deployment")
            else:
                print(f"\n🔧 System shows {annualized_return:.1f}% returns")
                print(f"   Consider further optimization before deployment")
        
        else:
            print("❌ Backtest failed - check your Alpaca configuration")
    
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
