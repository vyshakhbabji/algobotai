"""
🚀 REAL-TIME ALPACA TRADING SYSTEM
Live trading with real-time data and Alpaca execution

Features:
- Real-time market data from Alpaca
- Live ML model updates every hour
- Automatic position management
- Risk controls and monitoring
- Paper trading mode for safety
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import warnings
from typing import Dict, List, Optional
import traceback

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass

# ML imports
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')

class RealTimeAlpacaTrader:
    """Real-time trading system with Alpaca integration"""
    
    def __init__(self, config_file: str = "alpaca_config.json"):
        self.config = self._load_config(config_file)
        self.running = False
        self.positions = {}
        self.ml_models = {}
        self.last_model_update = {}
        
        # Initialize Alpaca clients
        self._setup_alpaca_clients()
        
        # Trading parameters
        self.max_positions = 8
        self.position_size_pct = 0.15
        self.min_signal_strength = 0.4
        self.model_update_interval = 3600  # 1 hour
        
        # Risk management
        self.max_daily_trades = 20
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.15
        self.daily_trade_count = 0
        
        # Trading universe
        self.trading_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
        
        print("🚀 REAL-TIME ALPACA TRADER INITIALIZED")
        print(f"📊 Paper Trading: {self.config['alpaca']['paper_trading']}")
        print(f"🎯 Trading Universe: {', '.join(self.trading_symbols)}")
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load config: {e}")
    
    def _setup_alpaca_clients(self):
        """Setup Alpaca API clients"""
        alpaca_config = self.config['alpaca']
        
        # Trading client
        self.trading_client = TradingClient(
            api_key=alpaca_config['api_key'],
            secret_key=alpaca_config['secret_key'],
            paper=alpaca_config.get('paper_trading', True)
        )
        
        # Data client
        self.data_client = StockHistoricalDataClient(
            api_key=alpaca_config['api_key'],
            secret_key=alpaca_config['secret_key']
        )
        
        # Live data stream
        self.stream = StockDataStream(
            api_key=alpaca_config['api_key'],
            secret_key=alpaca_config['secret_key']
        )
        
        print("✅ Alpaca clients initialized")
    
    def get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            account = self.trading_client.get_account()
            return {
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'status': account.status
            }
        except Exception as e:
            print(f"❌ Error getting account info: {e}")
            return {}
    
    def get_realtime_data(self, symbol: str, period: str = '1D') -> pd.DataFrame:
        """Get real-time historical data for a symbol"""
        try:
            # Get recent bars
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Hour,
                start=datetime.now() - timedelta(days=30)
            )
            
            bars = self.data_client.get_stock_bars(request_params)
            
            if symbol not in bars:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for bar in bars[symbol]:
                data.append({
                    'Date': bar.timestamp,
                    'Open': float(bar.open),
                    'High': float(bar.high),
                    'Low': float(bar.low),
                    'Close': float(bar.close),
                    'Volume': int(bar.volume)
                })
            
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            latest_quote = self.data_client.get_stock_latest_quote(request)
            
            if symbol in latest_quote:
                quote = latest_quote[symbol]
                return float(quote.bid_price + quote.ask_price) / 2
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting price for {symbol}: {e}")
            return None
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced trading features"""
        if df.empty or len(df) < 20:
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
    
    def train_ml_model(self, symbol: str) -> Dict:
        """Train ML model for symbol with latest data"""
        try:
            print(f"🧠 Training ML model for {symbol}...")
            
            # Get recent data
            df = self.get_realtime_data(symbol, '30D')
            if df.empty or len(df) < 100:
                print(f"⚠️ Insufficient data for {symbol}")
                return {'model': None, 'accuracy': 0.0}
            
            # Calculate features
            df_featured = self.calculate_features(df)
            if df_featured.empty:
                return {'model': None, 'accuracy': 0.0}
            
            # Feature selection
            feature_cols = [
                'rsi', 'macd', 'macd_histogram', 'bb_position',
                'volume_ratio', 'price_sma_20_ratio', 'momentum_5', 'momentum_10'
            ]
            
            available_features = [col for col in feature_cols if col in df_featured.columns]
            if len(available_features) < 5:
                return {'model': None, 'accuracy': 0.0}
            
            X = df_featured[available_features].fillna(0)
            
            # Create target (predict if price will go up in next 4 hours)
            forward_returns = df_featured['Close'].shift(-4) / df_featured['Close'] - 1
            y = (forward_returns > 0.01).astype(int)  # 1% threshold
            
            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Clean data
            train_mask = ~(y_train.isna() | X_train.isna().any(axis=1))
            test_mask = ~(y_test.isna() | X_test.isna().any(axis=1))
            
            if train_mask.sum() < 50 or test_mask.sum() < 10:
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
            
            print(f"✅ Model trained for {symbol} - Accuracy: {accuracy:.2f}")
            
            return {
                'model': model,
                'feature_cols': available_features,
                'accuracy': accuracy,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            print(f"❌ ML training failed for {symbol}: {e}")
            return {'model': None, 'accuracy': 0.0}
    
    def generate_trading_signal(self, symbol: str) -> Dict:
        """Generate trading signal for symbol"""
        try:
            # Get current data
            df = self.get_realtime_data(symbol)
            if df.empty or len(df) < 20:
                return {'signal': 0.0, 'confidence': 0.0, 'reason': 'Insufficient data'}
            
            # Calculate features
            df_featured = self.calculate_features(df)
            if df_featured.empty:
                return {'signal': 0.0, 'confidence': 0.0, 'reason': 'Feature calculation failed'}
            
            latest = df_featured.iloc[-1]
            
            # Technical analysis signal
            rsi = latest.get('rsi', 50)
            macd_hist = latest.get('macd_histogram', 0)
            bb_position = latest.get('bb_position', 0.5)
            volume_ratio = latest.get('volume_ratio', 1.0)
            momentum_5 = latest.get('momentum_5', 0)
            
            tech_signal = 0.0
            
            # RSI component
            if rsi < 30:
                tech_signal += 0.3
            elif rsi > 70:
                tech_signal -= 0.3
            
            # MACD component
            if macd_hist > 0:
                tech_signal += 0.2
            elif macd_hist < 0:
                tech_signal -= 0.2
            
            # Momentum component
            if momentum_5 > 0.02:
                tech_signal += 0.2
            elif momentum_5 < -0.02:
                tech_signal -= 0.2
            
            # Volume confirmation
            if volume_ratio > 1.2:
                tech_signal *= 1.1
            
            # Bollinger Band position
            if bb_position < 0.2:
                tech_signal += 0.1
            elif bb_position > 0.8:
                tech_signal -= 0.1
            
            # ML signal (if model available)
            ml_signal = 0.0
            ml_confidence = 0.0
            
            if symbol in self.ml_models and self.ml_models[symbol]['model'] is not None:
                model_info = self.ml_models[symbol]
                X = latest[model_info['feature_cols']].fillna(0).values.reshape(1, -1)
                ml_prob = model_info['model'].predict_proba(X)[0][1]
                ml_signal = (ml_prob - 0.5) * 2  # Convert to -1 to 1 range
                ml_confidence = model_info['accuracy']
            
            # Combine signals
            if ml_confidence > 0.6:
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
                'reason': f'RSI:{rsi:.1f}, MACD:{macd_hist:.3f}, Vol:{volume_ratio:.1f}'
            }
            
        except Exception as e:
            print(f"❌ Signal generation failed for {symbol}: {e}")
            return {'signal': 0.0, 'confidence': 0.0, 'reason': f'Error: {e}'}
    
    def execute_trade(self, symbol: str, action: str, quantity: int, reason: str = "") -> bool:
        """Execute trade through Alpaca"""
        try:
            if self.daily_trade_count >= self.max_daily_trades:
                print(f"⚠️ Daily trade limit reached ({self.max_daily_trades})")
                return False
            
            side = OrderSide.BUY if action == 'buy' else OrderSide.SELL
            
            market_order = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data=market_order)
            
            print(f"✅ {action.upper()} order submitted: {quantity} {symbol}")
            print(f"   Order ID: {order.id}")
            print(f"   Reason: {reason}")
            
            self.daily_trade_count += 1
            return True
            
        except Exception as e:
            print(f"❌ Trade execution failed: {e}")
            return False
    
    def check_risk_management(self) -> Dict:
        """Check risk management rules"""
        try:
            account = self.get_account_info()
            positions = self.trading_client.get_all_positions()
            
            risk_status = {
                'portfolio_value': account.get('portfolio_value', 0),
                'buying_power': account.get('buying_power', 0),
                'position_count': len(positions),
                'max_position_size': 0,
                'total_exposure': 0
            }
            
            # Check individual position sizes
            for position in positions:
                position_value = abs(float(position.market_value))
                risk_status['total_exposure'] += position_value
                
                if position_value > risk_status['max_position_size']:
                    risk_status['max_position_size'] = position_value
            
            # Risk alerts
            alerts = []
            
            if risk_status['position_count'] > self.max_positions:
                alerts.append(f"Too many positions: {risk_status['position_count']}")
            
            max_position_value = account.get('portfolio_value', 0) * self.position_size_pct
            if risk_status['max_position_size'] > max_position_value * 1.5:
                alerts.append(f"Oversized position detected")
            
            risk_status['alerts'] = alerts
            return risk_status
            
        except Exception as e:
            print(f"❌ Risk check failed: {e}")
            return {'alerts': ['Risk check failed']}
    
    def update_models(self):
        """Update ML models for all symbols"""
        print("🔄 Updating ML models...")
        
        for symbol in self.trading_symbols:
            # Check if update needed
            if (symbol not in self.last_model_update or
                (datetime.now() - self.last_model_update[symbol]).seconds > self.model_update_interval):
                
                model_info = self.train_ml_model(symbol)
                if model_info['model'] is not None:
                    self.ml_models[symbol] = model_info
                    self.last_model_update[symbol] = datetime.now()
        
        print(f"✅ Models updated - Active models: {len([m for m in self.ml_models.values() if m['model'] is not None])}")
    
    def trading_loop(self):
        """Main trading loop"""
        print("🎯 Starting trading loop...")
        
        last_model_update = datetime.now() - timedelta(hours=2)  # Force initial update
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Update models every hour
                if (current_time - last_model_update).seconds > self.model_update_interval:
                    self.update_models()
                    last_model_update = current_time
                
                # Check market hours (9:30 AM - 4:00 PM ET)
                market_open = current_time.replace(hour=9, minute=30, second=0)
                market_close = current_time.replace(hour=16, minute=0, second=0)
                
                if not (market_open <= current_time <= market_close):
                    print("📴 Market closed - waiting...")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                # Risk management check
                risk_status = self.check_risk_management()
                if risk_status['alerts']:
                    print(f"⚠️ Risk alerts: {', '.join(risk_status['alerts'])}")
                    time.sleep(60)  # Wait before next check
                    continue
                
                # Generate signals and trade
                print(f"📊 Scanning {len(self.trading_symbols)} symbols...")
                
                for symbol in self.trading_symbols:
                    try:
                        signal_info = self.generate_trading_signal(symbol)
                        signal = signal_info['signal']
                        confidence = signal_info['confidence']
                        
                        print(f"{symbol}: Signal={signal:.2f}, Conf={confidence:.2f} - {signal_info['reason']}")
                        
                        # Check if we should trade
                        if abs(signal) > self.min_signal_strength and confidence > 0.6:
                            
                            current_price = self.get_current_price(symbol)
                            if current_price is None:
                                continue
                            
                            account_info = self.get_account_info()
                            portfolio_value = account_info.get('portfolio_value', 0)
                            
                            # Position sizing
                            position_value = portfolio_value * self.position_size_pct
                            quantity = int(position_value / current_price)
                            
                            if quantity > 0:
                                action = 'buy' if signal > 0 else 'sell'
                                reason = f"Signal: {signal:.2f}, {signal_info['reason']}"
                                
                                success = self.execute_trade(symbol, action, quantity, reason)
                                if success:
                                    print(f"🎯 Trade executed: {action} {quantity} {symbol} @ ${current_price:.2f}")
                    
                    except Exception as e:
                        print(f"❌ Error processing {symbol}: {e}")
                        continue
                
                # Wait before next iteration
                print("⏱️ Waiting 5 minutes before next scan...")
                time.sleep(300)  # 5 minutes
                
            except KeyboardInterrupt:
                print("🛑 Trading stopped by user")
                break
            except Exception as e:
                print(f"❌ Trading loop error: {e}")
                traceback.print_exc()
                time.sleep(60)  # Wait 1 minute before retry
    
    def start_trading(self):
        """Start the real-time trading system"""
        if self.running:
            print("⚠️ Trading system already running")
            return
        
        print("🚀 STARTING REAL-TIME TRADING SYSTEM")
        print("=" * 45)
        
        # Validate connection
        account_info = self.get_account_info()
        if not account_info:
            print("❌ Cannot start - Alpaca connection failed")
            return
        
        print(f"💰 Portfolio Value: ${account_info['portfolio_value']:,.2f}")
        print(f"💵 Buying Power: ${account_info['buying_power']:,.2f}")
        print(f"🎯 Max Positions: {self.max_positions}")
        print(f"📈 Position Size: {self.position_size_pct*100:.1f}% per trade")
        
        # Reset daily counters
        self.daily_trade_count = 0
        
        # Start trading
        self.running = True
        self.trading_loop()
    
    def stop_trading(self):
        """Stop the trading system"""
        print("🛑 Stopping trading system...")
        self.running = False
    
    def get_status(self) -> Dict:
        """Get current system status"""
        account_info = self.get_account_info()
        risk_status = self.check_risk_management()
        
        return {
            'running': self.running,
            'account_info': account_info,
            'risk_status': risk_status,
            'daily_trades': self.daily_trade_count,
            'active_models': len([m for m in self.ml_models.values() if m.get('model') is not None]),
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def main():
    """Main execution function"""
    print("🚀 REAL-TIME ALPACA TRADING SYSTEM")
    print("=" * 40)
    print("Features:")
    print("✅ Real-time Alpaca market data")
    print("✅ ML model updates every hour") 
    print("✅ Automatic risk management")
    print("✅ Paper trading safety mode")
    print("=" * 40)
    
    try:
        # Create trader instance
        trader = RealTimeAlpacaTrader()
        
        # Show initial status
        status = trader.get_status()
        print(f"\n📊 INITIAL STATUS:")
        print(f"   Portfolio: ${status['account_info']['portfolio_value']:,.2f}")
        print(f"   Buying Power: ${status['account_info']['buying_power']:,.2f}")
        print(f"   Paper Trading: {trader.config['alpaca']['paper_trading']}")
        
        # Start trading
        print(f"\n🎯 Starting automated trading...")
        print(f"   Press Ctrl+C to stop")
        
        trader.start_trading()
        
    except KeyboardInterrupt:
        print(f"\n🛑 System stopped by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
