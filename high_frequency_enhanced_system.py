"""
🕐 HIGH-FREQUENCY LIVE TRADING SYSTEM
Enhancement A + B: 4-Hour Frequency + Real-time Data

Targeting 75-85% Annual Returns (vs 57.6% baseline)
- 4-hour trading checks (6am, 10am, 2pm, 6pm)
- Alpaca real-time data integration
- Enhanced ML model retraining every 4 hours
"""

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

warnings.filterwarnings('ignore')

# Enhanced configuration for high-frequency trading
HIGH_FREQ_CONFIG = {
    'trading_frequency_hours': 4,  # Check every 4 hours
    'market_hours': [6, 10, 14, 18],  # 6am, 10am, 2pm, 6pm ET
    'ml_retrain_frequency': 2,  # Retrain every 2 checks (8 hours)
    'use_realtime_data': True,
    'alpaca_enabled': True,
    'position_max_pct': 0.2,  # Max 20% per stock
    'min_signal_strength': 0.4,  # Higher threshold for more selective trading
    'partial_sell_threshold': 0.8,  # Sell 30% when signal weakens
    'max_positions': 8,  # More concentrated portfolio
    'lookback_days': 90,  # 3 months lookback
    'ml_features_enhanced': True
}

# Stock universe - focus on high-volume, liquid stocks
STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
    'V', 'WMT', 'JNJ', 'PG', 'DIS', 'NFLX', 'COST', 'HD', 'BAC', 'XOM'
]

class AlpacaDataProvider:
    """Real-time data provider using Alpaca API"""
    
    def __init__(self):
        # Use paper trading credentials for testing
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY', 'test_key'),
            secret_key=os.getenv('ALPACA_SECRET_KEY', 'test_secret'),
            base_url='https://paper-api.alpaca.markets',  # Paper trading
            api_version='v2'
        )
        self.use_alpaca = os.getenv('ALPACA_API_KEY') is not None
        
    def get_bars(self, symbol: str, timeframe: str, start: str, end: str = None) -> pd.DataFrame:
        """Get historical bars from Alpaca or fallback to Yahoo Finance"""
        try:
            if self.use_alpaca:
                # Convert timeframe to Alpaca format
                tf_map = {'1d': TimeFrame.Day, '1h': TimeFrame.Hour, '4h': '4Hour'}
                alpaca_tf = tf_map.get(timeframe, TimeFrame.Day)
                
                # Get data from Alpaca
                bars = self.api.get_bars(
                    symbol,
                    alpaca_tf,
                    start=start,
                    end=end or datetime.now().strftime('%Y-%m-%d'),
                    asof=None,
                    feed='iex'  # Real-time feed
                ).df
                
                if not bars.empty:
                    bars = bars.reset_index()
                    bars.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'TradeCount', 'VWAP']
                    bars = bars[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    bars['Date'] = pd.to_datetime(bars['Date'])
                    return bars
            
        except Exception as e:
            print(f"⚠️ Alpaca data fetch failed for {symbol}: {e}")
        
        # Fallback to Yahoo Finance
        return self._get_yahoo_data(symbol, timeframe, start, end)
    
    def _get_yahoo_data(self, symbol: str, timeframe: str, start: str, end: str = None) -> pd.DataFrame:
        """Fallback Yahoo Finance data fetcher"""
        try:
            # Calculate period for Yahoo Finance
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end) if end else datetime.now()
            days = (end_date - start_date).days
            
            period = '1y' if days > 300 else '6mo' if days > 150 else '3mo'
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval='1d')
            
            if data.empty:
                return pd.DataFrame()
                
            data = data.reset_index()
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Filter to requested date range
            mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
            return data[mask].copy()
            
        except Exception as e:
            print(f"❌ Yahoo Finance failed for {symbol}: {e}")
            return pd.DataFrame()

class HighFrequencyMLTrader:
    """High-frequency ML-enhanced trading system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or HIGH_FREQ_CONFIG
        self.data_provider = AlpacaDataProvider()
        self.ml_models = {}
        self.regime_models = {}
        self.positions = {}
        self.cash = 50000  # Starting capital
        self.total_value = self.cash
        self.trade_history = []
        self.model_update_counter = 0
        
        # Performance tracking
        self.daily_values = []
        self.daily_returns = []
        self.trade_count = 0
        self.winning_trades = 0
        
        print("🚀 High-Frequency ML Trading System Initialized")
        print(f"📊 Trading Frequency: Every {self.config['trading_frequency_hours']} hours")
        print(f"📡 Real-time Data: {'Enabled (Alpaca)' if self.data_provider.use_alpaca else 'Fallback (Yahoo)'}")
    
    def calculate_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced technical features for ML"""
        if df.empty or len(df) < 20:
            return df
            
        # Price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages (multiple timeframes)
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(df['Close'], 14)
        df['rsi_fast'] = self._calculate_rsi(df['Close'], 7)
        df['rsi_slow'] = self._calculate_rsi(df['Close'], 21)
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        df['bb_middle'] = df['Close'].rolling(bb_period).mean()
        bb_std = df['Close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Volatility features
        df['volatility'] = df['returns'].rolling(20).std()
        df['atr'] = self._calculate_atr(df, 14)
        
        # High-frequency specific features
        df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']
        df['overnight_return'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        
        # Market regime features
        df['trend_strength'] = abs(df['Close'] - df['sma_20']) / df['atr']
        df['momentum'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_acceleration'] = df['momentum'] - df['momentum'].shift(5)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect market regime using ML"""
        if df.empty or len(df) < 50:
            return 'ranging'
        
        # Create regime features
        features = []
        
        # Trend features
        sma_20 = df['Close'].rolling(20).mean()
        sma_50 = df['Close'].rolling(50).mean()
        trend_up = (sma_20 > sma_50).astype(int)
        
        # Volatility regime
        volatility = df['returns'].rolling(20).std()
        high_vol = (volatility > volatility.rolling(50).mean()).astype(int)
        
        # Momentum features
        momentum = df['Close'] / df['Close'].shift(20) - 1
        strong_momentum = (abs(momentum) > 0.05).astype(int)
        
        # Volume regime
        volume_ratio = df['Volume'] / df['Volume'].rolling(50).mean()
        high_volume = (volume_ratio > 1.2).astype(int)
        
        # Combine features for regime detection
        recent_trend = trend_up.tail(10).mean()
        recent_vol = high_vol.tail(10).mean()
        recent_momentum = strong_momentum.tail(10).mean()
        recent_volume = high_volume.tail(10).mean()
        
        # Regime classification
        if recent_trend > 0.6 and recent_momentum > 0.4:
            return 'trending_up'
        elif recent_trend < 0.4 and recent_momentum > 0.4:
            return 'trending_down'
        elif recent_vol > 0.6:
            return 'volatile'
        else:
            return 'ranging'
    
    def train_ml_models(self, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
        """Train ML models for signal enhancement and regime detection"""
        if df.empty or len(df) < 100:
            return {'accuracy': 0.0, 'feature_importance': 0.0}
        
        try:
            # Calculate features
            df = self.calculate_enhanced_features(df)
            df = df.dropna()
            
            if len(df) < 50:
                return {'accuracy': 0.0, 'feature_importance': 0.0}
            
            # Feature columns for ML
            feature_cols = [
                'rsi', 'rsi_fast', 'rsi_slow', 'macd', 'macd_histogram',
                'bb_position', 'volume_ratio', 'volatility', 'atr',
                'price_sma_5_ratio', 'price_sma_10_ratio', 'price_sma_20_ratio',
                'intraday_return', 'high_low_ratio', 'trend_strength',
                'momentum', 'momentum_acceleration'
            ]
            
            # Ensure all feature columns exist
            available_features = [col for col in feature_cols if col in df.columns]
            
            if len(available_features) < 5:
                return {'accuracy': 0.0, 'feature_importance': 0.0}
            
            X = df[available_features].fillna(0)
            
            # Create targets for signal strength prediction (forward returns)
            forward_returns = df['Close'].shift(-5) / df['Close'] - 1
            
            # Signal strength target (0.3 to 1.0 multiplier)
            signal_strength = np.clip(
                0.5 + (forward_returns * 10),  # Scale returns to strength
                0.3, 1.0
            )
            
            # Market regime target
            regime_target = df.apply(lambda row: self.detect_market_regime(
                df.loc[:row.name] if row.name > 50 else df.iloc[:50]
            ), axis=1)
            
            # Encode regime target
            regime_map = {'trending_up': 0, 'trending_down': 1, 'ranging': 2, 'volatile': 3}
            regime_encoded = regime_target.map(regime_map).fillna(2)
            
            # Split data (use 80% for training, 20% for validation)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_signal_train = signal_strength.iloc[:split_idx]
            y_signal_test = signal_strength.iloc[split_idx:]
            y_regime_train = regime_encoded.iloc[:split_idx]
            y_regime_test = regime_encoded.iloc[split_idx:]
            
            # Remove NaN values
            train_mask = ~(y_signal_train.isna() | X_train.isna().any(axis=1))
            test_mask = ~(y_signal_test.isna() | X_test.isna().any(axis=1))
            
            if train_mask.sum() < 20 or test_mask.sum() < 5:
                return {'accuracy': 0.0, 'feature_importance': 0.0}
            
            X_train_clean = X_train[train_mask]
            y_signal_train_clean = y_signal_train[train_mask]
            y_regime_train_clean = y_regime_train[train_mask]
            X_test_clean = X_test[test_mask]
            y_signal_test_clean = y_signal_test[test_mask]
            y_regime_test_clean = y_regime_test[test_mask]
            
            # Train signal strength model (LightGBM Regressor)
            signal_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                objective='regression',
                verbose=-1
            )
            
            signal_model.fit(X_train_clean, y_signal_train_clean)
            signal_pred = signal_model.predict(X_test_clean)
            signal_accuracy = 1 - np.mean(np.abs(signal_pred - y_signal_test_clean))
            
            # Train regime detection model (RandomForest Classifier)
            regime_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                min_samples_split=5
            )
            
            regime_model.fit(X_train_clean, y_regime_train_clean)
            regime_pred = regime_model.predict(X_test_clean)
            regime_accuracy = np.mean(regime_pred == y_regime_test_clean)
            
            # Store models
            self.ml_models[symbol] = {
                'signal_model': signal_model,
                'regime_model': regime_model,
                'feature_cols': available_features,
                'signal_accuracy': signal_accuracy,
                'regime_accuracy': regime_accuracy
            }
            
            return {
                'signal_accuracy': signal_accuracy,
                'regime_accuracy': regime_accuracy,
                'feature_importance': np.mean(signal_model.feature_importances_)
            }
            
        except Exception as e:
            print(f"❌ ML training failed for {symbol}: {e}")
            return {'accuracy': 0.0, 'feature_importance': 0.0}
    
    def get_ml_enhanced_signal(self, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
        """Get ML-enhanced trading signal"""
        if symbol not in self.ml_models or df.empty:
            return {
                'signal_strength': 0.5,
                'market_regime': 'ranging',
                'confidence': 0.0,
                'base_signal': 0.0
            }
        
        try:
            # Calculate features for latest data
            df_featured = self.calculate_enhanced_features(df.copy())
            if df_featured.empty:
                return {'signal_strength': 0.5, 'market_regime': 'ranging', 'confidence': 0.0, 'base_signal': 0.0}
            
            model_info = self.ml_models[symbol]
            latest_row = df_featured.iloc[-1]
            
            # Prepare features
            X = latest_row[model_info['feature_cols']].fillna(0).values.reshape(1, -1)
            
            # Get ML predictions
            signal_strength = model_info['signal_model'].predict(X)[0]
            regime_pred = model_info['regime_model'].predict(X)[0]
            
            # Map regime prediction back to string
            regime_map = {0: 'trending_up', 1: 'trending_down', 2: 'ranging', 3: 'volatile'}
            market_regime = regime_map.get(regime_pred, 'ranging')
            
            # Calculate base signal using traditional indicators
            rsi = latest_row.get('rsi', 50)
            macd = latest_row.get('macd', 0)
            bb_position = latest_row.get('bb_position', 0.5)
            
            # Base signal strength
            base_signal = 0.0
            if rsi < 30 and macd > 0:  # Oversold + positive MACD
                base_signal = 0.7
            elif rsi > 70 and macd < 0:  # Overbought + negative MACD
                base_signal = -0.7
            elif bb_position < 0.2:  # Near lower Bollinger Band
                base_signal = 0.5
            elif bb_position > 0.8:  # Near upper Bollinger Band
                base_signal = -0.5
            
            # Regime-based adjustment
            regime_multiplier = {
                'trending_up': 1.2,
                'trending_down': 0.8,
                'ranging': 1.0,
                'volatile': 0.9
            }.get(market_regime, 1.0)
            
            # Final signal strength (clamp to valid range)
            final_signal_strength = np.clip(signal_strength * regime_multiplier, 0.3, 1.0)
            
            return {
                'signal_strength': final_signal_strength,
                'market_regime': market_regime,
                'confidence': model_info['signal_accuracy'],
                'base_signal': base_signal
            }
            
        except Exception as e:
            print(f"❌ ML signal prediction failed for {symbol}: {e}")
            return {'signal_strength': 0.5, 'market_regime': 'ranging', 'confidence': 0.0, 'base_signal': 0.0}
    
    def make_trading_decision(self, symbol: str, current_price: float, ml_signal: Dict) -> Dict:
        """Make high-frequency trading decision"""
        signal_strength = ml_signal['signal_strength']
        base_signal = ml_signal['base_signal']
        market_regime = ml_signal['market_regime']
        
        current_position = self.positions.get(symbol, 0)
        position_value = current_position * current_price
        
        # Decision logic based on signal strength and regime
        decision = {
            'action': 'hold',
            'quantity': 0,
            'reason': 'No signal',
            'confidence': ml_signal['confidence']
        }
        
        # Calculate position sizing based on Kelly criterion and ML confidence
        max_position_value = self.total_value * self.config['position_max_pct']
        confidence = max(ml_signal['confidence'], 0.1)
        
        if base_signal > self.config['min_signal_strength']:
            # Buy signal
            if current_position == 0:
                # New position
                kelly_fraction = confidence * signal_strength
                position_size = min(max_position_value, self.cash * kelly_fraction)
                quantity = int(position_size / current_price)
                
                if quantity > 0 and len(self.positions) < self.config['max_positions']:
                    decision = {
                        'action': 'buy',
                        'quantity': quantity,
                        'reason': f'New position - Signal: {signal_strength:.2f}, Regime: {market_regime}',
                        'confidence': confidence
                    }
            elif position_value < max_position_value * 0.8:
                # Add to position
                additional_size = (max_position_value - position_value) * 0.5
                quantity = int(additional_size / current_price)
                
                if quantity > 0:
                    decision = {
                        'action': 'buy',
                        'quantity': quantity,
                        'reason': f'Add to position - Strong signal: {signal_strength:.2f}',
                        'confidence': confidence
                    }
        
        elif base_signal < -self.config['min_signal_strength'] and current_position > 0:
            # Sell signal
            if signal_strength < self.config['partial_sell_threshold']:
                # Partial sell
                sell_quantity = int(current_position * 0.3)
                decision = {
                    'action': 'sell',
                    'quantity': sell_quantity,
                    'reason': f'Partial sell - Weak signal: {signal_strength:.2f}',
                    'confidence': confidence
                }
            else:
                # Full sell
                decision = {
                    'action': 'sell',
                    'quantity': current_position,
                    'reason': f'Full sell - Strong negative signal: {base_signal:.2f}',
                    'confidence': confidence
                }
        
        return decision
    
    def execute_trade(self, symbol: str, decision: Dict, current_price: float) -> bool:
        """Execute trading decision"""
        if decision['action'] == 'hold' or decision['quantity'] == 0:
            return True
        
        quantity = decision['quantity']
        cost = quantity * current_price
        
        if decision['action'] == 'buy':
            if cost <= self.cash:
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                self.cash -= cost
                
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': current_price,
                    'cost': cost,
                    'reason': decision['reason'],
                    'confidence': decision['confidence']
                }
                self.trade_history.append(trade)
                self.trade_count += 1
                
                print(f"🟢 BUY {symbol}: {quantity} shares @ ${current_price:.2f} | {decision['reason']}")
                return True
            else:
                print(f"❌ Insufficient cash for {symbol} purchase: Need ${cost:.2f}, Have ${self.cash:.2f}")
                return False
        
        elif decision['action'] == 'sell':
            current_position = self.positions.get(symbol, 0)
            sell_quantity = min(quantity, current_position)
            
            if sell_quantity > 0:
                proceeds = sell_quantity * current_price
                self.positions[symbol] -= sell_quantity
                self.cash += proceeds
                
                if self.positions[symbol] <= 0:
                    del self.positions[symbol]
                
                # Calculate P&L for this trade
                profit = proceeds - (sell_quantity * current_price)  # Simplified P&L
                if profit > 0:
                    self.winning_trades += 1
                
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': sell_quantity,
                    'price': current_price,
                    'proceeds': proceeds,
                    'reason': decision['reason'],
                    'confidence': decision['confidence']
                }
                self.trade_history.append(trade)
                self.trade_count += 1
                
                print(f"🔴 SELL {symbol}: {sell_quantity} shares @ ${current_price:.2f} | {decision['reason']}")
                return True
        
        return False
    
    def run_trading_session(self, simulation_date: datetime) -> Dict:
        """Run a single high-frequency trading session"""
        session_results = {
            'timestamp': simulation_date,
            'trades_executed': 0,
            'ml_models_updated': 0,
            'portfolio_value': 0,
            'signals_generated': 0
        }
        
        print(f"\n🕐 Trading Session: {simulation_date.strftime('%Y-%m-%d %H:%M')}")
        
        # Update ML models if needed
        if self.model_update_counter % self.config['ml_retrain_frequency'] == 0:
            print("🧠 Retraining ML models...")
            for symbol in STOCK_UNIVERSE[:10]:  # Focus on top 10 for speed
                try:
                    # Get updated data
                    start_date = (simulation_date - timedelta(days=self.config['lookback_days'])).strftime('%Y-%m-%d')
                    end_date = simulation_date.strftime('%Y-%m-%d')
                    
                    df = self.data_provider.get_bars(symbol, '1d', start_date, end_date)
                    if not df.empty:
                        metrics = self.train_ml_models(symbol, df)
                        if metrics['signal_accuracy'] > 0:
                            session_results['ml_models_updated'] += 1
                            print(f"  ✅ {symbol}: Signal Acc: {metrics['signal_accuracy']:.1%}")
                
                except Exception as e:
                    print(f"  ❌ Failed to update {symbol}: {e}")
        
        self.model_update_counter += 1
        
        # Generate signals and execute trades
        for symbol in STOCK_UNIVERSE:
            try:
                # Get recent data for signal generation
                start_date = (simulation_date - timedelta(days=30)).strftime('%Y-%m-%d')
                end_date = simulation_date.strftime('%Y-%m-%d')
                
                df = self.data_provider.get_bars(symbol, '1d', start_date, end_date)
                if df.empty:
                    continue
                
                current_price = df['Close'].iloc[-1]
                
                # Get ML-enhanced signal
                ml_signal = self.get_ml_enhanced_signal(symbol, df)
                session_results['signals_generated'] += 1
                
                # Make trading decision
                decision = self.make_trading_decision(symbol, current_price, ml_signal)
                
                # Execute trade if needed
                if decision['action'] != 'hold':
                    success = self.execute_trade(symbol, decision, current_price)
                    if success:
                        session_results['trades_executed'] += 1
                
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
        
        # Update portfolio value
        portfolio_value = self.cash
        for symbol, quantity in self.positions.items():
            try:
                df = self.data_provider.get_bars(symbol, '1d', 
                    (simulation_date - timedelta(days=7)).strftime('%Y-%m-%d'),
                    simulation_date.strftime('%Y-%m-%d'))
                if not df.empty:
                    current_price = df['Close'].iloc[-1]
                    portfolio_value += quantity * current_price
            except:
                pass
        
        self.total_value = portfolio_value
        session_results['portfolio_value'] = portfolio_value
        
        print(f"💰 Portfolio Value: ${portfolio_value:,.2f} | Trades: {session_results['trades_executed']}")
        
        return session_results
    
    def run_high_frequency_simulation(self, start_date: str, end_date: str) -> Dict:
        """Run complete high-frequency trading simulation"""
        print("🚀 Starting High-Frequency Live Trading Simulation")
        print("=" * 60)
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Track performance
        session_results = []
        portfolio_values = []
        
        # Generate trading sessions (every 4 hours during weekdays)
        current_date = start_dt
        while current_date <= end_dt:
            # Only trade on weekdays
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                for hour in self.config['market_hours']:
                    session_time = current_date.replace(hour=hour, minute=0, second=0)
                    if session_time <= end_dt:
                        try:
                            session_result = self.run_trading_session(session_time)
                            session_results.append(session_result)
                            portfolio_values.append(session_result['portfolio_value'])
                        except Exception as e:
                            print(f"❌ Session failed at {session_time}: {e}")
            
            current_date += timedelta(days=1)
        
        # Calculate final performance metrics
        final_value = portfolio_values[-1] if portfolio_values else self.cash
        total_return = (final_value / self.cash - 1) * 100
        
        # Annualized return
        trading_days = len([d for d in pd.date_range(start_dt, end_dt) if d.weekday() < 5])
        annualized_return = ((final_value / self.cash) ** (252 / trading_days) - 1) * 100
        
        # Win rate
        win_rate = (self.winning_trades / max(self.trade_count, 1)) * 100
        
        # Calculate max drawdown
        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        results = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.cash + sum(self.positions.get(s, 0) * 100 for s in self.positions),  # Estimate
            'final_value': final_value,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'total_trades': self.trade_count,
            'winning_trades': self.winning_trades,
            'win_rate_pct': win_rate,
            'max_drawdown_pct': max_drawdown,
            'trading_sessions': len(session_results),
            'avg_trades_per_session': np.mean([r['trades_executed'] for r in session_results]),
            'final_positions': dict(self.positions),
            'cash_remaining': self.cash,
            'enhancement_summary': {
                'total_signals_generated': sum(r['signals_generated'] for r in session_results),
                'ml_model_updates': sum(r['ml_models_updated'] for r in session_results),
                'trading_frequency_hours': self.config['trading_frequency_hours'],
                'realtime_data_enabled': self.data_provider.use_alpaca
            }
        }
        
        print("\n" + "=" * 60)
        print("🎯 HIGH-FREQUENCY SIMULATION RESULTS")
        print("=" * 60)
        print(f"📊 Total Return: {total_return:.1f}%")
        print(f"📈 Annualized Return: {annualized_return:.1f}%")
        print(f"💰 Final Value: ${final_value:,.2f}")
        print(f"📈 Total Trades: {self.trade_count}")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        print(f"📉 Max Drawdown: {max_drawdown:.1f}%")
        print(f"🕐 Trading Sessions: {len(session_results)}")
        print(f"🔄 Avg Trades/Session: {results['avg_trades_per_session']:.1f}")
        print(f"🧠 ML Model Updates: {results['enhancement_summary']['ml_model_updates']}")
        print(f"📡 Real-time Data: {'✅' if self.data_provider.use_alpaca else '❌ (Yahoo fallback)'}")
        
        return results

def main():
    """Run high-frequency trading simulation"""
    
    # Test period (3 months)
    start_date = "2024-05-20"
    end_date = "2024-08-20"
    
    print("🚀 HIGH-FREQUENCY ML TRADING SYSTEM")
    print("Enhancement A + B: 4-Hour Frequency + Real-time Data")
    print("Target: 75-85% Annual Returns")
    print("=" * 60)
    
    # Initialize trader
    trader = HighFrequencyMLTrader(HIGH_FREQ_CONFIG)
    
    # Run simulation
    results = trader.run_high_frequency_simulation(start_date, end_date)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"high_frequency_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Compare with baseline
    baseline_return = 57.6
    improvement = results['annualized_return_pct'] - baseline_return
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"   Baseline (Daily): {baseline_return:.1f}%")
    print(f"   High-Freq (4hr): {results['annualized_return_pct']:.1f}%")
    print(f"   Improvement: {improvement:+.1f}%")
    
    if improvement > 0:
        print(f"🎉 SUCCESS! Enhancement A+B delivered {improvement:.1f}% boost!")
    else:
        print(f"⚠️ Need optimization - {improvement:.1f}% vs target +15-25%")
    
    return results

if __name__ == "__main__":
    main()
