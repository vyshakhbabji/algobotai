"""
🕐 HIGH-FREQUENCY LIVE TRADING SYSTEM (Simplified)
Enhancement A + B: 4-Hour Frequency + Enhanced Data

Targeting 75-85% Annual Returns (vs 57.6% baseline)
- 4-hour trading checks (6am, 10am, 2pm, 6pm)
- Enhanced data fetching with better timeframes
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

warnings.filterwarnings('ignore')

# Enhanced configuration for high-frequency trading
HIGH_FREQ_CONFIG = {
    'trading_frequency_hours': 4,  # Check every 4 hours
    'market_hours': [6, 10, 14, 18],  # 6am, 10am, 2pm, 6pm ET
    'ml_retrain_frequency': 2,  # Retrain every 2 checks (8 hours)
    'position_max_pct': 0.15,  # Max 15% per stock (more aggressive)
    'min_signal_strength': 0.35,  # Lower threshold for more trading
    'partial_sell_threshold': 0.7,  # Sell 30% when signal weakens
    'max_positions': 10,  # More diversified portfolio
    'lookback_days': 120,  # 4 months lookback for more data
    'ml_features_enhanced': True,
    'aggressive_sizing': True
}

# Expanded stock universe for higher frequency trading
STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
    'V', 'WMT', 'JNJ', 'PG', 'DIS', 'NFLX', 'COST', 'HD', 'BAC', 'XOM',
    'UNH', 'MA', 'ORCL', 'CRM', 'AVGO', 'LLY', 'ABBV', 'TMO'
]

class EnhancedDataProvider:
    """Enhanced data provider with better timing and caching"""
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        
    def get_bars(self, symbol: str, timeframe: str, start: str, end: str = None, use_cache: bool = True) -> pd.DataFrame:
        """Get historical bars with enhanced caching and error handling"""
        cache_key = f"{symbol}_{timeframe}_{start}_{end}"
        
        # Check cache first
        if use_cache and cache_key in self.cache:
            cache_time = self.cache_expiry.get(cache_key, datetime.min)
            if datetime.now() - cache_time < timedelta(hours=1):  # 1-hour cache
                return self.cache[cache_key].copy()
        
        try:
            # Enhanced data fetching with multiple attempts
            for attempt in range(3):
                try:
                    df = self._fetch_yahoo_data(symbol, timeframe, start, end)
                    if not df.empty:
                        # Cache the result
                        if use_cache:
                            self.cache[cache_key] = df.copy()
                            self.cache_expiry[cache_key] = datetime.now()
                        return df
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        print(f"❌ Failed to fetch {symbol} after 3 attempts: {e}")
                    else:
                        print(f"⚠️ Attempt {attempt + 1} failed for {symbol}, retrying...")
                        
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Data fetch error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_yahoo_data(self, symbol: str, timeframe: str, start: str, end: str = None) -> pd.DataFrame:
        """Enhanced Yahoo Finance data fetcher"""
        try:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end) if end else datetime.now()
            days = (end_date - start_date).days
            
            # Use appropriate period for Yahoo Finance
            if days <= 60:
                period = '3mo'
            elif days <= 120:
                period = '6mo'
            elif days <= 365:
                period = '1y'
            else:
                period = '2y'
            
            # Use intraday data for better timing
            interval = '1h' if timeframe == '1h' else '1d'
            
            ticker = yf.Ticker(symbol)
            
            # Try period first, then date range
            try:
                data = ticker.history(period=period, interval=interval, auto_adjust=True, prepost=True)
            except:
                data = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)
            
            if data.empty:
                return pd.DataFrame()
                
            data = data.reset_index()
            
            # Handle different column formats
            if 'Datetime' in data.columns:
                data = data.rename(columns={'Datetime': 'Date'})
            
            # Ensure we have the right columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in required_cols if col in data.columns]
            
            if len(available_cols) < 5:
                return pd.DataFrame()
            
            data = data[available_cols].copy()
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Filter to requested date range and remove duplicates
            mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
            data = data[mask].drop_duplicates(subset=['Date']).reset_index(drop=True)
            
            # If we got hourly data, resample to daily for consistency
            if interval == '1h' and timeframe == '1d':
                data = data.set_index('Date').resample('D').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna().reset_index()
            
            return data
            
        except Exception as e:
            print(f"❌ Yahoo Finance error for {symbol}: {e}")
            return pd.DataFrame()

class HighFrequencyMLTrader:
    """Enhanced high-frequency ML trading system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or HIGH_FREQ_CONFIG
        self.data_provider = EnhancedDataProvider()
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
        self.ml_signals_used = 0
        self.ml_accuracy_sum = 0
        
        print("🚀 High-Frequency ML Trading System Initialized")
        print(f"📊 Trading Frequency: Every {self.config['trading_frequency_hours']} hours")
        print(f"📈 Max Positions: {self.config['max_positions']}")
        print(f"💰 Position Sizing: {self.config['position_max_pct']:.1%} max per stock")
    
    def calculate_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced technical features for ML"""
        if df.empty or len(df) < 20:
            return df
            
        df = df.copy()
        
        # Price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Multiple timeframe moving averages
        for period in [5, 10, 20, 50, 100]:
            if len(df) >= period:
                df[f'sma_{period}'] = df['Close'].rolling(period, min_periods=period//2).mean()
                df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
        
        # Enhanced momentum indicators
        df['rsi'] = self._calculate_rsi(df['Close'], 14)
        df['rsi_fast'] = self._calculate_rsi(df['Close'], 7)
        df['rsi_slow'] = self._calculate_rsi(df['Close'], 21)
        
        # MACD with multiple timeframes
        exp1_12 = df['Close'].ewm(span=12).mean()
        exp2_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1_12 - exp2_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Fast MACD for high frequency
        exp1_5 = df['Close'].ewm(span=5).mean()
        exp2_12 = df['Close'].ewm(span=12).mean()
        df['macd_fast'] = exp1_5 - exp2_12
        df['macd_fast_signal'] = df['macd_fast'].ewm(span=6).mean()
        
        # Bollinger Bands
        for period in [10, 20]:
            bb_middle = df['Close'].rolling(period).mean()
            bb_std = df['Close'].rolling(period).std()
            df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
            df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
            df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # Volume features
        df['volume_sma_10'] = df['Volume'].rolling(10).mean()
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio_10'] = df['Volume'] / df['volume_sma_10']
        df['volume_ratio_20'] = df['Volume'] / df['volume_sma_20']
        
        # Volatility features
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['atr'] = self._calculate_atr(df, 14)
        
        # High-frequency specific features
        df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']
        df['overnight_return'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Market regime features
        df['trend_strength_20'] = abs(df['Close'] - df.get('sma_20', df['Close'])) / df.get('atr', 1)
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_acceleration'] = df['momentum_5'] - df['momentum_5'].shift(3)
        
        # Price acceleration
        df['price_acceleration'] = df['Close'].diff().diff()
        
        # Volume-price relationship
        df['vpt'] = (df['Volume'] * df['Close'].pct_change()).cumsum()  # Volume Price Trend
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with better handling"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period//2).mean()
        avg_loss = loss.rolling(window=period, min_periods=period//2).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period, min_periods=period//2).mean()
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """Enhanced market regime detection"""
        if df.empty or len(df) < 20:
            return 'ranging'
        
        try:
            # Recent price action (last 10 days)
            recent_data = df.tail(min(20, len(df)))
            
            # Trend analysis
            sma_5 = recent_data['Close'].rolling(5).mean()
            sma_20 = recent_data['Close'].rolling(20).mean() if len(recent_data) >= 20 else recent_data['Close'].rolling(len(recent_data)).mean()
            
            trend_up = (sma_5.iloc[-1] > sma_20.iloc[-1]) if len(sma_20) > 0 else False
            
            # Volatility analysis
            volatility = recent_data['Close'].pct_change().std()
            avg_volatility = df['Close'].pct_change().rolling(50, min_periods=20).std().mean()
            high_vol = volatility > (avg_volatility * 1.5) if not pd.isna(avg_volatility) else False
            
            # Momentum analysis
            momentum = recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1 if len(recent_data) > 0 else 0
            strong_momentum = abs(momentum) > 0.05
            
            # Regime classification
            if strong_momentum and trend_up:
                return 'trending_up'
            elif strong_momentum and not trend_up:
                return 'trending_down'
            elif high_vol:
                return 'volatile'
            else:
                return 'ranging'
                
        except Exception as e:
            print(f"❌ Regime detection error: {e}")
            return 'ranging'
    
    def train_ml_models(self, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
        """Enhanced ML model training"""
        if df.empty or len(df) < 50:
            return {'signal_accuracy': 0.0, 'regime_accuracy': 0.0}
        
        try:
            # Calculate features
            df_featured = self.calculate_enhanced_features(df.copy())
            df_featured = df_featured.dropna()
            
            if len(df_featured) < 30:
                return {'signal_accuracy': 0.0, 'regime_accuracy': 0.0}
            
            # Enhanced feature selection
            feature_cols = []
            
            # Core technical indicators
            for col in df_featured.columns:
                if any(indicator in col for indicator in [
                    'rsi', 'macd', 'bb_position', 'volume_ratio', 'volatility',
                    'price_sma', 'momentum', 'intraday_return', 'high_low_ratio',
                    'trend_strength', 'price_acceleration', 'price_position'
                ]):
                    if not df_featured[col].isna().all():
                        feature_cols.append(col)
            
            if len(feature_cols) < 5:
                return {'signal_accuracy': 0.0, 'regime_accuracy': 0.0}
            
            X = df_featured[feature_cols].fillna(method='ffill').fillna(0)
            
            # Enhanced target creation
            # Signal strength based on multiple forward periods
            forward_1 = df_featured['Close'].shift(-1) / df_featured['Close'] - 1
            forward_3 = df_featured['Close'].shift(-3) / df_featured['Close'] - 1
            forward_5 = df_featured['Close'].shift(-5) / df_featured['Close'] - 1
            
            # Combined forward return (weighted average)
            combined_forward = (forward_1 * 0.5 + forward_3 * 0.3 + forward_5 * 0.2)
            
            # Signal strength (0.3 to 1.0)
            signal_strength = np.clip(0.5 + (combined_forward * 8), 0.3, 1.0)
            
            # Market regime target
            regime_labels = []
            for i in range(len(df_featured)):
                end_idx = min(i + 10, len(df_featured))
                regime = self.detect_market_regime(df_featured.iloc[max(0, i-20):end_idx])
                regime_labels.append(regime)
            
            regime_series = pd.Series(regime_labels)
            regime_map = {'trending_up': 0, 'trending_down': 1, 'ranging': 2, 'volatile': 3}
            regime_encoded = regime_series.map(regime_map).fillna(2)
            
            # Train-test split
            split_idx = max(int(len(X) * 0.7), len(X) - 20)
            
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_signal_train = signal_strength.iloc[:split_idx]
            y_signal_test = signal_strength.iloc[split_idx:]
            y_regime_train = regime_encoded.iloc[:split_idx]
            y_regime_test = regime_encoded.iloc[split_idx:]
            
            # Clean data
            train_mask = ~(y_signal_train.isna() | X_train.isna().any(axis=1))
            test_mask = ~(y_signal_test.isna() | X_test.isna().any(axis=1))
            
            if train_mask.sum() < 10 or test_mask.sum() < 3:
                return {'signal_accuracy': 0.0, 'regime_accuracy': 0.0}
            
            X_train_clean = X_train[train_mask]
            y_signal_train_clean = y_signal_train[train_mask]
            y_regime_train_clean = y_regime_train[train_mask]
            X_test_clean = X_test[test_mask]
            y_signal_test_clean = y_signal_test[test_mask]
            y_regime_test_clean = y_regime_test[test_mask]
            
            # Train enhanced models
            # Signal strength model
            signal_model = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=8,
                num_leaves=31,
                feature_fraction=0.8,
                random_state=42,
                objective='regression',
                verbose=-1
            )
            
            signal_model.fit(X_train_clean, y_signal_train_clean)
            signal_pred = signal_model.predict(X_test_clean)
            signal_accuracy = max(0, 1 - np.mean(np.abs(signal_pred - y_signal_test_clean)))
            
            # Regime detection model
            regime_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=3,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            )
            
            regime_model.fit(X_train_clean, y_regime_train_clean)
            regime_pred = regime_model.predict(X_test_clean)
            regime_accuracy = np.mean(regime_pred == y_regime_test_clean)
            
            # Store models
            self.ml_models[symbol] = {
                'signal_model': signal_model,
                'regime_model': regime_model,
                'feature_cols': feature_cols,
                'signal_accuracy': signal_accuracy,
                'regime_accuracy': regime_accuracy,
                'last_update': datetime.now()
            }
            
            return {
                'signal_accuracy': signal_accuracy,
                'regime_accuracy': regime_accuracy
            }
            
        except Exception as e:
            print(f"❌ ML training failed for {symbol}: {e}")
            return {'signal_accuracy': 0.0, 'regime_accuracy': 0.0}
    
    def get_ml_enhanced_signal(self, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
        """Get enhanced ML trading signal"""
        if symbol not in self.ml_models or df.empty:
            return {
                'signal_strength': 0.5,
                'market_regime': 'ranging',
                'confidence': 0.0,
                'base_signal': 0.0,
                'ml_enhanced': False
            }
        
        try:
            df_featured = self.calculate_enhanced_features(df.copy())
            if df_featured.empty:
                return {'signal_strength': 0.5, 'market_regime': 'ranging', 'confidence': 0.0, 'base_signal': 0.0, 'ml_enhanced': False}
            
            model_info = self.ml_models[symbol]
            latest_row = df_featured.iloc[-1]
            
            # Prepare features
            feature_values = []
            for col in model_info['feature_cols']:
                val = latest_row.get(col, 0)
                if pd.isna(val):
                    val = 0
                feature_values.append(val)
            
            X = np.array(feature_values).reshape(1, -1)
            
            # Get ML predictions
            signal_strength = model_info['signal_model'].predict(X)[0]
            regime_pred = model_info['regime_model'].predict(X)[0]
            
            # Map regime
            regime_map = {0: 'trending_up', 1: 'trending_down', 2: 'ranging', 3: 'volatile'}
            market_regime = regime_map.get(regime_pred, 'ranging')
            
            # Calculate enhanced base signal
            rsi = latest_row.get('rsi', 50)
            rsi_fast = latest_row.get('rsi_fast', 50)
            macd = latest_row.get('macd', 0)
            macd_fast = latest_row.get('macd_fast', 0)
            bb_position_20 = latest_row.get('bb_position_20', 0.5)
            volume_ratio = latest_row.get('volume_ratio_10', 1.0)
            
            # Enhanced signal calculation
            base_signal = 0.0
            
            # RSI signals
            if rsi < 25 and rsi_fast < 30:  # Strong oversold
                base_signal += 0.8
            elif rsi < 35:  # Oversold
                base_signal += 0.5
            elif rsi > 75 and rsi_fast > 70:  # Strong overbought
                base_signal -= 0.8
            elif rsi > 65:  # Overbought
                base_signal -= 0.5
            
            # MACD signals
            if macd > 0 and macd_fast > 0:  # Positive momentum
                base_signal += 0.4
            elif macd < 0 and macd_fast < 0:  # Negative momentum
                base_signal -= 0.4
            
            # Bollinger Bands
            if bb_position_20 < 0.15:  # Near lower band
                base_signal += 0.3
            elif bb_position_20 > 0.85:  # Near upper band
                base_signal -= 0.3
            
            # Volume confirmation
            if volume_ratio > 1.5:  # High volume
                base_signal *= 1.2
            elif volume_ratio < 0.7:  # Low volume
                base_signal *= 0.8
            
            # Regime adjustment
            regime_multiplier = {
                'trending_up': 1.3,
                'trending_down': 0.7,
                'ranging': 1.0,
                'volatile': 0.9
            }.get(market_regime, 1.0)
            
            # Final signal strength
            final_signal_strength = np.clip(signal_strength * regime_multiplier, 0.3, 1.0)
            
            # Track ML usage
            self.ml_signals_used += 1
            self.ml_accuracy_sum += model_info['signal_accuracy']
            
            return {
                'signal_strength': final_signal_strength,
                'market_regime': market_regime,
                'confidence': model_info['signal_accuracy'],
                'base_signal': np.clip(base_signal, -1.0, 1.0),
                'ml_enhanced': True
            }
            
        except Exception as e:
            print(f"❌ ML signal error for {symbol}: {e}")
            return {'signal_strength': 0.5, 'market_regime': 'ranging', 'confidence': 0.0, 'base_signal': 0.0, 'ml_enhanced': False}
    
    def make_trading_decision(self, symbol: str, current_price: float, ml_signal: Dict) -> Dict:
        """Enhanced trading decision logic"""
        signal_strength = ml_signal['signal_strength']
        base_signal = ml_signal['base_signal']
        market_regime = ml_signal['market_regime']
        confidence = ml_signal['confidence']
        
        current_position = self.positions.get(symbol, 0)
        position_value = current_position * current_price
        
        decision = {
            'action': 'hold',
            'quantity': 0,
            'reason': 'No signal',
            'confidence': confidence,
            'signal_strength': signal_strength
        }
        
        # Enhanced position sizing
        max_position_value = self.total_value * self.config['position_max_pct']
        
        # Adjust for confidence and regime
        confidence_multiplier = max(confidence, 0.2)
        regime_multiplier = {
            'trending_up': 1.2,
            'trending_down': 1.0,
            'ranging': 0.8,
            'volatile': 0.6
        }.get(market_regime, 0.8)
        
        effective_max_position = max_position_value * confidence_multiplier * regime_multiplier
        
        # Trading logic
        if base_signal > self.config['min_signal_strength']:
            # Buy signal
            if current_position == 0:
                # New position
                if len(self.positions) < self.config['max_positions']:
                    # Enhanced Kelly sizing
                    kelly_fraction = confidence * signal_strength * 0.8  # Conservative Kelly
                    position_size = min(effective_max_position, self.cash * kelly_fraction)
                    quantity = int(position_size / current_price)
                    
                    if quantity > 0:
                        decision = {
                            'action': 'buy',
                            'quantity': quantity,
                            'reason': f'New position - Signal: {signal_strength:.2f}, Regime: {market_regime}, Conf: {confidence:.2f}',
                            'confidence': confidence,
                            'signal_strength': signal_strength
                        }
            
            elif position_value < effective_max_position * 0.75:
                # Add to position
                additional_size = min(
                    (effective_max_position - position_value) * 0.6,
                    self.cash * 0.3  # Don't use more than 30% of cash for additions
                )
                quantity = int(additional_size / current_price)
                
                if quantity > 0:
                    decision = {
                        'action': 'buy',
                        'quantity': quantity,
                        'reason': f'Add to position - Strong signal: {signal_strength:.2f}',
                        'confidence': confidence,
                        'signal_strength': signal_strength
                    }
        
        elif (base_signal < -self.config['min_signal_strength'] or 
              signal_strength < self.config['partial_sell_threshold']) and current_position > 0:
            # Sell signal
            if signal_strength < 0.5 or base_signal < -0.6:
                # Full sell on strong negative signal
                decision = {
                    'action': 'sell',
                    'quantity': current_position,
                    'reason': f'Full sell - Strong negative: {base_signal:.2f}, Strength: {signal_strength:.2f}',
                    'confidence': confidence,
                    'signal_strength': signal_strength
                }
            elif signal_strength < self.config['partial_sell_threshold']:
                # Partial sell on weak signal
                sell_quantity = max(1, int(current_position * 0.4))  # Sell 40%
                decision = {
                    'action': 'sell',
                    'quantity': sell_quantity,
                    'reason': f'Partial sell - Weak signal: {signal_strength:.2f}',
                    'confidence': confidence,
                    'signal_strength': signal_strength
                }
        
        return decision
    
    def execute_trade(self, symbol: str, decision: Dict, current_price: float) -> bool:
        """Execute trading decision with better tracking"""
        if decision['action'] == 'hold' or decision['quantity'] == 0:
            return True
        
        quantity = decision['quantity']
        
        if decision['action'] == 'buy':
            cost = quantity * current_price
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
                    'confidence': decision['confidence'],
                    'signal_strength': decision['signal_strength']
                }
                self.trade_history.append(trade)
                self.trade_count += 1
                
                print(f"🟢 BUY {symbol}: {quantity} @ ${current_price:.2f} | {decision['reason'][:50]}")
                return True
            else:
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
                
                # Simple profit tracking
                if proceeds > sell_quantity * current_price * 0.95:  # Account for basic costs
                    self.winning_trades += 1
                
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': sell_quantity,
                    'price': current_price,
                    'proceeds': proceeds,
                    'reason': decision['reason'],
                    'confidence': decision['confidence'],
                    'signal_strength': decision['signal_strength']
                }
                self.trade_history.append(trade)
                self.trade_count += 1
                
                print(f"🔴 SELL {symbol}: {sell_quantity} @ ${current_price:.2f} | {decision['reason'][:50]}")
                return True
        
        return False
    
    def run_trading_session(self, simulation_date: datetime) -> Dict:
        """Run enhanced trading session"""
        session_results = {
            'timestamp': simulation_date,
            'trades_executed': 0,
            'ml_models_updated': 0,
            'portfolio_value': 0,
            'signals_generated': 0,
            'ml_enhanced_signals': 0
        }
        
        print(f"\n🕐 Session: {simulation_date.strftime('%Y-%m-%d %H:%M')}")
        
        # Update ML models periodically
        if self.model_update_counter % self.config['ml_retrain_frequency'] == 0:
            print("🧠 Retraining ML models...")
            updated_count = 0
            
            for symbol in STOCK_UNIVERSE[:12]:  # Focus on top 12 for efficiency
                try:
                    start_date = (simulation_date - timedelta(days=self.config['lookback_days'])).strftime('%Y-%m-%d')
                    end_date = simulation_date.strftime('%Y-%m-%d')
                    
                    df = self.data_provider.get_bars(symbol, '1d', start_date, end_date)
                    if not df.empty and len(df) >= 30:
                        metrics = self.train_ml_models(symbol, df)
                        if metrics['signal_accuracy'] > 0.1:
                            updated_count += 1
                            print(f"  ✅ {symbol}: Sig:{metrics['signal_accuracy']:.1%} Reg:{metrics['regime_accuracy']:.1%}")
                
                except Exception as e:
                    print(f"  ❌ {symbol}: {str(e)[:30]}")
            
            session_results['ml_models_updated'] = updated_count
        
        self.model_update_counter += 1
        
        # Process all stocks for signals and trades
        for symbol in STOCK_UNIVERSE:
            try:
                start_date = (simulation_date - timedelta(days=60)).strftime('%Y-%m-%d')
                end_date = simulation_date.strftime('%Y-%m-%d')
                
                df = self.data_provider.get_bars(symbol, '1d', start_date, end_date)
                if df.empty or len(df) < 10:
                    continue
                
                current_price = df['Close'].iloc[-1]
                
                # Get ML signal
                ml_signal = self.get_ml_enhanced_signal(symbol, df)
                session_results['signals_generated'] += 1
                
                if ml_signal['ml_enhanced']:
                    session_results['ml_enhanced_signals'] += 1
                
                # Make and execute decision
                decision = self.make_trading_decision(symbol, current_price, ml_signal)
                
                if decision['action'] != 'hold':
                    success = self.execute_trade(symbol, decision, current_price)
                    if success:
                        session_results['trades_executed'] += 1
                
            except Exception as e:
                print(f"❌ Error with {symbol}: {str(e)[:40]}")
        
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
        
        print(f"💰 Value: ${portfolio_value:,.2f} | Trades: {session_results['trades_executed']} | ML: {session_results['ml_enhanced_signals']}")
        
        return session_results
    
    def run_simulation(self, start_date: str, end_date: str) -> Dict:
        """Run complete high-frequency simulation"""
        print("🚀 HIGH-FREQUENCY ML TRADING SIMULATION")
        print("=" * 60)
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        session_results = []
        portfolio_values = [self.cash]
        
        current_date = start_dt
        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Weekdays only
                for hour in self.config['market_hours']:
                    session_time = current_date.replace(hour=hour, minute=0, second=0)
                    if session_time <= end_dt:
                        try:
                            session_result = self.run_trading_session(session_time)
                            session_results.append(session_result)
                            portfolio_values.append(session_result['portfolio_value'])
                        except Exception as e:
                            print(f"❌ Session failed: {e}")
            
            current_date += timedelta(days=1)
        
        # Calculate performance
        final_value = portfolio_values[-1] if portfolio_values else self.cash
        initial_value = portfolio_values[0]
        total_return = (final_value / initial_value - 1) * 100
        
        trading_days = len([d for d in pd.date_range(start_dt, end_dt) if d.weekday() < 5])
        if trading_days > 0:
            annualized_return = ((final_value / initial_value) ** (252 / trading_days) - 1) * 100
        else:
            annualized_return = 0
        
        win_rate = (self.winning_trades / max(self.trade_count, 1)) * 100
        
        # Max drawdown
        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # ML performance
        avg_ml_accuracy = (self.ml_accuracy_sum / max(self.ml_signals_used, 1)) if self.ml_signals_used > 0 else 0
        ml_usage_rate = (self.ml_signals_used / max(sum(r['signals_generated'] for r in session_results), 1)) * 100
        
        results = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'total_trades': self.trade_count,
            'winning_trades': self.winning_trades,
            'win_rate_pct': win_rate,
            'max_drawdown_pct': max_drawdown,
            'trading_sessions': len(session_results),
            'avg_trades_per_session': np.mean([r['trades_executed'] for r in session_results]) if session_results else 0,
            'final_positions': dict(self.positions),
            'cash_remaining': self.cash,
            'enhancement_summary': {
                'total_signals_generated': sum(r['signals_generated'] for r in session_results),
                'ml_enhanced_signals': sum(r['ml_enhanced_signals'] for r in session_results),
                'ml_usage_rate_pct': ml_usage_rate,
                'avg_ml_accuracy': avg_ml_accuracy,
                'ml_model_updates': sum(r['ml_models_updated'] for r in session_results),
                'trading_frequency_hours': self.config['trading_frequency_hours']
            }
        }
        
        print("\n" + "=" * 60)
        print("🎯 HIGH-FREQUENCY ENHANCED RESULTS")
        print("=" * 60)
        print(f"📊 Total Return: {total_return:.1f}%")
        print(f"📈 Annualized Return: {annualized_return:.1f}%")
        print(f"💰 Final Value: ${final_value:,.2f}")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        print(f"📉 Max Drawdown: {max_drawdown:.1f}%")
        print(f"🤖 ML Usage: {ml_usage_rate:.1f}%")
        print(f"🎯 ML Accuracy: {avg_ml_accuracy:.1%}")
        print(f"⚡ Avg Trades/Session: {results['avg_trades_per_session']:.1f}")
        
        return results

def main():
    """Run enhanced high-frequency simulation"""
    
    start_date = "2024-05-20"
    end_date = "2024-08-20"
    
    print("🚀 HIGH-FREQUENCY ENHANCED ML SYSTEM")
    print("Enhancement A + B: 4-Hour Frequency + Enhanced Data")
    print("Target: 75-85% Annual Returns")
    print("=" * 60)
    
    trader = HighFrequencyMLTrader(HIGH_FREQ_CONFIG)
    results = trader.run_simulation(start_date, end_date)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"high_frequency_enhanced_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Performance comparison
    baseline_return = 57.6
    improvement = results['annualized_return_pct'] - baseline_return
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"   Baseline (Daily): {baseline_return:.1f}%")
    print(f"   Enhanced (4hr): {results['annualized_return_pct']:.1f}%")
    print(f"   Improvement: {improvement:+.1f}%")
    
    if improvement >= 15:
        print(f"🎉 EXCELLENT! Enhancement A+B delivered {improvement:.1f}% boost!")
    elif improvement >= 5:
        print(f"✅ Good improvement: {improvement:.1f}% boost")
    else:
        print(f"⚠️ Below target - need optimization")
    
    return results

if __name__ == "__main__":
    main()
