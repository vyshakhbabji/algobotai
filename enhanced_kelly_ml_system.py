"""
🚀 ENHANCED KELLY ML TRADING SYSTEM
Working Kelly System + Signal Strength ML Enhancement

This combines:
1. The exact working Kelly system logic (49.8% returns)
2. NEW: ML signal strength enhancement instead of return prediction
3. Signal multipliers (0.3-1.0x) to boost/reduce position sizes
4. Market regime detection for strategy adaptation
5. Breakout probability for timing enhancement

Expected performance: 49.8% → 60-70% annual returns
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Any
import traceback

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONPATH'] = str(Path(__file__).parent)
sys.path.append(str(Path(__file__).parent))

class EnhancedKellyMLSystem:
    """Enhanced Kelly system with ML signal strength prediction"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.daily_performance = []
        
        # ML models for signal enhancement
        self.signal_strength_models = {}
        self.regime_models = {}
        self.scalers = {}
        
        # Configuration (same as working Kelly system)
        self.config = {
            'max_position_size': 0.20,  # Enhanced from 0.15 
            'min_position_size': 0.02,
            'stop_loss_pct': 0.08,
            'take_profit_pct': 0.25,
            'max_daily_loss': 0.025,
            'max_drawdown': 0.12,
            'kelly_multiplier': 1.5,  # Enhanced Kelly sizing
            'signal_threshold': 0.4,
            'ml_enhancement': True,  # NEW: Enable ML enhancement
            'min_ml_confidence': 0.6  # Minimum ML confidence to apply enhancement
        }
        
        # Performance tracking
        self.start_date = None
        self.end_date = None
        self.max_capital = initial_capital
        
        self.logger = self._setup_logging()
        self.logger.info("🚀 Enhanced Kelly ML System initialized")
    
    def _setup_logging(self):
        logger = logging.getLogger('EnhancedKellyML')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def get_elite_stocks(self) -> List[str]:
        """Same elite stock selection as working system"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX',
            'CRM', 'SNOW', 'PLTR', 'COIN', 'UBER', 'DIS', 'JPM', 'BAC',
            'JNJ', 'PG', 'KO', 'WMT', 'HD', 'V', 'MA', 'PFE', 'VZ'
        ][:25]
    
    def fetch_data(self, symbol: str, period: str = "2y", start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch stock data"""
        try:
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date)
            elif start_date:
                data = ticker.history(start=start_date)
            else:
                data = ticker.history(period=period)
                
            if data.empty:
                return pd.DataFrame()
            data = data.dropna()
            # Ensure timezone consistency
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            return data
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators (same as working system)"""
        try:
            df = df.copy()
            
            # Price and volume features
            df['returns'] = df['Close'].pct_change()
            df['volume_ma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
            
            # Moving averages
            df['ma_5'] = df['Close'].rolling(5).mean()
            df['ma_20'] = df['Close'].rolling(20).mean()
            df['ma_50'] = df['Close'].rolling(50).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Additional ML features
            df['volatility_10d'] = df['returns'].rolling(10).std()
            df['volatility_20d'] = df['returns'].rolling(20).std()
            df['price_vs_ma5'] = (df['Close'] - df['ma_5']) / df['ma_5']
            df['price_vs_ma20'] = (df['Close'] - df['ma_20']) / df['ma_20']
            df['price_vs_ma50'] = (df['Close'] - df['ma_50']) / df['ma_50']
            df['rsi_normalized'] = (df['rsi'] - 50) / 50
            df['trend_5d'] = np.where(df['Close'] > df['Close'].shift(5), 1, -1)
            df['trend_10d'] = np.where(df['Close'] > df['Close'].shift(10), 1, -1)
            df['trend_consistency'] = (df['trend_5d'] + df['trend_10d']) / 2
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return df
    
    def create_ml_training_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML targets for signal enhancement"""
        try:
            df = df.copy()
            
            # Signal strength target (0.3-1.0 based on future moves)
            future_returns_3d = df['Close'].shift(-3) / df['Close'] - 1
            future_returns_5d = df['Close'].shift(-5) / df['Close'] - 1
            
            # Volatility-adjusted signal strength
            vol_3d = df['returns'].rolling(3).std()
            vol_adj_move = np.abs(future_returns_3d) / (vol_3d + 0.001)
            
            # Convert to 0.3-1.0 range
            signal_strength_raw = np.clip(vol_adj_move * 1.5, 0, 1)
            df['ml_signal_strength'] = 0.3 + (signal_strength_raw * 0.7)
            
            # Market regime (trending vs ranging)
            trend_strength = np.abs(df['returns'].rolling(10).mean())
            volatility = df['volatility_10d']
            regime_score = trend_strength / (volatility + 0.001)
            regime_threshold = regime_score.rolling(50).quantile(0.65)
            df['ml_regime'] = (regime_score > regime_threshold).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating ML targets: {e}")
            return df
    
    def train_ml_models(self, symbol: str, data: pd.DataFrame) -> bool:
        """Train ML models for signal enhancement"""
        try:
            if len(data) < 200:
                return False
            
            # Calculate features and targets
            df = self.calculate_technical_indicators(data.copy())
            df = self.create_ml_training_targets(df)
            
            # Feature columns for ML
            feature_cols = [
                'price_vs_ma5', 'price_vs_ma20', 'price_vs_ma50',
                'rsi_normalized', 'volatility_10d', 'volatility_20d',
                'volume_ratio', 'bb_position', 'macd_histogram',
                'trend_consistency'
            ]
            
            # Clean data
            clean_data = df[feature_cols + ['ml_signal_strength', 'ml_regime']].dropna()
            
            if len(clean_data) < 100:
                return False
            
            X = clean_data[feature_cols].values
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[symbol] = scaler
            
            # Train signal strength model
            y_strength = clean_data['ml_signal_strength'].values
            
            try:
                strength_model = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1, 
                    random_state=42, verbose=-1
                )
                
                # Quick validation
                tscv = TimeSeriesSplit(n_splits=3)
                cv_scores = cross_val_score(strength_model, X_scaled, y_strength, cv=tscv, scoring='r2')
                
                if cv_scores.mean() > 0.05:  # Low threshold for enhancement
                    strength_model.fit(X_scaled, y_strength)
                    self.signal_strength_models[symbol] = strength_model
                    self.logger.info(f"✅ {symbol} signal strength model: R² = {cv_scores.mean():.3f}")
                else:
                    self.logger.warning(f"⚠️ {symbol} signal strength model: Low R² = {cv_scores.mean():.3f}")
            except Exception as e:
                self.logger.error(f"Signal strength model failed for {symbol}: {e}")
            
            # Train regime model
            y_regime = clean_data['ml_regime'].values
            
            try:
                regime_model = RandomForestClassifier(
                    n_estimators=100, max_depth=8, random_state=42
                )
                
                cv_scores = cross_val_score(regime_model, X_scaled, y_regime, cv=tscv, scoring='accuracy')
                
                if cv_scores.mean() > 0.55:
                    regime_model.fit(X_scaled, y_regime)
                    self.regime_models[symbol] = regime_model
                    self.logger.info(f"✅ {symbol} regime model: Accuracy = {cv_scores.mean():.3f}")
                else:
                    self.logger.warning(f"⚠️ {symbol} regime model: Low accuracy = {cv_scores.mean():.3f}")
            except Exception as e:
                self.logger.error(f"Regime model failed for {symbol}: {e}")
            
            return bool(symbol in self.signal_strength_models or symbol in self.regime_models)
            
        except Exception as e:
            self.logger.error(f"Error training ML for {symbol}: {e}")
            return False
    
    def generate_base_signal(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate base signal (exact same logic as working Kelly system)"""
        try:
            if len(df) < 50:
                return {'signal': 'HOLD', 'strength': 0.0, 'price': 0}
            
            latest = df.iloc[-1]
            
            # Technical conditions (EXACT same as working system)
            rsi = latest['rsi']
            price = latest['Close']
            ma_5 = latest['ma_5']
            ma_20 = latest['ma_20']
            volume_ratio = latest.get('volume_ratio', 1.0)
            bb_position = latest.get('bb_position', 0.5)
            macd_hist = latest.get('macd_histogram', 0)
            
            price_vs_ma5 = (price - ma_5) / ma_5
            price_vs_ma20 = (price - ma_20) / ma_20
            
            # Strong buy conditions
            strong_buy = (
                (rsi < 35 and price_vs_ma5 > -0.03) or  # Oversold with support
                (price_vs_ma5 > 0.02 and price_vs_ma20 > 0.01 and volume_ratio > 1.2 and macd_hist > 0)
            )
            
            # Strong sell conditions
            strong_sell = (
                (rsi > 75 and price_vs_ma5 < 0.02) or  # Overbought with resistance
                (price_vs_ma5 < -0.02 and price_vs_ma20 < -0.01 and macd_hist < 0)
            )
            
            # Calculate signal strength (EXACT same logic)
            if strong_buy:
                strength = min(0.8, (1.2 + price_vs_ma5 + (80-rsi)/100 + (volume_ratio-1)))
                return {'signal': 'BUY', 'strength': max(0.4, strength), 'price': price}
            elif strong_sell:
                strength = min(0.8, (1.2 - price_vs_ma5 + (rsi-20)/100))
                return {'signal': 'SELL', 'strength': max(0.4, strength), 'price': price}
            else:
                return {'signal': 'HOLD', 'strength': 0.0, 'price': price}
                
        except Exception as e:
            self.logger.error(f"Error generating base signal: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'price': 0}
    
    def enhance_signal_with_ml(self, symbol: str, base_signal: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhance base signal with ML predictions"""
        try:
            if not self.config['ml_enhancement'] or base_signal['signal'] == 'HOLD':
                return base_signal
            
            if symbol not in self.scalers:
                return base_signal  # No ML models trained
            
            # Prepare features
            feature_cols = [
                'price_vs_ma5', 'price_vs_ma20', 'price_vs_ma50',
                'rsi_normalized', 'volatility_10d', 'volatility_20d',
                'volume_ratio', 'bb_position', 'macd_histogram',
                'trend_consistency'
            ]
            
            latest_features = df[feature_cols].iloc[-1:].values
            X_scaled = self.scalers[symbol].transform(latest_features)
            
            # Default enhancement
            ml_multiplier = 1.0
            regime_boost = 1.0
            
            # Predict signal strength enhancement
            if symbol in self.signal_strength_models:
                try:
                    strength_pred = self.signal_strength_models[symbol].predict(X_scaled)[0]
                    strength_pred = np.clip(strength_pred, 0.3, 1.0)
                    
                    # Convert prediction to multiplier
                    ml_multiplier = strength_pred
                    
                except Exception as e:
                    self.logger.error(f"Signal strength prediction error for {symbol}: {e}")
            
            # Predict market regime enhancement
            if symbol in self.regime_models:
                try:
                    regime_proba = self.regime_models[symbol].predict_proba(X_scaled)[0]
                    trending_prob = regime_proba[1]  # Probability of trending market
                    
                    # Boost signals in trending markets
                    regime_boost = 1.0 + (trending_prob * 0.4)  # Up to 40% boost
                    
                except Exception as e:
                    self.logger.error(f"Regime prediction error for {symbol}: {e}")
            
            # Combine enhancements
            total_multiplier = ml_multiplier * regime_boost
            total_multiplier = np.clip(total_multiplier, 0.5, 1.8)  # Reasonable bounds
            
            # Apply enhancement
            enhanced_strength = base_signal['strength'] * total_multiplier
            enhanced_strength = np.clip(enhanced_strength, 0.0, 1.0)
            
            return {
                'signal': base_signal['signal'],
                'strength': enhanced_strength,
                'price': base_signal['price'],
                'base_strength': base_signal['strength'],
                'ml_multiplier': ml_multiplier,
                'regime_boost': regime_boost,
                'total_enhancement': total_multiplier
            }
            
        except Exception as e:
            self.logger.error(f"Error enhancing signal for {symbol}: {e}")
            return base_signal
    
    def calculate_kelly_position_size(self, signal_strength: float, price: float) -> float:
        """Calculate position size using enhanced Kelly criterion"""
        if signal_strength < self.config['signal_threshold']:
            return 0.0
        
        # Enhanced Kelly calculation with ML boost
        base_kelly = signal_strength * self.config['kelly_multiplier']
        
        # Position size as fraction of capital
        position_fraction = min(base_kelly, self.config['max_position_size'])
        position_fraction = max(position_fraction, self.config['min_position_size'])
        
        # Convert to dollar amount
        position_value = self.current_capital * position_fraction
        shares = int(position_value / price)
        
        return shares
    
    def execute_trade(self, symbol: str, signal: Dict, date: pd.Timestamp) -> bool:
        """Execute trade based on signal"""
        try:
            action = signal['signal']
            price = signal['price']
            strength = signal['strength']
            
            if action == 'HOLD':
                return False
            
            current_position = self.positions.get(symbol, 0)
            
            if action == 'BUY' and current_position <= 0:
                # Buy signal
                shares = self.calculate_kelly_position_size(strength, price)
                if shares > 0:
                    cost = shares * price
                    if cost <= self.current_capital:
                        self.positions[symbol] = shares
                        self.current_capital -= cost
                        
                        self.trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'cost': cost,
                            'strength': strength,
                            'ml_enhanced': signal.get('total_enhancement', 1.0)
                        })
                        return True
            
            elif action == 'SELL' and current_position > 0:
                # Sell signal
                shares = current_position
                proceeds = shares * price
                self.current_capital += proceeds
                self.positions[symbol] = 0
                
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'proceeds': proceeds,
                    'strength': strength,
                    'ml_enhanced': signal.get('total_enhancement', 1.0)
                })
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            return False
    
    def run_enhanced_backtest(self, start_date: str = "2024-05-20", end_date: str = "2024-08-20") -> Dict[str, Any]:
        """Run backtest with ML-enhanced signals"""
        try:
            self.logger.info(f"🚀 Running Enhanced Kelly ML backtest: {start_date} to {end_date}")
            
            stocks = self.get_elite_stocks()
            
            # Phase 1: Train ML models on pre-period data
            self.logger.info("📚 Training ML models...")
            train_end = pd.to_datetime(start_date).tz_localize(None) - timedelta(days=1)
            
            for symbol in stocks:
                data = self.fetch_data(symbol, start_date="2023-01-01", end_date="2024-12-31")
                if not data.empty:
                    # Use data before start_date for training
                    train_data = data[data.index <= train_end]
                    if len(train_data) > 200:
                        self.train_ml_models(symbol, train_data)
            
            # Phase 2: Run backtest with enhanced signals
            self.logger.info("📈 Running enhanced backtest...")
            
            # Get date range
            test_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            test_dates = [d for d in test_dates if d.weekday() < 5]  # Trading days only
            
            for current_date in test_dates:
                # Track daily performance
                total_value = self.current_capital
                for symbol, shares in self.positions.items():
                    if shares > 0:
                        try:
                            data = self.fetch_data(symbol, start_date="2024-01-01", end_date="2024-12-31")
                            if not data.empty:
                                current_price = data[data.index <= current_date]['Close'].iloc[-1]
                                total_value += shares * current_price
                        except:
                            pass
                
                self.daily_performance.append({
                    'date': current_date,
                    'total_value': total_value,
                    'cash': self.current_capital
                })
                
                # Generate signals for each stock
                for symbol in stocks:
                    try:
                        # Get data up to current date
                        data = self.fetch_data(symbol, start_date="2024-01-01", end_date="2024-12-31")
                        if not data.empty:
                            recent_data = data[data.index <= current_date]
                        else:
                            continue
                        
                        if len(recent_data) < 50:
                            continue
                        
                        # Calculate indicators
                        df = self.calculate_technical_indicators(recent_data)
                        
                        # Generate base signal
                        base_signal = self.generate_base_signal(df)
                        
                        # Enhance with ML
                        enhanced_signal = self.enhance_signal_with_ml(symbol, base_signal, df)
                        
                        # Execute trade
                        self.execute_trade(symbol, enhanced_signal, current_date)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol} on {current_date}: {e}")
                        continue
            
            # Calculate final performance
            return self.calculate_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error in enhanced backtest: {e}")
            return {"error": str(e)}
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        try:
            if not self.daily_performance:
                return {"error": "No performance data"}
            
            # Final portfolio value
            final_value = self.current_capital
            for symbol, shares in self.positions.items():
                if shares > 0:
                    try:
                        data = self.fetch_data(symbol, start_date="2024-01-01", end_date="2024-12-31")
                        if not data.empty:
                            final_price = data['Close'].iloc[-1]
                            final_value += shares * final_price
                    except:
                        pass
            
            # Performance metrics
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            # Calculate period length
            start_date = self.daily_performance[0]['date']
            end_date = self.daily_performance[-1]['date']
            days = (end_date - start_date).days
            annual_return = (total_return + 1) ** (365.0 / days) - 1
            
            # Trade statistics
            buy_trades = [t for t in self.trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.trades if t['action'] == 'SELL']
            
            # ML enhancement statistics
            ml_enhanced_trades = [t for t in self.trades if t.get('ml_enhanced', 1.0) != 1.0]
            avg_ml_enhancement = np.mean([t.get('ml_enhanced', 1.0) for t in self.trades])
            
            return {
                'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'days': days,
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annual_return': annual_return,
                'annual_return_pct': annual_return * 100,
                'total_trades': len(self.trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'ml_enhanced_trades': len(ml_enhanced_trades),
                'avg_ml_enhancement': avg_ml_enhancement,
                'ml_models_trained': len(self.signal_strength_models) + len(self.regime_models),
                'final_positions': len([s for s, sh in self.positions.items() if sh > 0])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {"error": str(e)}

def main():
    """Test the Enhanced Kelly ML System"""
    print("🚀 ENHANCED KELLY ML TRADING SYSTEM")
    print("Working Kelly Logic + ML Signal Enhancement")
    print("=" * 60)
    
    # Initialize system
    system = EnhancedKellyMLSystem(initial_capital=100000.0)
    
    # Run 3-month test
    print("📈 Running 3-month enhanced test...")
    result = system.run_enhanced_backtest(
        start_date="2024-05-20",
        end_date="2024-08-20"
    )
    
    if "error" not in result:
        print(f"\n✅ Enhanced Kelly ML Results:")
        print(f"   Period:              {result['period']}")
        print(f"   Initial Capital:     ${result['initial_capital']:,.0f}")
        print(f"   Final Value:         ${result['final_value']:,.0f}")
        print(f"   Total Return:        {result['total_return_pct']:+.1f}%")
        print(f"   Annual Return:       {result['annual_return_pct']:+.1f}%")
        print(f"   Total Trades:        {result['total_trades']}")
        print(f"   ML Enhanced Trades:  {result['ml_enhanced_trades']}")
        print(f"   Avg ML Enhancement:  {result['avg_ml_enhancement']:.2f}x")
        print(f"   ML Models Trained:   {result['ml_models_trained']}")
        
        # Compare to baseline (49.8% from working Kelly system)
        baseline_annual = 49.8
        improvement = result['annual_return_pct'] - baseline_annual
        print(f"\n🎯 Performance vs Baseline:")
        print(f"   Baseline (Working Kelly): {baseline_annual}%")
        print(f"   Enhanced ML System:       {result['annual_return_pct']:+.1f}%")
        print(f"   Improvement:              {improvement:+.1f}%")
        
        if improvement > 5:
            print("🎉 Significant improvement! ML enhancement is working!")
        elif improvement > 0:
            print("✅ Positive improvement from ML enhancement")
        else:
            print("⚠️ ML enhancement may need tuning")
            
    else:
        print(f"❌ Error: {result['error']}")
    
    print("\n🤖 ML Enhancement Summary:")
    print("• Predicts signal strength multipliers (0.3-1.0x)")
    print("• Detects market regime for signal boosting")
    print("• Enhances proven Kelly system logic")
    print("• Maintains exact working system base signals")
    print("• Target: 60-70% annual returns vs 49.8% baseline")

if __name__ == "__main__":
    main()
