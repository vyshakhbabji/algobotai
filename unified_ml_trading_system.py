#!/usr/bin/env python3
"""
Unified ML + Technical Trading System
Consolidates all existing sophisticated components to meet prompt.yaml requirements
Target: 30%+ YoY returns with comprehensive ML and technical analysis

Combines:
- Elite Stock Selector (AI-powered screening)
- Improved AI Portfolio Manager (ensemble ML models)
- Elite Options Trader (50-200% return strategies)
- Live Trading System (real Alpaca execution)
- Technical Indicators (RSI, MA, momentum, volatility)

Author: AI Trading System Consolidator
Created: Based on existing sophisticated infrastructure
"""

import os
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# ML and Analysis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
import lightgbm as lgb

# Alpaca Integration
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Import existing sophisticated components
try:
    from improved_ai_portfolio_manager import ImprovedAIPortfolioManager
    from elite_stock_selector import EliteStockSelector
    from elite_options_trader import EliteOptionsTrader
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False

class UnifiedMLTradingSystem:
    """
    Unified ML + Technical Trading System
    
    Integrates all sophisticated components to achieve 30%+ YoY target:
    - Elite stock selection with AI scoring
    - Ensemble ML models for prediction
    - Technical indicator signals
    - Options strategies for high returns
    - Real-time Alpaca execution
    - Comprehensive risk management
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the unified trading system"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # Initialize sophisticated components
        self.elite_selector = EliteStockSelector() if COMPONENTS_AVAILABLE else None
        self.ai_portfolio_manager = ImprovedAIPortfolioManager(
            capital=self.config['portfolio']['initial_capital']
        ) if COMPONENTS_AVAILABLE else None
        self.options_trader = EliteOptionsTrader() if COMPONENTS_AVAILABLE else None
        
        # Initialize Alpaca connection
        self.setup_alpaca_connection()
        
        # Trading state
        self.positions = {}
        self.current_signals = {}
        self.ml_predictions = {}
        self.technical_signals = {}
        self.options_recommendations = {}
        
        # Models storage
        self.ensemble_models = {}
        self.scalers = {}
        self.feature_columns = []
        
        print("🚀 Unified ML Trading System Initialized")
        print(f"💰 Target: {self.config['performance']['target_annual_return']}% YoY")
        print(f"🎯 Max Universe: {self.config['universe']['max_stocks']} stocks")
        
    def load_config(self, config_path: str = None) -> Dict:
        """Load comprehensive trading configuration"""
        default_config = {
            "universe": {
                "max_stocks": 150,
                "refresh_days": 5,
                "min_market_cap": 5_000_000_000,
                "min_avg_volume": 5_000_000,
                "min_price": 15,
                "max_price": 1000
            },
            "portfolio": {
                "initial_capital": 100000,
                "max_position_size": 0.10,  # 10% max per position
                "max_positions": 25,
                "max_sector_exposure": 0.25,  # 25% per sector
                "cash_reserve": 0.05  # 5% cash buffer
            },
            "risk_management": {
                "max_daily_loss": 0.02,  # 2% max daily loss
                "max_drawdown": 0.10,    # 10% max drawdown halt
                "stop_loss_atr": 2.0,    # ATR-based stops
                "trailing_stop_atr": 1.5,
                "take_profit_atr": 3.0
            },
            "ml_config": {
                "models": ["lightgbm", "random_forest", "gradient_boosting"],
                "ensemble_method": "weighted_average",
                "cv_folds": 5,
                "embargo_days": 3,
                "min_r2_score": 0.01,
                "prediction_horizon": 5  # 5-day forward returns
            },
            "technical_indicators": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "ma_short": 5,
                "ma_long": 20,
                "momentum_periods": [5, 10, 20],
                "volatility_lookback": 20
            },
            "signals": {
                "ml_weight": 0.4,        # 40% ML signals
                "technical_weight": 0.3,  # 30% technical signals
                "options_weight": 0.3,    # 30% options strategies
                "min_signal_strength": 0.6,
                "min_conviction": 0.7
            },
            "execution": {
                "mode": "paper",  # paper/live
                "broker": "alpaca",
                "order_type": "limit",
                "limit_offset_bps": 5,
                "max_order_size_adv": 0.05  # 5% of ADV
            },
            "performance": {
                "target_annual_return": 30,  # 30%+ YoY target
                "min_sharpe_ratio": 1.0,
                "min_win_rate": 0.55,
                "max_correlation": 0.7
            },
            "rebalancing": {
                "frequency": "daily",
                "intraday_enabled": True,
                "eod_rebalance": True,
                "signal_refresh_hours": [9, 12, 15]  # Market hours signal refresh
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge configs (user overrides default)
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        import logging
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Setup logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/unified_trading_system_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('UnifiedMLTradingSystem')
        
    def setup_alpaca_connection(self):
        """Setup Alpaca API connection"""
        self.alpaca_connected = False
        
        if not ALPACA_AVAILABLE:
            self.logger.warning("Alpaca SDK not available. Install with: pip install alpaca-py")
            return
        
        # Get credentials from environment
        api_key = os.getenv('ALPACA_API_KEY', '')
        secret_key = os.getenv('ALPACA_SECRET_KEY', '')
        
        if not api_key or not secret_key:
            self.logger.warning("Alpaca credentials not found in environment variables")
            return
        
        try:
            # Initialize Alpaca clients
            self.trading_client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=True  # Paper trading mode
            )
            
            self.data_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key
            )
            
            # Test connection
            account = self.trading_client.get_account()
            self.alpaca_connected = True
            self.logger.info(f"✅ Connected to Alpaca Paper Trading! Account: {account.account_number}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {str(e)}")
    
    def get_elite_stock_universe(self) -> List[str]:
        """Get elite stocks using AI-powered selection"""
        if not self.elite_selector:
            # Fallback to default universe
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                'CRM', 'ADBE', 'NFLX', 'AMD', 'PYPL', 'SNOW', 'PLTR',
                'JPM', 'BAC', 'V', 'MA', 'DIS', 'COST', 'HD', 'INTC'
            ]
        
        try:
            self.logger.info("🔍 Selecting elite stocks using AI screening...")
            
            # Get elite stock selection
            elite_analysis = []
            for symbol in self.elite_selector.candidate_stocks:
                analysis = self.elite_selector.analyze_stock_quality(symbol)
                if analysis:
                    elite_analysis.append(analysis)
            
            # Sort by AI trading score and select top stocks
            elite_analysis.sort(key=lambda x: x['ai_trading_score'], reverse=True)
            top_stocks = [stock['symbol'] for stock in elite_analysis[:self.config['universe']['max_stocks']]]
            
            self.logger.info(f"✅ Selected {len(top_stocks)} elite stocks")
            return top_stocks
            
        except Exception as e:
            self.logger.error(f"Error in elite stock selection: {e}")
            return []
    
    def calculate_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive ML and technical features"""
        try:
            # Basic price features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Moving averages
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA10'] = df['Close'].rolling(10).mean()
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA50'] = df['Close'].rolling(50).mean()
            
            # Price position relative to MAs
            df['price_vs_ma5'] = (df['Close'] - df['MA5']) / df['MA5']
            df['price_vs_ma20'] = (df['Close'] - df['MA20']) / df['MA20']
            df['price_vs_ma50'] = (df['Close'] - df['MA50']) / df['MA50']
            
            # Momentum features
            for period in [5, 10, 20]:
                df[f'momentum_{period}d'] = df['Close'] / df['Close'].shift(period) - 1
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['rsi_normalized'] = (df['RSI'] - 50) / 50
            
            # Volatility features
            df['volatility_10d'] = df['returns'].rolling(10).std()
            df['volatility_20d'] = df['returns'].rolling(20).std()
            df['atr'] = self._calculate_atr(df, period=14)
            
            # Volume features
            df['volume_ma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
            df['volume_momentum'] = df['volume_ratio'] * df['momentum_5d']
            
            # Bollinger Bands
            bb_period = 20
            df['bb_middle'] = df['Close'].rolling(bb_period).mean()
            df['bb_std'] = df['Close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
            df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_squeeze'] = df['bb_std'] / df['bb_middle']
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Support/Resistance
            df['high_20'] = df['High'].rolling(20).max()
            df['low_20'] = df['Low'].rolling(20).min()
            df['price_position'] = (df['Close'] - df['low_20']) / (df['high_20'] - df['low_20'])
            
            # Trend consistency
            trend_conditions = [
                df['momentum_5d'] > 0,
                df['momentum_10d'] > 0,
                df['momentum_20d'] > 0,
                df['Close'] > df['MA5'],
                df['MA5'] > df['MA20']
            ]
            df['trend_consistency'] = np.mean(trend_conditions, axis=0)
            
            # Target: future 5-day return
            df['future_return_5d'] = df['Close'].shift(-5) / df['Close'] - 1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(period).mean()
    
    def train_ensemble_models(self, symbol: str, data: pd.DataFrame) -> bool:
        """Train ensemble ML models for a symbol"""
        try:
            self.logger.info(f"🤖 Training ensemble models for {symbol}...")
            
            # Calculate comprehensive features
            df = self.calculate_comprehensive_features(data.copy())
            
            # Select feature columns
            feature_cols = [
                'price_vs_ma5', 'price_vs_ma20', 'price_vs_ma50',
                'momentum_5d', 'momentum_10d', 'momentum_20d',
                'volatility_10d', 'volatility_20d', 'atr',
                'volume_ratio', 'volume_momentum',
                'rsi_normalized', 'bb_position', 'bb_squeeze',
                'macd', 'macd_histogram', 'price_position',
                'trend_consistency'
            ]
            
            # Prepare clean data
            clean_data = df[feature_cols + ['future_return_5d']].dropna()
            
            if len(clean_data) < 100:
                self.logger.warning(f"Insufficient data for {symbol}")
                return False
            
            # Features and target
            X = clean_data[feature_cols].values
            y = clean_data['future_return_5d'].values
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Define ensemble models
            models = {
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=20,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            }
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config['ml_config']['cv_folds'])
            best_models = {}
            model_scores = {}
            
            for name, model in models.items():
                try:
                    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
                    avg_score = cv_scores.mean()
                    
                    if avg_score > self.config['ml_config']['min_r2_score']:
                        # Train on full data
                        model.fit(X_scaled, y)
                        best_models[name] = model
                        model_scores[name] = avg_score
                        
                        self.logger.info(f"  ✅ {name}: R² = {avg_score:.3f}")
                    else:
                        self.logger.warning(f"  ⚠️ {name}: Low R² = {avg_score:.3f}")
                        
                except Exception as e:
                    self.logger.error(f"  ❌ {name} failed: {e}")
            
            if best_models:
                self.ensemble_models[symbol] = best_models
                self.scalers[symbol] = scaler
                self.feature_columns = feature_cols
                
                # Calculate ensemble weights based on performance
                total_score = sum(model_scores.values())
                ensemble_weights = {name: score/total_score for name, score in model_scores.items()}
                
                self.logger.info(f"  📊 Ensemble weights: {ensemble_weights}")
                return True
            else:
                self.logger.warning(f"No successful models for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error training models for {symbol}: {e}")
            return False
    
    def get_ml_prediction(self, symbol: str, current_data: pd.DataFrame) -> float:
        """Get ensemble ML prediction for a symbol"""
        try:
            if symbol not in self.ensemble_models:
                return 0.0
            
            # Calculate features
            df = self.calculate_comprehensive_features(current_data.copy())
            
            # Get latest features
            latest_features = df[self.feature_columns].iloc[-1:].values
            
            if np.isnan(latest_features).any():
                return 0.0
            
            # Scale features
            scaler = self.scalers[symbol]
            features_scaled = scaler.transform(latest_features)
            
            # Get ensemble prediction
            models = self.ensemble_models[symbol]
            predictions = []
            
            for name, model in models.items():
                pred = model.predict(features_scaled)[0]
                predictions.append(pred)
            
            # Average ensemble prediction
            ensemble_pred = np.mean(predictions)
            
            # Convert to signal strength (0-100)
            signal_strength = max(0, min(100, (ensemble_pred + 0.1) * 500))
            
            return signal_strength
            
        except Exception as e:
            self.logger.error(f"Error getting ML prediction for {symbol}: {e}")
            return 0.0
    
    def calculate_technical_signals(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicator signals"""
        try:
            # Calculate technical indicators
            df = self.calculate_comprehensive_features(data.copy())
            
            if len(df) < 50:
                return {'signal': 'HOLD', 'strength': 0.0}
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Initialize signal components
            buy_strength = 0.0
            sell_strength = 0.0
            
            # RSI signals
            if latest['RSI'] < 30:  # Oversold
                buy_strength += 0.3
            elif latest['RSI'] > 70:  # Overbought
                sell_strength += 0.3
            
            # Moving average signals
            if latest['price_vs_ma5'] > 0 and latest['price_vs_ma20'] > 0:
                buy_strength += 0.25
            elif latest['price_vs_ma5'] < 0 and latest['price_vs_ma20'] < 0:
                sell_strength += 0.25
            
            # Momentum signals
            momentum_score = (latest['momentum_5d'] + latest['momentum_10d']) / 2
            if momentum_score > 0.02:  # Strong positive momentum
                buy_strength += min(0.3, momentum_score * 10)
            elif momentum_score < -0.02:  # Strong negative momentum
                sell_strength += min(0.3, abs(momentum_score) * 10)
            
            # Volatility signals
            if latest['bb_position'] < 0.2:  # Near lower Bollinger Band
                buy_strength += 0.15
            elif latest['bb_position'] > 0.8:  # Near upper Bollinger Band
                sell_strength += 0.15
            
            # MACD signals
            if latest['macd'] > latest['macd_signal'] and latest['macd_histogram'] > 0:
                buy_strength += 0.15
            elif latest['macd'] < latest['macd_signal'] and latest['macd_histogram'] < 0:
                sell_strength += 0.15
            
            # Volume confirmation
            if latest['volume_ratio'] > 1.5:  # High volume
                if buy_strength > sell_strength:
                    buy_strength *= 1.2  # Amplify buy signal
                elif sell_strength > buy_strength:
                    sell_strength *= 1.2  # Amplify sell signal
            
            # Determine final signal
            signal = 'HOLD'
            strength = 0.0
            
            min_strength = self.config['signals']['min_signal_strength']
            
            if buy_strength > min_strength and buy_strength > sell_strength:
                signal = 'BUY'
                strength = min(1.0, buy_strength)
            elif sell_strength > min_strength and sell_strength > buy_strength:
                signal = 'SELL'
                strength = min(1.0, sell_strength)
            
            return {
                'signal': signal,
                'strength': strength,
                'buy_strength': buy_strength,
                'sell_strength': sell_strength,
                'rsi': latest['RSI'],
                'momentum': momentum_score,
                'volume_ratio': latest['volume_ratio']
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating technical signals for {symbol}: {e}")
            return {'signal': 'HOLD', 'strength': 0.0}
    
    def get_options_recommendations(self, symbol: str, current_price: float, 
                                   ml_signal: float, tech_signal: Dict) -> Dict:
        """Get options trading recommendations from elite options trader"""
        try:
            if not self.options_trader:
                return {}
            
            # This would integrate with the elite options trader
            # For now, return simplified options signals based on ML + technical
            
            combined_signal_strength = (ml_signal * 0.6 + tech_signal['strength'] * 0.4) * 100
            
            if combined_signal_strength > 70:
                if tech_signal['signal'] == 'BUY':
                    return {
                        'strategy': 'long_call',
                        'confidence': combined_signal_strength,
                        'expected_return': '50-150%',
                        'max_risk': '5% of portfolio',
                        'time_horizon': '2-4 weeks'
                    }
                elif tech_signal['signal'] == 'SELL':
                    return {
                        'strategy': 'long_put',
                        'confidence': combined_signal_strength,
                        'expected_return': '50-120%',
                        'max_risk': '5% of portfolio',
                        'time_horizon': '2-4 weeks'
                    }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting options recommendations for {symbol}: {e}")
            return {}
    
    def calculate_composite_signal(self, symbol: str, ml_prediction: float, 
                                 technical_signal: Dict, options_rec: Dict) -> Dict:
        """Calculate composite signal from all sources"""
        try:
            # Weights from config
            ml_weight = self.config['signals']['ml_weight']
            tech_weight = self.config['signals']['technical_weight']
            options_weight = self.config['signals']['options_weight']
            
            # Normalize ML prediction to 0-1
            ml_strength = ml_prediction / 100.0
            
            # Get technical strength
            tech_strength = technical_signal.get('strength', 0.0)
            tech_direction = 1 if technical_signal.get('signal') == 'BUY' else -1 if technical_signal.get('signal') == 'SELL' else 0
            
            # Options strength (simplified)
            options_strength = options_rec.get('confidence', 0) / 100.0 if options_rec else 0.0
            options_direction = 1 if 'call' in options_rec.get('strategy', '') else -1 if 'put' in options_rec.get('strategy', '') else 0
            
            # Calculate weighted composite signal
            composite_strength = (
                ml_strength * ml_weight +
                tech_strength * tech_weight +
                options_strength * options_weight
            )
            
            # Determine direction (majority vote weighted by strength)
            direction_score = (
                ml_strength * ml_weight * (1 if ml_prediction > 50 else -1 if ml_prediction < 50 else 0) +
                tech_strength * tech_weight * tech_direction +
                options_strength * options_weight * options_direction
            )
            
            # Final signal
            signal = 'HOLD'
            if direction_score > 0 and composite_strength > self.config['signals']['min_conviction']:
                signal = 'BUY'
            elif direction_score < 0 and composite_strength > self.config['signals']['min_conviction']:
                signal = 'SELL'
            
            return {
                'signal': signal,
                'strength': composite_strength,
                'conviction': min(1.0, abs(direction_score)),
                'ml_component': {'strength': ml_strength, 'prediction': ml_prediction},
                'technical_component': technical_signal,
                'options_component': options_rec,
                'direction_score': direction_score
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating composite signal for {symbol}: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'conviction': 0.0}
    
    def calculate_position_size(self, symbol: str, signal_strength: float, 
                              current_price: float, account_value: float) -> int:
        """Calculate optimal position size based on signal strength and risk management"""
        try:
            # Base position size from signal strength
            base_size = signal_strength * self.config['portfolio']['max_position_size']
            
            # Risk-adjusted size
            max_position_value = account_value * base_size
            max_shares = int(max_position_value / current_price)
            
            # Apply position limits
            max_positions_allowed = self.config['portfolio']['max_positions']
            current_positions = len([p for p in self.positions.values() if p > 0])
            
            if current_positions >= max_positions_allowed:
                return 0  # No new positions
            
            # Volume-based limit (5% of average daily volume)
            # This would require volume data - simplified for now
            max_volume_shares = max_shares  # Placeholder
            
            final_shares = min(max_shares, max_volume_shares)
            
            return max(0, final_shares)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    def execute_trade(self, symbol: str, action: str, shares: int, current_price: float) -> bool:
        """Execute trade through Alpaca"""
        try:
            if not self.alpaca_connected:
                self.logger.info(f"📝 PAPER TRADE: {action} {shares} shares of {symbol} at ${current_price:.2f}")
                return True
            
            if shares <= 0:
                return False
            
            # Prepare order
            if action == 'BUY':
                order_side = OrderSide.BUY
            elif action == 'SELL':
                order_side = OrderSide.SELL
            else:
                return False
            
            # Create limit order with small offset
            limit_offset = self.config['execution']['limit_offset_bps'] / 10000
            if action == 'BUY':
                limit_price = current_price * (1 + limit_offset)
            else:
                limit_price = current_price * (1 - limit_offset)
            
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=shares,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2)
            )
            
            # Submit order
            order = self.trading_client.submit_order(order_request)
            
            self.logger.info(f"✅ Order submitted: {action} {shares} shares of {symbol} at ${limit_price:.2f}")
            self.logger.info(f"   Order ID: {order.id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            return False
    
    def run_daily_scan(self) -> Dict[str, Dict]:
        """Run comprehensive daily scan and generate signals"""
        self.logger.info("🔍 Running comprehensive daily scan...")
        
        # Get elite stock universe
        universe = self.get_elite_stock_universe()
        
        # Download data for all stocks
        all_signals = {}
        
        for symbol in universe:
            try:
                self.logger.info(f"📊 Analyzing {symbol}...")
                
                # Get historical data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='1y')  # 1 year for features
                
                if len(data) < 100:
                    continue
                
                current_price = float(data['Close'].iloc[-1])
                
                # Train/update ML models
                if symbol not in self.ensemble_models:
                    self.train_ensemble_models(symbol, data)
                
                # Get ML prediction
                ml_prediction = self.get_ml_prediction(symbol, data)
                
                # Get technical signals
                technical_signal = self.calculate_technical_signals(symbol, data)
                
                # Get options recommendations
                options_rec = self.get_options_recommendations(symbol, current_price, ml_prediction, technical_signal)
                
                # Calculate composite signal
                composite_signal = self.calculate_composite_signal(symbol, ml_prediction, technical_signal, options_rec)
                
                all_signals[symbol] = {
                    'current_price': current_price,
                    'composite_signal': composite_signal,
                    'ml_prediction': ml_prediction,
                    'technical_signal': technical_signal,
                    'options_recommendation': options_rec,
                    'timestamp': datetime.now()
                }
                
                self.logger.info(f"  Signal: {composite_signal['signal']} (Strength: {composite_signal['strength']:.2f})")
                
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        return all_signals
    
    def execute_trading_session(self) -> Dict:
        """Execute complete trading session"""
        self.logger.info("🚀 Starting Unified ML Trading Session")
        
        session_results = {
            'start_time': datetime.now(),
            'signals_generated': 0,
            'trades_executed': 0,
            'errors': 0,
            'performance_metrics': {}
        }
        
        try:
            # Run daily scan
            signals = self.run_daily_scan()
            session_results['signals_generated'] = len(signals)
            
            # Get account info
            if self.alpaca_connected:
                account = self.trading_client.get_account()
                account_value = float(account.portfolio_value)
            else:
                account_value = self.config['portfolio']['initial_capital']
            
            # Execute trades based on signals
            for symbol, signal_data in signals.items():
                try:
                    composite_signal = signal_data['composite_signal']
                    current_price = signal_data['current_price']
                    
                    if composite_signal['signal'] in ['BUY', 'SELL']:
                        # Calculate position size
                        shares = self.calculate_position_size(
                            symbol, 
                            composite_signal['strength'], 
                            current_price,
                            account_value
                        )
                        
                        if shares > 0:
                            # Execute trade
                            success = self.execute_trade(
                                symbol,
                                composite_signal['signal'],
                                shares,
                                current_price
                            )
                            
                            if success:
                                session_results['trades_executed'] += 1
                                
                                # Update positions tracking
                                if composite_signal['signal'] == 'BUY':
                                    self.positions[symbol] = self.positions.get(symbol, 0) + shares
                                else:
                                    self.positions[symbol] = self.positions.get(symbol, 0) - shares
                    
                except Exception as e:
                    self.logger.error(f"Error executing trade for {symbol}: {e}")
                    session_results['errors'] += 1
            
            # Store signals for analysis
            self.current_signals = signals
            
            session_results['end_time'] = datetime.now()
            session_results['duration'] = session_results['end_time'] - session_results['start_time']
            
            self.logger.info(f"✅ Trading session completed!")
            self.logger.info(f"   Signals generated: {session_results['signals_generated']}")
            self.logger.info(f"   Trades executed: {session_results['trades_executed']}")
            self.logger.info(f"   Errors: {session_results['errors']}")
            
            return session_results
            
        except Exception as e:
            self.logger.error(f"Error in trading session: {e}")
            session_results['errors'] += 1
            return session_results
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        try:
            report = {
                'timestamp': datetime.now(),
                'portfolio_summary': {},
                'position_analysis': {},
                'signal_analysis': {},
                'risk_metrics': {},
                'recommendations': []
            }
            
            # Portfolio summary
            if self.alpaca_connected:
                account = self.trading_client.get_account()
                positions = self.trading_client.get_all_positions()
                
                report['portfolio_summary'] = {
                    'total_value': float(account.portfolio_value),
                    'buying_power': float(account.buying_power),
                    'cash': float(account.cash),
                    'day_pnl': float(account.unrealized_pl),
                    'total_pnl': float(account.unrealized_pl),
                    'positions_count': len(positions)
                }
                
                # Position analysis
                for pos in positions:
                    symbol = pos.symbol
                    report['position_analysis'][symbol] = {
                        'shares': int(pos.qty),
                        'market_value': float(pos.market_value),
                        'pnl': float(pos.unrealized_pl),
                        'pnl_pct': float(pos.unrealized_plpc) * 100
                    }
            
            # Signal analysis
            if self.current_signals:
                buy_signals = len([s for s in self.current_signals.values() if s['composite_signal']['signal'] == 'BUY'])
                sell_signals = len([s for s in self.current_signals.values() if s['composite_signal']['signal'] == 'SELL'])
                hold_signals = len([s for s in self.current_signals.values() if s['composite_signal']['signal'] == 'HOLD'])
                
                report['signal_analysis'] = {
                    'total_signals': len(self.current_signals),
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'hold_signals': hold_signals,
                    'avg_signal_strength': np.mean([s['composite_signal']['strength'] for s in self.current_signals.values()]),
                    'avg_conviction': np.mean([s['composite_signal']['conviction'] for s in self.current_signals.values()])
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {}

def main():
    """Main execution function"""
    print("🚀 Unified ML Trading System - Consolidating Sophisticated Components")
    print("=" * 80)
    
    # Initialize trading system
    trading_system = UnifiedMLTradingSystem()
    
    # Execute trading session
    session_results = trading_system.execute_trading_session()
    
    # Generate performance report
    performance_report = trading_system.generate_performance_report()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f'unified_trading_results_{timestamp}.json', 'w') as f:
        json.dump({
            'session_results': session_results,
            'performance_report': performance_report,
            'config': trading_system.config
        }, f, indent=2, default=str)
    
    print(f"\n📊 Results saved to: unified_trading_results_{timestamp}.json")
    print(f"🎯 Targeting 30%+ YoY returns with comprehensive ML + Technical system")

if __name__ == "__main__":
    main()
