#!/usr/bin/env python3
"""
🎯 ALPACA COMPREHENSIVE BACKTESTER
2-Year Training + 3-Month Forward Test with NO DATA LEAKAGE

Features:
- 2 years of Alpaca data for training (Jan 2023 - Jan 2025)
- 3-month forward test period (Feb 2025 - May 2025) - UNSEEN DATA
- 100-stock universe with AI screening
- ML model training with validation
- Full trading simulation with position management
- Comprehensive performance analysis

Author: AI Trading System
Target: 30%+ Annual Returns with Alpaca Data
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
from typing import Dict, List, Tuple, Optional
import traceback

# Alpaca imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ML imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

class AlpacaComprehensiveBacktester:
    """
    Comprehensive backtesting system with strict data isolation
    """
    
    def __init__(self, config_file: str = "alpaca_config.json"):
        """Initialize with Alpaca configuration"""
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Setup Alpaca data client
        self.data_client = StockHistoricalDataClient(
            api_key=self.config['alpaca']['api_key'],
            secret_key=self.config['alpaca']['secret_key']
        )
        
        # Date boundaries - STRICT NO LEAKAGE
        self.training_start = "2023-06-01"  # 2 years of training data
        self.training_end = "2025-05-31"    # End of training period  
        self.forward_start = "2025-06-01"   # Start of forward test (UNSEEN)
        self.forward_end = "2025-08-21"     # End of forward test (3 months)
        
        # Trading configuration
        self.initial_capital = 100000  # $100k starting capital
        self.max_positions = 12        # Maximum concurrent positions
        self.position_size_pct = 0.08  # 8% per position (allows diversification)
        self.min_signal_strength = 0.35 # Minimum signal to trade
        
        # Risk management
        self.stop_loss_pct = 0.08      # 8% stop loss
        self.take_profit_pct = 0.20    # 20% take profit
        self.max_correlation = 0.7     # Max correlation between positions
        
        # Results tracking
        self.universe_data = {}
        self.trained_models = {}
        self.trades = []
        self.portfolio_history = []
        self.performance_metrics = {}
        
        print("🚀 ALPACA COMPREHENSIVE BACKTESTER INITIALIZED")
        print(f"📅 Training Period: {self.training_start} to {self.training_end}")
        print(f"🔮 Forward Test: {self.forward_start} to {self.forward_end} (UNSEEN)")
        print(f"💰 Initial Capital: ${self.initial_capital:,}")
    
    def get_comprehensive_universe(self) -> List[str]:
        """Get comprehensive 100-stock universe"""
        
        # Tech giants
        tech_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
            'NFLX', 'ADBE', 'CRM', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM',
            'AVGO', 'TXN', 'MU', 'LRCX', 'AMAT', 'KLAC', 'MRVL', 'SNPS'
        ]
        
        # Financial sector
        financial_stocks = [
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC',
            'BK', 'AXP', 'COF', 'DFS', 'SCHW', 'BLK', 'SPGI'
        ]
        
        # Healthcare & biotech
        healthcare_stocks = [
            'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'DHR', 'ABT', 'LLY',
            'BMY', 'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ISRG'
        ]
        
        # Consumer & retail
        consumer_stocks = [
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'SBUX',
            'MCD', 'DIS', 'CMCSA', 'VZ', 'T', 'PM', 'MO'
        ]
        
        # Growth & momentum
        growth_stocks = [
            'PLTR', 'SNOW', 'COIN', 'RBLX', 'U', 'NET', 'DDOG', 'CRWD',
            'ZM', 'DOCU', 'SHOP', 'SQ', 'PYPL', 'ROKU', 'UBER', 'LYFT'
        ]
        
        # Industrial & energy
        industrial_stocks = [
            'CAT', 'BA', 'GE', 'MMM', 'HON', 'UNP', 'LMT', 'RTX',
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'HAL', 'KMI'
        ]
        
        # Combine all categories
        all_symbols = (tech_stocks + financial_stocks + healthcare_stocks + 
                      consumer_stocks + growth_stocks + industrial_stocks)
        
        # Return exactly 100 unique symbols
        unique_symbols = list(set(all_symbols))
        return unique_symbols[:100]
    
    def fetch_alpaca_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data from Alpaca with robust error handling"""
        try:
            print(f"   📊 Fetching {symbol}: {start_date} to {end_date}", end=" ")
            
            # Convert string dates to datetime objects
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                limit=5000  # Ensure we get enough data
            )
            
            bars = self.data_client.get_stock_bars(request_params)
            
            # Check if we got data
            if not bars or symbol not in bars:
                print("❌ No bars returned")
                return pd.DataFrame()
            
            symbol_bars = bars[symbol]
            if not symbol_bars:
                print("❌ Empty symbol bars")
                return pd.DataFrame()
            
            # Convert to DataFrame with proper error handling
            data = []
            for bar in symbol_bars:
                try:
                    data.append({
                        'Date': bar.timestamp.date(),
                        'Open': float(bar.open),
                        'High': float(bar.high),
                        'Low': float(bar.low),
                        'Close': float(bar.close),
                        'Volume': int(bar.volume)
                    })
                except Exception as bar_error:
                    continue  # Skip problematic bars
            
            if not data:
                print("❌ No valid data points")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            
            # Additional data validation
            if len(df) < 100:  # Need minimum data points
                print(f"❌ Insufficient data: {len(df)} days")
                return pd.DataFrame()
            
            # Check for data quality
            if df['Close'].isnull().sum() > len(df) * 0.1:  # More than 10% null
                print(f"❌ Too many null values")
                return pd.DataFrame()
            
            print(f"✅ {len(df)} days")
            return df
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            return pd.DataFrame()
    
    def download_universe_data(self) -> Dict[str, pd.DataFrame]:
        """Download historical data for entire universe with comprehensive validation"""
        universe = self.get_comprehensive_universe()
        print(f"📥 Downloading 2-year data for {len(universe)} symbols...")
        print(f"📅 Period: {self.training_start} to {self.forward_end}")
        
        universe_data = {}
        successful_count = 0
        failed_symbols = []
        
        for i, symbol in enumerate(universe):
            print(f"Progress: {i+1}/{len(universe)}")
            
            # Download FULL period (training + forward test)
            data = self.fetch_alpaca_data(symbol, self.training_start, self.forward_end)
            
            if not data.empty:
                # Strict validation for 2-year + 3-month period
                expected_days = 600  # Approximately 2 years + 3 months of trading days
                
                if len(data) >= expected_days * 0.8:  # At least 80% of expected days
                    universe_data[symbol] = data
                    successful_count += 1
                    print(f"   ✅ Added {symbol} with {len(data)} days")
                else:
                    failed_symbols.append(f"{symbol} ({len(data)} days)")
                    print(f"   ❌ Insufficient data: {symbol} ({len(data)} < {expected_days * 0.8:.0f})")
            else:
                failed_symbols.append(f"{symbol} (no data)")
                print(f"   ❌ No data: {symbol}")
            
            # Rate limiting - respect Alpaca API limits
            if i % 10 == 0 and i > 0:
                print(f"   ⏸️ Rate limiting pause (processed {i} symbols)...")
                time.sleep(3)  # Longer pause for API stability
        
        print(f"\n📊 DATA DOWNLOAD SUMMARY:")
        print(f"   ✅ Successful: {successful_count} symbols")
        print(f"   ❌ Failed: {len(failed_symbols)} symbols")
        
        if failed_symbols:
            print(f"   Failed symbols: {', '.join(failed_symbols[:10])}")
            if len(failed_symbols) > 10:
                print(f"   ... and {len(failed_symbols) - 10} more")
        
        if successful_count < 50:
            print(f"⚠️ WARNING: Only {successful_count} symbols downloaded. Need at least 50 for robust testing.")
        
        return universe_data
    
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical features"""
        df = data.copy()
        
        # Price-based features
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # Momentum indicators
        df['rsi'] = self.calculate_rsi(df['Close'])
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volatility features
        df['atr'] = self.calculate_atr(df)
        df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['Close'])
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Price relationships
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_sma_ratio'] = df['Close'] / df['sma_20']
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Target variables (for training)
        df['next_return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['next_return_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        df['target_binary'] = (df['next_return'] > 0.01).astype(int)  # >1% return
        df['target_regression'] = df['next_return']
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(window=period).mean()
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        return upper, lower
    
    def screen_elite_stocks(self, universe_data: Dict[str, pd.DataFrame]) -> List[str]:
        """AI-powered stock screening (using training data only)"""
        print("🔍 AI Elite Stock Screening (Training Data Only)")
        
        elite_stocks = []
        
        for symbol, data in universe_data.items():
            # CRITICAL: Use only training data for screening
            training_data = data[data.index <= self.training_end]
            
            if len(training_data) < 400:  # Need sufficient training data
                continue
            
            # Screening criteria
            avg_volume = training_data['Volume'].tail(60).mean()
            avg_price = training_data['Close'].tail(60).mean()
            volatility = training_data['Close'].pct_change().std()
            
            # Quality filters
            if (avg_volume > 1000000 and  # Liquid stocks
                avg_price > 10 and        # No penny stocks
                avg_price < 1000 and      # Not too expensive
                volatility > 0.01 and     # Some movement
                volatility < 0.1):        # Not too volatile
                
                elite_stocks.append(symbol)
        
        print(f"✅ Screened {len(elite_stocks)} elite stocks from {len(universe_data)}")
        return elite_stocks
    
    def train_ml_models(self, elite_stocks: List[str]) -> Dict[str, Dict]:
        """Train ML models using ONLY training data"""
        print(f"🤖 Training ML Models ({len(elite_stocks)} stocks)")
        print("⚠️ STRICT DATA ISOLATION: Using only training period data")
        
        trained_models = {}
        
        for symbol in elite_stocks:
            print(f"   Training {symbol}...", end=" ")
            
            # Get full data for this symbol
            full_data = self.universe_data[symbol]
            
            # CRITICAL: Use only training data
            training_data = full_data[full_data.index <= self.training_end].copy()
            
            if len(training_data) < 300:
                print("❌ Insufficient training data")
                continue
            
            # Create features
            features_data = self.create_advanced_features(training_data)
            
            # Feature columns
            feature_cols = [
                'sma_5', 'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
                'atr', 'bb_position', 'volume_ratio', 'high_low_ratio',
                'close_sma_ratio', 'momentum_5', 'momentum_20'
            ]
            
            # Clean data
            clean_data = features_data[feature_cols + ['target_binary', 'target_regression']].dropna()
            
            if len(clean_data) < 200:
                print("❌ Insufficient clean data")
                continue
            
            # Prepare training data
            X = clean_data[feature_cols]
            y_binary = clean_data['target_binary']
            y_regression = clean_data['target_regression']
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train classification model
            clf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            clf_model.fit(X_scaled, y_binary)
            
            # Train regression model
            reg_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            reg_model.fit(X_scaled, y_regression)
            
            # Validate model performance
            accuracy = clf_model.score(X_scaled, y_binary)
            r2 = reg_model.score(X_scaled, y_regression)
            
            if accuracy > 0.52 and r2 > 0.01:  # Minimum thresholds
                trained_models[symbol] = {
                    'classifier': clf_model,
                    'regressor': reg_model,
                    'scaler': scaler,
                    'features': feature_cols,
                    'accuracy': accuracy,
                    'r2_score': r2,
                    'feature_importance': dict(zip(feature_cols, clf_model.feature_importances_))
                }
                print(f"✅ Acc: {accuracy:.1%}, R²: {r2:.3f}")
            else:
                print(f"❌ Poor performance (Acc: {accuracy:.1%}, R²: {r2:.3f})")
        
        print(f"✅ Successfully trained {len(trained_models)} models")
        return trained_models
    
    def generate_trading_signal(self, symbol: str, current_date: str) -> Dict:
        """Generate comprehensive trading signal with multiple decision types"""
        
        if symbol not in self.trained_models:
            return {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 'position_size': 0.0}
        
        try:
            # Get data up to current date (NO FUTURE LEAKAGE)
            full_data = self.universe_data[symbol]
            available_data = full_data[full_data.index <= current_date].copy()
            
            if len(available_data) < 50:
                return {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 'position_size': 0.0}
            
            # Create features
            features_data = self.create_advanced_features(available_data)
            latest_features = features_data[self.trained_models[symbol]['features']].iloc[-1]
            
            if latest_features.isnull().any():
                return {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 'position_size': 0.0}
            
            # Scale features
            scaler = self.trained_models[symbol]['scaler']
            features_scaled = scaler.transform([latest_features.values])
            
            # Get predictions
            clf_model = self.trained_models[symbol]['classifier']
            reg_model = self.trained_models[symbol]['regressor']
            
            # Classification prediction (buy/hold probability)
            buy_prob = clf_model.predict_proba(features_scaled)[0][1]
            
            # Regression prediction (expected return)
            expected_return = reg_model.predict(features_scaled)[0]
            
            # Technical signal strength
            current_price = available_data['Close'].iloc[-1]
            sma_20 = available_data['Close'].rolling(20).mean().iloc[-1]
            rsi = self.calculate_rsi(available_data['Close']).iloc[-1]
            
            # Volume analysis
            recent_volume = available_data['Volume'].tail(5).mean()
            avg_volume = available_data['Volume'].tail(60).mean()
            volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volatility analysis
            volatility = available_data['Close'].pct_change().tail(20).std()
            
            # Combine signals with sophisticated logic
            ml_signal_strength = (buy_prob - 0.5) * 2  # Convert to -1 to 1 range
            tech_signal_strength = (current_price / sma_20 - 1) * 5  # Trend strength
            rsi_signal = (50 - rsi) / 50  # RSI momentum (oversold = positive)
            volume_signal = min((volume_surge - 1) * 2, 1.0)  # Volume confirmation
            
            # Final signal calculation with weights
            combined_strength = (
                0.4 * ml_signal_strength + 
                0.25 * tech_signal_strength + 
                0.15 * rsi_signal +
                0.10 * volume_signal +
                0.10 * (expected_return * 20)  # Expected return component
            )
            
            # Determine signal and position sizing
            signal, position_size = self.determine_trading_action(
                combined_strength, buy_prob, expected_return, volatility, volume_surge
            )
            
            return {
                'signal': signal,
                'strength': abs(combined_strength),
                'confidence': buy_prob,
                'expected_return': expected_return,
                'position_size': position_size,
                'current_price': current_price,
                'ml_component': ml_signal_strength,
                'tech_component': tech_signal_strength,
                'rsi_component': rsi_signal,
                'volume_component': volume_signal,
                'volatility': volatility,
                'rsi': rsi
            }
            
        except Exception as e:
            return {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 'position_size': 0.0}
    
    def determine_trading_action(self, combined_strength: float, buy_prob: float, 
                               expected_return: float, volatility: float, volume_surge: float) -> Tuple[str, float]:
        """Determine specific trading action and position size"""
        
        # Base position size on signal strength and confidence
        base_position_size = self.position_size_pct
        
        # Strong signals with high confidence
        if combined_strength > 0.7 and buy_prob > 0.75:
            return 'BUY', base_position_size * 1.5  # Full size
        
        # Good signals with decent confidence
        elif combined_strength > 0.5 and buy_prob > 0.65:
            return 'BUY', base_position_size  # Normal size
        
        # Moderate signals - partial position
        elif combined_strength > self.min_signal_strength and buy_prob > 0.6:
            return 'PARTIAL_BUY', base_position_size * 0.6  # Reduced size
        
        # Strong sell signals
        elif combined_strength < -0.7 and buy_prob < 0.25:
            return 'SELL', 1.0  # Sell all
        
        # Moderate sell signals
        elif combined_strength < -0.5 and buy_prob < 0.35:
            return 'PARTIAL_SELL', 0.5  # Sell half
        
        # Weak sell signals or risk management
        elif combined_strength < -self.min_signal_strength:
            return 'PARTIAL_SELL', 0.3  # Reduce position
        
        # Default to hold
        else:
            return 'HOLD', 0.0
    
    def run_forward_test(self) -> Dict:
        """Run 3-month forward test with NO data leakage"""
        print("🔮 STARTING 3-MONTH FORWARD TEST")
        print("⚠️ CRITICAL: Using ONLY unseen forward test data")
        print(f"📅 Period: {self.forward_start} to {self.forward_end}")
        
        # Generate trading dates
        forward_dates = pd.bdate_range(start=self.forward_start, end=self.forward_end)
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'total_value': self.initial_capital
        }
        
        trades = []
        daily_performance = []
        
        print(f"📊 Simulating {len(forward_dates)} trading days...")
        
        for i, date in enumerate(forward_dates):
            date_str = date.strftime('%Y-%m-%d')
            
            if i % 20 == 0:
                print(f"   Day {i+1}/{len(forward_dates)}: {date_str}")
            
            # Generate signals for all symbols
            daily_signals = {}
            
            for symbol in self.trained_models.keys():
                signal_info = self.generate_trading_signal(symbol, date_str)
                if signal_info['signal'] != 'HOLD':
                    daily_signals[symbol] = signal_info
            
            # Execute trades based on signals
            self.execute_daily_trades(portfolio, daily_signals, date_str, trades)
            
            # Calculate portfolio value
            portfolio_value = self.calculate_portfolio_value(portfolio, date_str)
            
            daily_performance.append({
                'date': date_str,
                'portfolio_value': portfolio_value,
                'cash': portfolio['cash'],
                'positions': len([p for p in portfolio['positions'].values() if p['shares'] > 0])
            })
        
        # Calculate final results
        return self.calculate_results(daily_performance, trades)
    
    def execute_daily_trades(self, portfolio: Dict, signals: Dict, date_str: str, trades: List):
        """Execute sophisticated trades based on signal types (BUY/SELL/HOLD/PARTIAL_BUY/PARTIAL_SELL)"""
        
        current_positions = len([p for p in portfolio['positions'].values() if p['shares'] > 0])
        
        # Sort signals by strength for priority execution
        sorted_signals = sorted(signals.items(), key=lambda x: x[1]['strength'], reverse=True)
        
        for symbol, signal_info in sorted_signals:
            signal = signal_info['signal']
            strength = signal_info['strength']
            price = signal_info['current_price']
            position_size = signal_info['position_size']
            confidence = signal_info['confidence']
            
            # Execute different signal types
            if signal == 'BUY' and current_positions < self.max_positions:
                # Full buy order
                position_value = portfolio['cash'] * position_size
                shares = int(position_value / price)
                
                if shares > 0 and portfolio['cash'] > position_value * 1.1:
                    total_cost = shares * price * 1.003  # Include transaction costs
                    
                    portfolio['cash'] -= total_cost
                    portfolio['positions'][symbol] = {
                        'shares': shares,
                        'avg_price': price,
                        'entry_date': date_str,
                        'entry_signal_strength': strength
                    }
                    current_positions += 1
                    
                    trades.append({
                        'date': date_str,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': price,
                        'value': shares * price,
                        'strength': strength,
                        'confidence': confidence,
                        'signal_type': 'FULL_BUY'
                    })
            
            elif signal == 'PARTIAL_BUY' and current_positions < self.max_positions:
                # Partial buy order (smaller position)
                position_value = portfolio['cash'] * position_size
                shares = int(position_value / price)
                
                if shares > 0 and portfolio['cash'] > position_value * 1.1:
                    total_cost = shares * price * 1.003
                    
                    if symbol in portfolio['positions'] and portfolio['positions'][symbol]['shares'] > 0:
                        # Add to existing position
                        existing_shares = portfolio['positions'][symbol]['shares']
                        existing_avg = portfolio['positions'][symbol]['avg_price']
                        new_avg = (existing_shares * existing_avg + shares * price) / (existing_shares + shares)
                        
                        portfolio['positions'][symbol]['shares'] += shares
                        portfolio['positions'][symbol]['avg_price'] = new_avg
                    else:
                        # New position
                        portfolio['positions'][symbol] = {
                            'shares': shares,
                            'avg_price': price,
                            'entry_date': date_str,
                            'entry_signal_strength': strength
                        }
                        current_positions += 1
                    
                    portfolio['cash'] -= total_cost
                    
                    trades.append({
                        'date': date_str,
                        'symbol': symbol,
                        'action': 'PARTIAL_BUY',
                        'shares': shares,
                        'price': price,
                        'value': shares * price,
                        'strength': strength,
                        'confidence': confidence,
                        'signal_type': 'PARTIAL_BUY'
                    })
            
            elif signal == 'SELL' and symbol in portfolio['positions']:
                # Full sell order
                position = portfolio['positions'][symbol]
                if position['shares'] > 0:
                    shares = position['shares']
                    trade_value = shares * price * 0.997  # Transaction costs
                    
                    portfolio['cash'] += trade_value
                    portfolio['positions'][symbol] = {'shares': 0, 'avg_price': 0, 'entry_date': '', 'entry_signal_strength': 0}
                    
                    # Calculate P&L
                    entry_price = position['avg_price']
                    pnl = (price - entry_price) * shares
                    pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0
                    
                    trades.append({
                        'date': date_str,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': shares,
                        'price': price,
                        'value': shares * price,
                        'strength': strength,
                        'confidence': confidence,
                        'signal_type': 'FULL_SELL',
                        'entry_price': entry_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })
            
            elif signal == 'PARTIAL_SELL' and symbol in portfolio['positions']:
                # Partial sell order
                position = portfolio['positions'][symbol]
                if position['shares'] > 0:
                    shares_to_sell = int(position['shares'] * position_size)
                    if shares_to_sell > 0:
                        trade_value = shares_to_sell * price * 0.997
                        
                        portfolio['cash'] += trade_value
                        portfolio['positions'][symbol]['shares'] -= shares_to_sell
                        
                        # Calculate P&L for sold portion
                        entry_price = position['avg_price']
                        pnl = (price - entry_price) * shares_to_sell
                        pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0
                        
                        trades.append({
                            'date': date_str,
                            'symbol': symbol,
                            'action': 'PARTIAL_SELL',
                            'shares': shares_to_sell,
                            'price': price,
                            'value': shares_to_sell * price,
                            'strength': strength,
                            'confidence': confidence,
                            'signal_type': 'PARTIAL_SELL',
                            'entry_price': entry_price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'remaining_shares': portfolio['positions'][symbol]['shares']
                        })
            
            # Risk management: Stop-loss and take-profit checks
            elif symbol in portfolio['positions']:
                position = portfolio['positions'][symbol]
                if position['shares'] > 0:
                    entry_price = position['avg_price']
                    current_pnl_pct = (price - entry_price) / entry_price
                    
                    # Stop-loss trigger
                    if current_pnl_pct <= -self.stop_loss_pct:
                        shares = position['shares']
                        trade_value = shares * price * 0.997
                        
                        portfolio['cash'] += trade_value
                        portfolio['positions'][symbol] = {'shares': 0, 'avg_price': 0, 'entry_date': '', 'entry_signal_strength': 0}
                        
                        trades.append({
                            'date': date_str,
                            'symbol': symbol,
                            'action': 'STOP_LOSS',
                            'shares': shares,
                            'price': price,
                            'value': shares * price,
                            'strength': 0,
                            'confidence': 0,
                            'signal_type': 'RISK_MANAGEMENT',
                            'entry_price': entry_price,
                            'pnl': (price - entry_price) * shares,
                            'pnl_pct': current_pnl_pct
                        })
                    
                    # Take-profit trigger
                    elif current_pnl_pct >= self.take_profit_pct:
                        shares = position['shares']
                        trade_value = shares * price * 0.997
                        
                        portfolio['cash'] += trade_value
                        portfolio['positions'][symbol] = {'shares': 0, 'avg_price': 0, 'entry_date': '', 'entry_signal_strength': 0}
                        
                        trades.append({
                            'date': date_str,
                            'symbol': symbol,
                            'action': 'TAKE_PROFIT',
                            'shares': shares,
                            'price': price,
                            'value': shares * price,
                            'strength': 0,
                            'confidence': 0,
                            'signal_type': 'RISK_MANAGEMENT',
                            'entry_price': entry_price,
                            'pnl': (price - entry_price) * shares,
                            'pnl_pct': current_pnl_pct
                        })
    
    def calculate_portfolio_value(self, portfolio: Dict, date_str: str) -> float:
        """Calculate total portfolio value"""
        total_value = portfolio['cash']
        
        for symbol, position in portfolio['positions'].items():
            if position['shares'] > 0 and symbol in self.universe_data:
                try:
                    symbol_data = self.universe_data[symbol]
                    available_data = symbol_data[symbol_data.index <= date_str]
                    current_price = available_data['Close'].iloc[-1]
                    total_value += position['shares'] * current_price
                except:
                    continue
        
        return total_value
    
    def calculate_results(self, daily_performance: List, trades: List) -> Dict:
        """Calculate comprehensive results"""
        
        if not daily_performance:
            return {}
        
        # Basic performance metrics
        initial_value = daily_performance[0]['portfolio_value']
        final_value = daily_performance[-1]['portfolio_value']
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        values = [d['portfolio_value'] for d in daily_performance]
        daily_returns = np.diff(values) / values[:-1]
        
        # Risk metrics
        volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = (total_return * 252 / len(daily_returns)) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Trading statistics
        n_trades = len(trades)
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        # Calculate win rate
        profitable_trades = 0
        total_profit = 0
        
        for sell_trade in sell_trades:
            symbol = sell_trade['symbol']
            # Find corresponding buy trade
            buy_trade = None
            for bt in reversed(buy_trades):
                if bt['symbol'] == symbol and bt['date'] <= sell_trade['date']:
                    buy_trade = bt
                    break
            
            if buy_trade:
                profit = (sell_trade['price'] - buy_trade['price']) * sell_trade['shares']
                total_profit += profit
                if profit > 0:
                    profitable_trades += 1
        
        win_rate = profitable_trades / len(sell_trades) if sell_trades else 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': total_return * (365 / 90),  # 3 months to annual
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'n_trades': n_trades,
            'n_buy_trades': len(buy_trades),
            'n_sell_trades': len(sell_trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'daily_performance': daily_performance,
            'trades': trades,
            'period_start': self.forward_start,
            'period_end': self.forward_end,
            'data_source': 'alpaca',
            'models_used': len(self.trained_models)
        }
        
        return results
    
    def run_comprehensive_backtest(self) -> Dict:
        """Run complete backtesting pipeline"""
        
        print("🚀 STARTING COMPREHENSIVE ALPACA BACKTEST")
        print("=" * 70)
        print("📋 PHASE OVERVIEW:")
        print("   1. Download 2-year universe data")
        print("   2. Screen elite stocks (training data only)")
        print("   3. Train ML models (training data only)")
        print("   4. Run 3-month forward test (unseen data)")
        print("   5. Calculate comprehensive results")
        print("=" * 70)
        
        try:
            # Phase 1: Download universe data
            print("\n📥 PHASE 1: Download Universe Data")
            self.universe_data = self.download_universe_data()
            
            if not self.universe_data:
                raise Exception("No data downloaded")
            
            # Phase 2: Screen elite stocks
            print("\n🔍 PHASE 2: Elite Stock Screening")
            elite_stocks = self.screen_elite_stocks(self.universe_data)
            
            if not elite_stocks:
                raise Exception("No elite stocks identified")
            
            # Phase 3: Train ML models
            print("\n🤖 PHASE 3: ML Model Training")
            self.trained_models = self.train_ml_models(elite_stocks)
            
            if not self.trained_models:
                raise Exception("No successful models trained")
            
            # Phase 4: Forward test
            print("\n🔮 PHASE 4: Forward Test Simulation")
            results = self.run_forward_test()
            
            # Phase 5: Results analysis
            print("\n📊 PHASE 5: Results Analysis")
            self.display_results(results)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"alpaca_comprehensive_backtest_{timestamp}.json"
            
            # Convert numpy types for JSON serialization
            json_results = self.prepare_results_for_json(results)
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved: {results_file}")
            return results
            
        except Exception as e:
            print(f"\n❌ Backtest failed: {e}")
            traceback.print_exc()
            return {}
    
    def prepare_results_for_json(self, results: Dict) -> Dict:
        """Prepare results for JSON serialization"""
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, np.integer):
                json_results[key] = int(value)
            elif isinstance(value, np.floating):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        return json_results
    
    def display_results(self, results: Dict):
        """Display comprehensive results"""
        
        print("🎯 COMPREHENSIVE BACKTEST RESULTS")
        print("=" * 50)
        
        print(f"💰 PERFORMANCE METRICS:")
        print(f"   Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"   Final Value: ${results['final_value']:,.2f}")
        print(f"   Total Return: {results['total_return']:.2%}")
        print(f"   Annualized Return: {results['annualized_return']:.2%}")
        print(f"   Volatility: {results['volatility']:.2%}")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
        
        print(f"\n📊 TRADING STATISTICS:")
        print(f"   Total Trades: {results['n_trades']}")
        print(f"   Buy Trades: {results['n_buy_trades']}")
        print(f"   Sell Trades: {results['n_sell_trades']}")
        print(f"   Win Rate: {results['win_rate']:.1%}")
        print(f"   Total Profit: ${results['total_profit']:,.2f}")
        
        print(f"\n🤖 MODEL STATISTICS:")
        print(f"   Models Trained: {results['models_used']}")
        print(f"   Elite Stocks: {len(self.trained_models)}")
        print(f"   Universe Size: {len(self.universe_data)}")
        
        print(f"\n📅 PERIOD ANALYSIS:")
        print(f"   Training: {self.training_start} to {self.training_end}")
        print(f"   Forward Test: {results['period_start']} to {results['period_end']}")
        print(f"   Data Source: {results['data_source']}")
        
        # Performance assessment
        target_return = 0.30  # 30% annual target
        achieved_annual = results['annualized_return']
        
        print(f"\n🎯 TARGET ASSESSMENT:")
        print(f"   Target: {target_return:.0%} annual")
        print(f"   Achieved: {achieved_annual:.1%} annual")
        
        if achieved_annual >= target_return:
            print("   ✅ TARGET ACHIEVED!")
        else:
            print("   ⚠️ Below target")
        
        print("\n🚀 BACKTEST COMPLETE!")

def main():
    """Run comprehensive backtest"""
    
    try:
        backtester = AlpacaComprehensiveBacktester()
        results = backtester.run_comprehensive_backtest()
        
        if results:
            print("\n✅ Comprehensive backtest completed successfully!")
        else:
            print("\n❌ Backtest failed")
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
