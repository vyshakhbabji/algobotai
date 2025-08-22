"""
🎯 ENHANCEMENT C: PARAMETER OPTIMIZATION
Quick Win: 5-10% Performance Boost

Optimizing:
- Signal thresholds for each stock
- Position sizing rules  
- Risk management parameters
- ML confidence thresholds
- Trading frequency parameters
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
from typing import Dict, List, Tuple
import warnings
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Enhanced parameter optimization ranges
PARAMETER_RANGES = {
    'position_max_pct': [0.08, 0.12, 0.15, 0.18, 0.22],  # Position sizing
    'min_signal_strength': [0.3, 0.35, 0.4, 0.45, 0.5],  # Entry threshold
    'partial_sell_threshold': [0.6, 0.65, 0.7, 0.75, 0.8],  # Partial sell trigger
    'ml_confidence_threshold': [0.4, 0.5, 0.6, 0.65, 0.7],  # ML confidence filter
    'regime_volatility_threshold': [1.2, 1.3, 1.4, 1.5, 1.6],  # Volatility regime
    'momentum_lookback': [5, 7, 10, 12, 15],  # Momentum calculation period
    'volume_threshold': [1.1, 1.2, 1.3, 1.4, 1.5]  # Volume confirmation
}

# Stock-specific parameter ranges (top performers get individual optimization)
STOCK_SPECIFIC_RANGES = {
    'NVDA': {
        'position_max_pct': [0.15, 0.18, 0.22, 0.25],
        'min_signal_strength': [0.3, 0.35, 0.4]
    },
    'META': {
        'position_max_pct': [0.12, 0.15, 0.18, 0.22],
        'min_signal_strength': [0.35, 0.4, 0.45]
    },
    'JPM': {
        'position_max_pct': [0.08, 0.12, 0.15],
        'min_signal_strength': [0.4, 0.45, 0.5]
    }
}

# Top performing stocks from our baseline
TOP_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']

class ParameterOptimizer:
    """Advanced parameter optimization system"""
    
    def __init__(self):
        self.optimization_results = []
        self.best_params = {}
        self.baseline_return = 57.6  # Our current best
        
    def generate_parameter_combinations(self, focused: bool = True) -> List[Dict]:
        """Generate smart parameter combinations for testing"""
        combinations = []
        
        if focused:
            # Smart sampling - focus on promising ranges
            base_params = {
                'position_max_pct': 0.15,
                'min_signal_strength': 0.4,
                'partial_sell_threshold': 0.7,
                'ml_confidence_threshold': 0.6,
                'regime_volatility_threshold': 1.4,
                'momentum_lookback': 10,
                'volume_threshold': 1.3
            }
            
            # Create variations around base parameters
            for param, base_value in base_params.items():
                param_range = PARAMETER_RANGES[param]
                
                # Find closest values in range
                base_idx = min(range(len(param_range)), 
                             key=lambda i: abs(param_range[i] - base_value))
                
                # Test neighboring values
                test_indices = [
                    max(0, base_idx - 1),
                    base_idx,
                    min(len(param_range) - 1, base_idx + 1)
                ]
                
                for idx in test_indices:
                    test_params = base_params.copy()
                    test_params[param] = param_range[idx]
                    combinations.append(test_params)
            
            # Add some random combinations
            for _ in range(10):
                random_params = {}
                for param, param_range in PARAMETER_RANGES.items():
                    random_params[param] = np.random.choice(param_range)
                combinations.append(random_params)
        
        else:
            # Full grid search (use for smaller parameter sets)
            param_names = list(PARAMETER_RANGES.keys())
            param_values = [PARAMETER_RANGES[param] for param in param_names]
            
            for combo in itertools.product(*param_values):
                param_dict = dict(zip(param_names, combo))
                combinations.append(param_dict)
        
        # Remove duplicates
        unique_combinations = []
        seen = set()
        for combo in combinations:
            combo_tuple = tuple(sorted(combo.items()))
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                unique_combinations.append(combo)
        
        return unique_combinations
    
    def create_optimized_config(self, params: Dict) -> Dict:
        """Create trading config with optimized parameters"""
        return {
            'position_max_pct': params['position_max_pct'],
            'min_signal_strength': params['min_signal_strength'],
            'partial_sell_threshold': params['partial_sell_threshold'],
            'ml_confidence_threshold': params['ml_confidence_threshold'],
            'regime_volatility_threshold': params['regime_volatility_threshold'],
            'momentum_lookback': params['momentum_lookback'],
            'volume_threshold': params['volume_threshold'],
            'max_positions': 8,
            'lookback_days': 90,
            'optimization_run': True
        }
    
    def calculate_enhanced_features(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate features with optimized parameters"""
        if df.empty or len(df) < 20:
            return df
            
        df = df.copy()
        momentum_lookback = params['momentum_lookback']
        
        # Price features
        df['returns'] = df['Close'].pct_change()
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            if len(df) >= period:
                df[f'sma_{period}'] = df['Close'].rolling(period).mean()
                df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['Close'], 14)
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        bb_period = 20
        bb_middle = df['Close'].rolling(bb_period).mean()
        bb_std = df['Close'].rolling(bb_period).std()
        df['bb_upper'] = bb_middle + (bb_std * 2)
        df['bb_lower'] = bb_middle - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume with optimized threshold
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['high_volume'] = (df['volume_ratio'] > params['volume_threshold']).astype(int)
        
        # Volatility with optimized threshold
        df['volatility'] = df['returns'].rolling(20).std()
        df['avg_volatility'] = df['volatility'].rolling(50).mean()
        df['high_volatility'] = (df['volatility'] > df['avg_volatility'] * params['regime_volatility_threshold']).astype(int)
        
        # Optimized momentum
        if len(df) > momentum_lookback:
            df['momentum'] = df['Close'] / df['Close'].shift(momentum_lookback) - 1
            df['momentum_strength'] = abs(df['momentum'])
        
        # ATR
        df['atr'] = self._calculate_atr(df, 14)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
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
    
    def detect_market_regime(self, df: pd.DataFrame, params: Dict) -> str:
        """Enhanced regime detection with optimized parameters"""
        if df.empty or len(df) < 20:
            return 'ranging'
        
        recent_data = df.tail(20)
        
        # Trend analysis
        sma_20 = recent_data['Close'].rolling(20).mean()
        trend_up = (recent_data['Close'].iloc[-1] > sma_20.iloc[-1])
        
        # Volatility analysis with optimized threshold
        volatility = recent_data['returns'].std()
        avg_volatility = df['returns'].rolling(50).std().mean()
        high_vol = volatility > (avg_volatility * params['regime_volatility_threshold'])
        
        # Momentum analysis with optimized lookback
        momentum_period = params['momentum_lookback']
        if len(recent_data) >= momentum_period:
            momentum = recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[-momentum_period] - 1
            strong_momentum = abs(momentum) > 0.05
        else:
            strong_momentum = False
        
        # Regime classification
        if strong_momentum and trend_up:
            return 'trending_up'
        elif strong_momentum and not trend_up:
            return 'trending_down'
        elif high_vol:
            return 'volatile'
        else:
            return 'ranging'
    
    def train_optimized_ml_model(self, symbol: str, df: pd.DataFrame, params: Dict) -> Dict:
        """Train ML model with optimized parameters"""
        try:
            df_featured = self.calculate_enhanced_features(df.copy(), params)
            df_featured = df_featured.dropna()
            
            if len(df_featured) < 50:
                return {'signal_accuracy': 0.0, 'regime_accuracy': 0.0}
            
            # Feature selection
            feature_cols = [
                'rsi', 'macd', 'bb_position', 'volume_ratio', 'volatility',
                'price_sma_20_ratio', 'momentum', 'momentum_strength', 'high_volume', 'high_volatility'
            ]
            
            available_features = [col for col in feature_cols if col in df_featured.columns]
            if len(available_features) < 5:
                return {'signal_accuracy': 0.0, 'regime_accuracy': 0.0}
            
            X = df_featured[available_features].fillna(0)
            
            # Enhanced target creation
            forward_returns = df_featured['Close'].shift(-5) / df_featured['Close'] - 1
            signal_strength = np.clip(0.5 + (forward_returns * 8), 0.3, 1.0)
            
            # Regime target
            regime_labels = []
            for i in range(len(df_featured)):
                end_idx = min(i + 10, len(df_featured))
                regime = self.detect_market_regime(df_featured.iloc[max(0, i-20):end_idx], params)
                regime_labels.append(regime)
            
            regime_map = {'trending_up': 0, 'trending_down': 1, 'ranging': 2, 'volatile': 3}
            regime_encoded = pd.Series(regime_labels).map(regime_map).fillna(2)
            
            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_signal_train = signal_strength.iloc[:split_idx]
            y_signal_test = signal_strength.iloc[split_idx:]
            y_regime_train = regime_encoded.iloc[:split_idx]
            y_regime_test = regime_encoded.iloc[split_idx:]
            
            # Clean data
            train_mask = ~(y_signal_train.isna() | X_train.isna().any(axis=1))
            test_mask = ~(y_signal_test.isna() | X_test.isna().any(axis=1))
            
            if train_mask.sum() < 20 or test_mask.sum() < 5:
                return {'signal_accuracy': 0.0, 'regime_accuracy': 0.0}
            
            X_train_clean = X_train[train_mask]
            y_signal_train_clean = y_signal_train[train_mask]
            y_regime_train_clean = y_regime_train[train_mask]
            X_test_clean = X_test[test_mask]
            y_signal_test_clean = y_signal_test[test_mask]
            y_regime_test_clean = y_regime_test[test_mask]
            
            # Train models
            signal_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            
            signal_model.fit(X_train_clean, y_signal_train_clean)
            signal_pred = signal_model.predict(X_test_clean)
            signal_accuracy = max(0, 1 - np.mean(np.abs(signal_pred - y_signal_test_clean)))
            
            regime_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            
            regime_model.fit(X_train_clean, y_regime_train_clean)
            regime_pred = regime_model.predict(X_test_clean)
            regime_accuracy = np.mean(regime_pred == y_regime_test_clean)
            
            return {
                'signal_model': signal_model,
                'regime_model': regime_model,
                'feature_cols': available_features,
                'signal_accuracy': signal_accuracy,
                'regime_accuracy': regime_accuracy
            }
            
        except Exception as e:
            print(f"❌ ML training failed for {symbol}: {e}")
            return {'signal_accuracy': 0.0, 'regime_accuracy': 0.0}
    
    def get_optimized_signal(self, symbol: str, df: pd.DataFrame, params: Dict, model_info: Dict) -> Dict:
        """Get trading signal with optimized parameters"""
        if df.empty or not model_info or 'signal_model' not in model_info:
            return {
                'signal_strength': 0.5,
                'base_signal': 0.0,
                'confidence': 0.0,
                'market_regime': 'ranging'
            }
        
        try:
            df_featured = self.calculate_enhanced_features(df.copy(), params)
            if df_featured.empty:
                return {'signal_strength': 0.5, 'base_signal': 0.0, 'confidence': 0.0, 'market_regime': 'ranging'}
            
            latest_row = df_featured.iloc[-1]
            
            # Prepare features
            X = latest_row[model_info['feature_cols']].fillna(0).values.reshape(1, -1)
            
            # Get ML predictions
            signal_strength = model_info['signal_model'].predict(X)[0]
            regime_pred = model_info['regime_model'].predict(X)[0]
            
            regime_map = {0: 'trending_up', 1: 'trending_down', 2: 'ranging', 3: 'volatile'}
            market_regime = regime_map.get(regime_pred, 'ranging')
            
            # Calculate base signal with optimized thresholds
            rsi = latest_row.get('rsi', 50)
            macd = latest_row.get('macd', 0)
            bb_position = latest_row.get('bb_position', 0.5)
            volume_ratio = latest_row.get('volume_ratio', 1.0)
            
            base_signal = 0.0
            
            # Enhanced signal logic with optimized parameters
            if rsi < 30 and macd > 0:
                base_signal = 0.8
            elif rsi < 35:
                base_signal = 0.5
            elif rsi > 70 and macd < 0:
                base_signal = -0.8
            elif rsi > 65:
                base_signal = -0.5
            
            if bb_position < 0.2:
                base_signal += 0.3
            elif bb_position > 0.8:
                base_signal -= 0.3
            
            # Volume confirmation with optimized threshold
            if volume_ratio > params['volume_threshold']:
                base_signal *= 1.2
            elif volume_ratio < 0.8:
                base_signal *= 0.8
            
            # Regime adjustment
            regime_multiplier = {
                'trending_up': 1.3,
                'trending_down': 0.7,
                'ranging': 1.0,
                'volatile': 0.9
            }.get(market_regime, 1.0)
            
            final_signal_strength = np.clip(signal_strength * regime_multiplier, 0.3, 1.0)
            
            return {
                'signal_strength': final_signal_strength,
                'base_signal': np.clip(base_signal, -1.0, 1.0),
                'confidence': model_info['signal_accuracy'],
                'market_regime': market_regime
            }
            
        except Exception as e:
            print(f"❌ Signal generation failed for {symbol}: {e}")
            return {'signal_strength': 0.5, 'base_signal': 0.0, 'confidence': 0.0, 'market_regime': 'ranging'}
    
    def make_optimized_trading_decision(self, symbol: str, current_price: float, 
                                      signal_info: Dict, params: Dict, 
                                      positions: Dict, cash: float, total_value: float) -> Dict:
        """Make trading decision with optimized parameters"""
        signal_strength = signal_info['signal_strength']
        base_signal = signal_info['base_signal']
        confidence = signal_info['confidence']
        
        current_position = positions.get(symbol, 0)
        position_value = current_position * current_price
        
        decision = {
            'action': 'hold',
            'quantity': 0,
            'reason': 'No signal'
        }
        
        # Apply confidence filter
        if confidence < params['ml_confidence_threshold']:
            return decision
        
        # Enhanced position sizing with optimized parameters
        max_position_value = total_value * params['position_max_pct']
        
        if base_signal > params['min_signal_strength']:
            # Buy signal
            if current_position == 0:
                # New position with Kelly sizing
                kelly_fraction = confidence * signal_strength * 0.6
                position_size = min(max_position_value, cash * kelly_fraction)
                quantity = int(position_size / current_price)
                
                if quantity > 0 and len(positions) < params.get('max_positions', 8):
                    decision = {
                        'action': 'buy',
                        'quantity': quantity,
                        'reason': f'New position - Signal: {signal_strength:.2f}, Conf: {confidence:.2f}'
                    }
            
            elif position_value < max_position_value * 0.8:
                # Add to position
                additional_size = (max_position_value - position_value) * 0.5
                quantity = int(additional_size / current_price)
                
                if quantity > 0:
                    decision = {
                        'action': 'buy',
                        'quantity': quantity,
                        'reason': f'Add to position - Strong signal: {signal_strength:.2f}'
                    }
        
        elif (base_signal < -params['min_signal_strength'] or 
              signal_strength < params['partial_sell_threshold']) and current_position > 0:
            # Sell signal with optimized thresholds
            if signal_strength < 0.5 or base_signal < -0.6:
                # Full sell
                decision = {
                    'action': 'sell',
                    'quantity': current_position,
                    'reason': f'Full sell - Strong negative: {base_signal:.2f}'
                }
            elif signal_strength < params['partial_sell_threshold']:
                # Partial sell
                sell_quantity = int(current_position * 0.4)
                decision = {
                    'action': 'sell',
                    'quantity': sell_quantity,
                    'reason': f'Partial sell - Weak signal: {signal_strength:.2f}'
                }
        
        return decision
    
    def run_optimized_simulation(self, params: Dict, start_date: str = "2024-05-20", 
                                end_date: str = "2024-08-20") -> Dict:
        """Run trading simulation with optimized parameters"""
        
        # Initialize portfolio
        cash = 50000
        positions = {}
        total_value = cash
        trade_history = []
        ml_models = {}
        
        # Date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        portfolio_values = [cash]
        trade_count = 0
        
        # Progress tracking
        model_updates = 0
        current_date = start_dt
        
        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Weekdays only
                try:
                    # Update ML models every 5 days
                    if (current_date - start_dt).days % 5 == 0:
                        for symbol in TOP_STOCKS[:8]:  # Focus on top 8 for speed
                            try:
                                ml_start = (current_date - timedelta(days=90)).strftime('%Y-%m-%d')
                                ml_end = current_date.strftime('%Y-%m-%d')
                                
                                ticker = yf.Ticker(symbol)
                                df = ticker.history(period='6mo')
                                
                                if not df.empty and len(df) >= 30:
                                    df = df.reset_index()
                                    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                                    
                                    model_info = self.train_optimized_ml_model(symbol, df, params)
                                    if model_info['signal_accuracy'] > 0.1:
                                        ml_models[symbol] = model_info
                                        model_updates += 1
                            except:
                                pass
                    
                    # Generate signals and trade
                    for symbol in TOP_STOCKS:
                        try:
                            ticker = yf.Ticker(symbol)
                            df = ticker.history(period='3mo')
                            
                            if df.empty or len(df) < 20:
                                continue
                            
                            df = df.reset_index()
                            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                            current_price = df['Close'].iloc[-1]
                            
                            # Get signal
                            if symbol in ml_models:
                                signal_info = self.get_optimized_signal(symbol, df, params, ml_models[symbol])
                            else:
                                signal_info = {'signal_strength': 0.5, 'base_signal': 0.0, 'confidence': 0.0, 'market_regime': 'ranging'}
                            
                            # Make decision
                            decision = self.make_optimized_trading_decision(
                                symbol, current_price, signal_info, params, positions, cash, total_value
                            )
                            
                            # Execute trade
                            if decision['action'] == 'buy':
                                cost = decision['quantity'] * current_price
                                if cost <= cash:
                                    positions[symbol] = positions.get(symbol, 0) + decision['quantity']
                                    cash -= cost
                                    trade_count += 1
                                    
                                    trade_history.append({
                                        'date': current_date,
                                        'symbol': symbol,
                                        'action': 'buy',
                                        'quantity': decision['quantity'],
                                        'price': current_price,
                                        'reason': decision['reason']
                                    })
                            
                            elif decision['action'] == 'sell' and symbol in positions:
                                sell_quantity = min(decision['quantity'], positions[symbol])
                                proceeds = sell_quantity * current_price
                                positions[symbol] -= sell_quantity
                                cash += proceeds
                                trade_count += 1
                                
                                if positions[symbol] <= 0:
                                    del positions[symbol]
                                
                                trade_history.append({
                                    'date': current_date,
                                    'symbol': symbol,
                                    'action': 'sell',
                                    'quantity': sell_quantity,
                                    'price': current_price,
                                    'reason': decision['reason']
                                })
                        
                        except Exception as e:
                            continue
                    
                    # Update portfolio value
                    portfolio_value = cash
                    for symbol, quantity in positions.items():
                        try:
                            ticker = yf.Ticker(symbol)
                            current_price = ticker.history(period='1d')['Close'].iloc[-1]
                            portfolio_value += quantity * current_price
                        except:
                            pass
                    
                    total_value = portfolio_value
                    portfolio_values.append(portfolio_value)
                
                except Exception as e:
                    pass
            
            current_date += timedelta(days=1)
        
        # Calculate performance
        final_value = portfolio_values[-1]
        initial_value = portfolio_values[0]
        total_return = (final_value / initial_value - 1) * 100
        
        trading_days = len([d for d in pd.date_range(start_dt, end_dt) if d.weekday() < 5])
        annualized_return = ((final_value / initial_value) ** (252 / trading_days) - 1) * 100
        
        return {
            'params': params,
            'final_value': final_value,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'total_trades': trade_count,
            'model_updates': model_updates,
            'final_positions': len(positions)
        }
    
    def optimize_parameters(self, max_tests: int = 20) -> Dict:
        """Run parameter optimization"""
        print("🎯 PARAMETER OPTIMIZATION")
        print("Enhancement C: Optimizing trading parameters")
        print("=" * 50)
        
        # Generate parameter combinations
        combinations = self.generate_parameter_combinations(focused=True)[:max_tests]
        
        print(f"🧪 Testing {len(combinations)} parameter combinations...")
        
        best_result = None
        best_return = self.baseline_return
        
        for i, params in enumerate(combinations):
            try:
                print(f"\n🔬 Test {i+1}/{len(combinations)}: ", end="")
                
                # Run simulation
                result = self.run_optimized_simulation(params)
                
                annual_return = result['annualized_return_pct']
                improvement = annual_return - self.baseline_return
                
                print(f"{annual_return:.1f}% ({improvement:+.1f}%)")
                print(f"   Params: pos_max={params['position_max_pct']:.2f}, min_sig={params['min_signal_strength']:.2f}")
                
                # Track result
                self.optimization_results.append(result)
                
                # Update best if improved
                if annual_return > best_return:
                    best_return = annual_return
                    best_result = result
                    print(f"   🎉 NEW BEST: {annual_return:.1f}% (+{improvement:.1f}%)")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        # Save results
        if best_result:
            self.best_params = best_result['params']
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"parameter_optimization_results_{timestamp}.json"
            
            optimization_summary = {
                'baseline_return': self.baseline_return,
                'best_result': best_result,
                'improvement_pct': best_result['annualized_return_pct'] - self.baseline_return,
                'all_results': self.optimization_results,
                'total_tests': len(combinations)
            }
            
            with open(results_file, 'w') as f:
                json.dump(optimization_summary, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {results_file}")
        
        return best_result or {'annualized_return_pct': self.baseline_return}

def main():
    """Run parameter optimization"""
    
    print("🎯 ENHANCEMENT C: PARAMETER OPTIMIZATION")
    print("Target: 5-10% performance boost (62-67% annual returns)")
    print("=" * 60)
    
    optimizer = ParameterOptimizer()
    best_result = optimizer.optimize_parameters(max_tests=15)  # Quick test
    
    baseline = 57.6
    optimized = best_result['annualized_return_pct']
    improvement = optimized - baseline
    
    print(f"\n" + "=" * 60)
    print("🏆 PARAMETER OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"📊 Baseline Return: {baseline:.1f}%")
    print(f"🎯 Optimized Return: {optimized:.1f}%")
    print(f"📈 Improvement: {improvement:+.1f}%")
    
    if improvement >= 5:
        print(f"🎉 SUCCESS! Enhancement C delivered {improvement:.1f}% boost!")
        print("✅ Ready for Enhancement D (Advanced ML Models)")
    elif improvement >= 2:
        print(f"✅ Good improvement: {improvement:.1f}% boost")
        print("💡 Proceed to Enhancement D for additional gains")
    else:
        print(f"⚠️ Modest improvement: {improvement:.1f}%")
        print("🔧 Consider additional parameter tuning")
    
    # Show best parameters
    if 'params' in best_result:
        print(f"\n🔧 OPTIMIZED PARAMETERS:")
        for param, value in best_result['params'].items():
            print(f"   {param}: {value}")
    
    return best_result

if __name__ == "__main__":
    main()
