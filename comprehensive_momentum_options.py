#!/usr/bin/env python3
"""
COMPREHENSIVE MOMENTUM OPTIONS TRADING MODEL
Advanced options strategies based on institutional momentum features
Integrates with deployed momentum portfolio for leveraged returns
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import json
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import norm
import math

class ComprehensiveMomentumOptionsModel:
    def __init__(self):
        # Our deployed momentum portfolio stocks (WINNING FOUNDATION)
        self.momentum_stocks = ['AMD', 'GE', 'PLTR', 'MSFT', 'NVDA', 'JNJ', 'CAT', 'GOOGL']
        
        # Institutional momentum parameters (matching deployed portfolio)
        self.momentum_periods = {
            'long_momentum': 126,    # 6 months (primary institutional signal)
            'medium_momentum': 63,   # 3 months (confirmation signal)
            'short_momentum': 21,    # 1 month (timing signal)
            'micro_momentum': 5      # 1 week (entry/exit timing)
        }
        
        # Options strategy parameters
        self.strategy_configs = {
            'momentum_calls': {
                'min_momentum_score': 0.75,      # Strong momentum required
                'max_iv_percentile': 50,         # Prefer low IV
                'target_dte': [30, 45],          # Days to expiration
                'delta_range': [0.45, 0.65],     # ATM to slightly ITM
                'max_position_size': 0.05,       # 5% of portfolio max
                'profit_target': 1.00,           # 100% gain target
                'stop_loss': 0.50                # 50% loss limit
            },
            'momentum_spreads': {
                'min_momentum_score': 0.50,      # Moderate momentum
                'min_iv_percentile': 40,         # Higher IV preferred
                'target_dte': [20, 40],          # Shorter duration
                'long_delta': [0.60, 0.75],      # ITM long leg
                'short_delta': [0.25, 0.40],     # OTM short leg
                'max_spread_width': 10,          # $10 max width
                'profit_target': 0.50,           # 50% of max profit
                'stop_loss': 0.75                # 75% loss (limited risk)
            },
            'momentum_straddles': {
                'min_volatility_expansion': 0.20, # 20% vol increase expected
                'min_momentum_acceleration': 0.03, # Accelerating momentum
                'target_dte': [15, 30],           # Short-term volatility play
                'delta_target': 0.50,             # ATM straddle
                'max_position_size': 0.03,        # 3% of portfolio max
                'profit_target': 0.75,            # 75% gain target
                'stop_loss': 0.60                 # 60% loss limit
            },
            'momentum_iron_condors': {
                'momentum_range': [0.25, 0.60],   # Moderate momentum (range-bound)
                'min_iv_percentile': 60,          # High IV for premium collection
                'target_dte': [25, 45],           # Medium duration
                'wing_width': 5,                  # $5 wings
                'profit_target': 0.25,            # 25% of max profit
                'stop_loss': 2.0                  # 200% of credit received
            }
        }
        
        # Risk management parameters
        self.risk_params = {
            'max_portfolio_options_allocation': 0.20,  # 20% max in options
            'max_single_position': 0.05,               # 5% max per position
            'correlation_limit': 0.70,                 # Max correlation between positions
            'vix_threshold': 35,                       # Reduce exposure above VIX 35
            'daily_var_limit': 0.02                    # 2% daily VaR limit
        }
        
    def calculate_momentum_features_for_options(self, data):
        """Calculate momentum features specifically for options trading decisions"""
        if len(data) < 100:
            return None
        
        # Handle MultiIndex columns from yfinance
        if hasattr(data.columns, 'levels'):
            # MultiIndex columns - extract the data
            symbol = data.columns[0][1] if len(data.columns[0]) > 1 else 'UNKNOWN'
            close_prices = data[('Close', symbol)].values if ('Close', symbol) in data.columns else data.iloc[:, 0].values
            volume_data = data[('Volume', symbol)].values if ('Volume', symbol) in data.columns else None
            high_prices = data[('High', symbol)].values if ('High', symbol) in data.columns else close_prices
            low_prices = data[('Low', symbol)].values if ('Low', symbol) in data.columns else close_prices
        else:
            # Regular columns
            close_prices = data['Close'].values
            volume_data = data['Volume'].values if 'Volume' in data.columns else None
            high_prices = data['High'].values if 'High' in data.columns else close_prices
            low_prices = data['Low'].values if 'Low' in data.columns else close_prices
        
        features = {}
        
        # === CORE MOMENTUM METRICS (matching deployed portfolio) ===
        for period_name, period_days in self.momentum_periods.items():
            if len(close_prices) > period_days + 10:
                # Raw momentum
                momentum = (close_prices[-1] - close_prices[-period_days-1]) / close_prices[-period_days-1]
                features[f'{period_name}_momentum'] = momentum
                
                # Risk-adjusted momentum
                period_returns = np.diff(close_prices[-period_days:]) / close_prices[-period_days:-1]
                if len(period_returns) > 0 and np.std(period_returns) > 1e-8:
                    risk_adj = np.mean(period_returns) / np.std(period_returns) * np.sqrt(252)
                    features[f'{period_name}_risk_adj'] = risk_adj
                else:
                    features[f'{period_name}_risk_adj'] = 0
                
                # Momentum acceleration (for straddle strategies)
                if len(close_prices) > period_days * 2:
                    prev_momentum = (close_prices[-period_days-1] - close_prices[-period_days*2-1]) / close_prices[-period_days*2-1]
                    features[f'{period_name}_acceleration'] = momentum - prev_momentum
                else:
                    features[f'{period_name}_acceleration'] = 0
                
                # Momentum consistency (trend reliability)
                positive_periods = np.sum(period_returns > 0) / len(period_returns) if len(period_returns) > 0 else 0.5
                features[f'{period_name}_consistency'] = positive_periods
        
        # === VOLATILITY METRICS (crucial for options) ===
        volatility_periods = [10, 20, 30, 60]
        
        for vol_period in volatility_periods:
            if len(close_prices) > vol_period:
                returns = np.diff(close_prices[-vol_period:]) / close_prices[-vol_period:-1]
                
                # Realized volatility (annualized)
                realized_vol = np.std(returns) * np.sqrt(252)
                features[f'realized_vol_{vol_period}d'] = realized_vol
                
                # Volatility trend
                if vol_period >= 20 and len(close_prices) > vol_period * 2:
                    recent_vol = np.std(np.diff(close_prices[-vol_period//2:]) / close_prices[-vol_period//2:-1]) * np.sqrt(252)
                    historical_vol = np.std(np.diff(close_prices[-vol_period:-vol_period//2]) / close_prices[-vol_period:-vol_period//2-1]) * np.sqrt(252)
                    
                    if historical_vol > 0:
                        features[f'vol_trend_{vol_period}d'] = (recent_vol - historical_vol) / historical_vol
                    else:
                        features[f'vol_trend_{vol_period}d'] = 0
                
                # Downside volatility (for protective strategies)
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_vol = np.std(negative_returns) * np.sqrt(252)
                    features[f'downside_vol_{vol_period}d'] = downside_vol
                else:
                    features[f'downside_vol_{vol_period}d'] = 0
        
        # Volatility regime detection
        if 'realized_vol_20d' in features and 'realized_vol_60d' in features:
            features['vol_regime'] = features['realized_vol_20d'] / features['realized_vol_60d']
        
        # === VOLUME ANALYSIS (for options liquidity assessment) ===
        if volume_data is not None and len(volume_data) > 30:
            # Volume momentum
            recent_vol = np.mean(volume_data[-5:])
            historical_vol = np.mean(volume_data[-21:-1])
            features['volume_momentum'] = (recent_vol - historical_vol) / historical_vol if historical_vol > 0 else 0
            
            # Volume surge detection (for event-driven strategies)
            avg_volume_60d = np.mean(volume_data[-60:])
            current_volume = volume_data[-1]
            features['volume_surge'] = current_volume / avg_volume_60d if avg_volume_60d > 0 else 1
            
            # Volume-price correlation (confirms momentum)
            if len(volume_data) >= 20:
                price_changes = np.diff(close_prices[-20:])
                volume_changes = np.diff(volume_data[-20:])
                if len(price_changes) > 0 and len(volume_changes) > 0:
                    correlation = np.corrcoef(price_changes, volume_changes)[0,1]
                    features['price_volume_corr'] = correlation if not np.isnan(correlation) else 0
        
        # === TECHNICAL LEVELS (for strike selection) ===
        # Support and resistance levels
        if len(close_prices) > 50:
            # Recent high/low levels
            high_20d = np.max(high_prices[-20:])
            low_20d = np.min(low_prices[-20:])
            current_price = close_prices[-1]
            
            features['resistance_distance'] = (high_20d - current_price) / current_price
            features['support_distance'] = (current_price - low_20d) / current_price
            features['range_position'] = (current_price - low_20d) / (high_20d - low_20d) if high_20d > low_20d else 0.5
            
            # Moving average levels
            ma_periods = [10, 20, 50]
            for ma_period in ma_periods:
                if len(close_prices) > ma_period:
                    ma = np.mean(close_prices[-ma_period:])
                    features[f'price_vs_ma{ma_period}'] = (current_price - ma) / ma
        
        # === IMPLIED VOLATILITY ESTIMATION (for IV rank approximation) ===
        # Using GARCH-like volatility estimation
        if len(close_prices) > 60:
            returns = np.diff(close_prices[-60:]) / close_prices[-60:-1]
            
            # Simple EWMA volatility
            lambda_decay = 0.94
            weights = np.array([lambda_decay**(i) for i in range(len(returns))])
            weights = weights / np.sum(weights)
            
            ewma_var = np.sum(weights * returns**2)
            ewma_vol = np.sqrt(ewma_var * 252)
            features['ewma_volatility'] = ewma_vol
            
            # Volatility percentile (approximate IV rank)
            historical_vols = []
            for i in range(30, len(returns)):
                period_returns = returns[i-30:i]
                period_vol = np.std(period_returns) * np.sqrt(252)
                historical_vols.append(period_vol)
            
            if historical_vols:
                current_vol = features.get('realized_vol_20d', ewma_vol)
                vol_percentile = np.percentile(historical_vols, 
                                             [vol for vol in historical_vols if vol < current_vol])
                features['vol_percentile'] = len([vol for vol in historical_vols if vol < current_vol]) / len(historical_vols)
        
        # === MOMENTUM PATTERN RECOGNITION ===
        # Trend acceleration
        if len(close_prices) > 30:
            short_trend = (close_prices[-1] - close_prices[-6]) / close_prices[-6]
            medium_trend = (close_prices[-6] - close_prices[-16]) / close_prices[-16]
            long_trend = (close_prices[-16] - close_prices[-31]) / close_prices[-31]
            
            features['trend_acceleration'] = short_trend - medium_trend
            features['trend_consistency'] = 1 if (short_trend > 0) == (medium_trend > 0) == (long_trend > 0) else 0
        
        # Breakout detection
        if len(close_prices) > 20:
            bollinger_period = 20
            bollinger_std = 2
            
            bb_ma = np.mean(close_prices[-bollinger_period:])
            bb_std = np.std(close_prices[-bollinger_period:])
            
            upper_band = bb_ma + (bollinger_std * bb_std)
            lower_band = bb_ma - (bollinger_std * bb_std)
            
            features['bb_position'] = (close_prices[-1] - bb_ma) / (bb_std * bollinger_std) if bb_std > 0 else 0
            features['bb_squeeze'] = 1 if bb_std < np.mean([np.std(close_prices[-i-bollinger_period:-i]) for i in range(1, 6)]) else 0
        
        # === MARKET CONTEXT ===
        try:
            # VIX for market volatility context
            vix_data = yf.download('^VIX', start=str(data.index[0].date()), end=str(data.index[-1].date() + timedelta(days=1)), progress=False)
            if not vix_data.empty:
                # Handle MultiIndex columns
                if hasattr(vix_data.columns, 'levels'):
                    vix_close = vix_data[('Close', '^VIX')] if ('Close', '^VIX') in vix_data.columns else vix_data.iloc[:, 0]
                else:
                    vix_close = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
                
                features['vix_level'] = float(vix_close.iloc[-1])
                if len(vix_close) > 6:
                    features['vix_trend'] = float((vix_close.iloc[-1] - vix_close.iloc[-6]) / vix_close.iloc[-6])
                else:
                    features['vix_trend'] = 0
            else:
                features['vix_level'] = 20  # Default neutral level
                features['vix_trend'] = 0
            
            # Market correlation (SPY)
            spy_data = yf.download('SPY', start=str(data.index[0].date()), end=str(data.index[-1].date() + timedelta(days=1)), progress=False)
            if not spy_data.empty and len(spy_data) > 20:
                # Handle MultiIndex columns
                if hasattr(spy_data.columns, 'levels'):
                    spy_close = spy_data[('Close', 'SPY')] if ('Close', 'SPY') in spy_data.columns else spy_data.iloc[:, 0]
                else:
                    spy_close = spy_data['Close'] if 'Close' in spy_data.columns else spy_data.iloc[:, 0]
                
                stock_returns = np.diff(close_prices[-20:]) / close_prices[-20:-1]
                spy_returns = np.diff(spy_close.values[-20:]) / spy_close.values[-20:-1]
                
                if len(stock_returns) == len(spy_returns):
                    market_correlation = np.corrcoef(stock_returns, spy_returns)[0,1]
                    features['market_correlation'] = float(market_correlation) if not np.isnan(market_correlation) else 0
                else:
                    features['market_correlation'] = 0
            else:
                features['market_correlation'] = 0
                
        except Exception as e:
            print(f"   ⚠️  Market data error: {str(e)[:50]}...")
            features['vix_level'] = 20
            features['vix_trend'] = 0
            features['market_correlation'] = 0
        
        # Clean features
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0
        
        return features
    
    def calculate_momentum_score(self, features):
        """Calculate comprehensive momentum score for options strategy selection"""
        if not features:
            return 0
        
        # Weighted momentum calculation (matching institutional approach)
        long_momentum = features.get('long_momentum_momentum', 0) * 0.4
        medium_momentum = features.get('medium_momentum_momentum', 0) * 0.3
        short_momentum = features.get('short_momentum_momentum', 0) * 0.2
        micro_momentum = features.get('micro_momentum_momentum', 0) * 0.1
        
        raw_momentum = long_momentum + medium_momentum + short_momentum + micro_momentum
        
        # Risk adjustment
        risk_adj_factor = features.get('long_momentum_risk_adj', 0)
        if risk_adj_factor > 1:
            risk_bonus = 0.1
        elif risk_adj_factor < -0.5:
            risk_bonus = -0.1
        else:
            risk_bonus = 0
        
        # Consistency bonus
        consistency = features.get('long_momentum_consistency', 0.5)
        consistency_bonus = (consistency - 0.5) * 0.2
        
        # Acceleration factor
        acceleration = features.get('medium_momentum_acceleration', 0)
        acceleration_bonus = acceleration * 0.5
        
        # Volume confirmation
        volume_momentum = features.get('volume_momentum', 0)
        price_volume_corr = features.get('price_volume_corr', 0)
        volume_bonus = 0.05 if volume_momentum > 0.2 and price_volume_corr > 0.3 else 0
        
        total_score = raw_momentum + risk_bonus + consistency_bonus + acceleration_bonus + volume_bonus
        
        # Normalize to 0-1 scale
        normalized_score = max(0, min(1, (total_score + 0.5) / 1.0))
        
        return normalized_score
    
    def identify_options_strategies(self, symbol, features):
        """Identify optimal options strategies based on momentum analysis"""
        momentum_score = self.calculate_momentum_score(features)
        strategies = []
        
        current_price = features.get('price_vs_ma10', 0) + 1  # Approximate current price relative
        vol_percentile = features.get('vol_percentile', 0.5) * 100
        vol_trend = features.get('vol_trend_20d', 0)
        acceleration = features.get('medium_momentum_acceleration', 0)
        vix_level = features.get('vix_level', 20)
        
        # === MOMENTUM CALLS ===
        if momentum_score >= self.strategy_configs['momentum_calls']['min_momentum_score']:
            if vol_percentile <= self.strategy_configs['momentum_calls']['max_iv_percentile']:
                strategy = {
                    'strategy_type': 'momentum_calls',
                    'symbol': symbol,
                    'rationale': f"Strong momentum ({momentum_score:.2f}) with low IV ({vol_percentile:.0f}%)",
                    'momentum_score': momentum_score,
                    'vol_percentile': vol_percentile,
                    'target_dte': self.strategy_configs['momentum_calls']['target_dte'],
                    'delta_range': self.strategy_configs['momentum_calls']['delta_range'],
                    'position_size': min(0.05, momentum_score * 0.06),
                    'profit_target': self.strategy_configs['momentum_calls']['profit_target'],
                    'stop_loss': self.strategy_configs['momentum_calls']['stop_loss'],
                    'priority': momentum_score * (1 - vol_percentile/100)
                }
                strategies.append(strategy)
        
        # === MOMENTUM SPREADS ===
        if (momentum_score >= self.strategy_configs['momentum_spreads']['min_momentum_score'] and
            vol_percentile >= self.strategy_configs['momentum_spreads']['min_iv_percentile']):
            
            strategy = {
                'strategy_type': 'momentum_spreads',
                'symbol': symbol,
                'rationale': f"Moderate momentum ({momentum_score:.2f}) with elevated IV ({vol_percentile:.0f}%)",
                'momentum_score': momentum_score,
                'vol_percentile': vol_percentile,
                'target_dte': self.strategy_configs['momentum_spreads']['target_dte'],
                'long_delta': self.strategy_configs['momentum_spreads']['long_delta'],
                'short_delta': self.strategy_configs['momentum_spreads']['short_delta'],
                'position_size': min(0.04, momentum_score * 0.05),
                'profit_target': self.strategy_configs['momentum_spreads']['profit_target'],
                'stop_loss': self.strategy_configs['momentum_spreads']['stop_loss'],
                'priority': momentum_score * (vol_percentile/100)
            }
            strategies.append(strategy)
        
        # === MOMENTUM STRADDLES ===
        if (abs(acceleration) >= self.strategy_configs['momentum_straddles']['min_momentum_acceleration'] and
            vol_trend >= self.strategy_configs['momentum_straddles']['min_volatility_expansion']):
            
            strategy = {
                'strategy_type': 'momentum_straddles',
                'symbol': symbol,
                'rationale': f"Momentum acceleration ({acceleration:.3f}) with vol expansion ({vol_trend:.2f})",
                'momentum_score': momentum_score,
                'acceleration': acceleration,
                'vol_trend': vol_trend,
                'target_dte': self.strategy_configs['momentum_straddles']['target_dte'],
                'delta_target': self.strategy_configs['momentum_straddles']['delta_target'],
                'position_size': min(0.03, abs(acceleration) * 0.5),
                'profit_target': self.strategy_configs['momentum_straddles']['profit_target'],
                'stop_loss': self.strategy_configs['momentum_straddles']['stop_loss'],
                'priority': abs(acceleration) * vol_trend
            }
            strategies.append(strategy)
        
        # === IRON CONDORS (for range-bound momentum) ===
        range_momentum = self.strategy_configs['momentum_iron_condors']['momentum_range']
        if (range_momentum[0] <= momentum_score <= range_momentum[1] and
            vol_percentile >= self.strategy_configs['momentum_iron_condors']['min_iv_percentile']):
            
            range_position = features.get('range_position', 0.5)
            if 0.3 <= range_position <= 0.7:  # Trading in middle of range
                strategy = {
                    'strategy_type': 'momentum_iron_condors',
                    'symbol': symbol,
                    'rationale': f"Range-bound momentum ({momentum_score:.2f}) with high IV ({vol_percentile:.0f}%)",
                    'momentum_score': momentum_score,
                    'vol_percentile': vol_percentile,
                    'range_position': range_position,
                    'target_dte': self.strategy_configs['momentum_iron_condors']['target_dte'],
                    'wing_width': self.strategy_configs['momentum_iron_condors']['wing_width'],
                    'position_size': 0.02,
                    'profit_target': self.strategy_configs['momentum_iron_condors']['profit_target'],
                    'stop_loss': self.strategy_configs['momentum_iron_condors']['stop_loss'],
                    'priority': vol_percentile/100 * (1 - abs(range_position - 0.5) * 2)
                }
                strategies.append(strategy)
        
        # Sort strategies by priority
        strategies.sort(key=lambda x: x['priority'], reverse=True)
        
        return strategies
    
    def analyze_options_portfolio(self):
        """Analyze entire momentum portfolio for options opportunities"""
        print("📈 COMPREHENSIVE MOMENTUM OPTIONS ANALYSIS")
        print("=" * 60)
        print("🏛️ Using institutional momentum features for options selection")
        print("🎯 Identifying optimal leverage opportunities in momentum portfolio")
        print("=" * 60)
        
        all_strategies = []
        
        for symbol in self.momentum_stocks:
            print(f"\n🔍 Analyzing options opportunities for {symbol}...")
            
            try:
                # Get comprehensive data
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=200)
                
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                print(f"   📊 Downloaded {len(data)} rows for {symbol}")
                if not data.empty:
                    print(f"   📅 Date range: {str(data.index[0])[:10]} to {str(data.index[-1])[:10]}")
                    
                    # Handle price range with MultiIndex columns
                    if hasattr(data.columns, 'levels'):
                        symbol_name = data.columns[0][1] if len(data.columns[0]) > 1 else symbol
                        close_col = ('Close', symbol_name) if ('Close', symbol_name) in data.columns else data.columns[0]
                        price_min = float(data[close_col].min())
                        price_max = float(data[close_col].max())
                    else:
                        price_min = float(data['Close'].min())
                        price_max = float(data['Close'].max())
                    
                    print(f"   💰 Price range: ${price_min:.2f} - ${price_max:.2f}")
                
                if data.empty or len(data) < 100:
                    print(f"   ❌ Insufficient data for {symbol}")
                    continue
                
                # Calculate momentum features
                features = self.calculate_momentum_features_for_options(data)
                
                if not features:
                    print(f"   ❌ Could not calculate features for {symbol}")
                    continue
                
                # Get current price
                if hasattr(data.columns, 'levels'):
                    symbol_name = data.columns[0][1] if len(data.columns[0]) > 1 else symbol
                    current_price = float(data[('Close', symbol_name)].iloc[-1]) if ('Close', symbol_name) in data.columns else float(data.iloc[-1, 0])
                else:
                    current_price = float(data['Close'].iloc[-1])
                
                # Identify strategies
                strategies = self.identify_options_strategies(symbol, features)
                
                if strategies:
                    print(f"   ✅ {len(strategies)} strategies identified for {symbol} (${current_price:.2f})")
                    
                    for strategy in strategies:
                        strategy['current_price'] = current_price
                        strategy['analysis_date'] = datetime.now().isoformat()
                        all_strategies.append(strategy)
                        
                        print(f"      🎯 {strategy['strategy_type'].upper()}: {strategy['rationale']}")
                        print(f"         Priority: {strategy['priority']:.3f}, Size: {strategy['position_size']:.1%}")
                else:
                    momentum_score = self.calculate_momentum_score(features)
                    vol_percentile = features.get('vol_percentile', 0.5) * 100
                    print(f"   ⚪ No suitable strategies for {symbol}")
                    print(f"      Momentum: {momentum_score:.2f}, IV Percentile: {vol_percentile:.0f}%")
                    
            except Exception as e:
                print(f"   ❌ Error analyzing {symbol}: {str(e)}")
        
        # Portfolio-level analysis
        if all_strategies:
            # Sort all strategies by priority
            all_strategies.sort(key=lambda x: x['priority'], reverse=True)
            
            # Apply portfolio-level risk management
            selected_strategies = self.apply_portfolio_risk_management(all_strategies)
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"momentum_options_strategies_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(selected_strategies, f, indent=2, default=str)
            
            print(f"\n💾 Options analysis saved to {filename}")
            
            # Summary
            self.print_options_portfolio_summary(selected_strategies)
            
            return selected_strategies
        else:
            print("\n❌ No options strategies identified across portfolio")
            return []
    
    def apply_portfolio_risk_management(self, strategies):
        """Apply portfolio-level risk management to options strategies"""
        print(f"\n🛡️ APPLYING PORTFOLIO RISK MANAGEMENT")
        print(f"   Strategies before filtering: {len(strategies)}")
        
        selected = []
        total_allocation = 0
        symbol_allocations = {}
        strategy_type_counts = {}
        
        for strategy in strategies:
            symbol = strategy['symbol']
            strategy_type = strategy['strategy_type']
            position_size = strategy['position_size']
            
            # Check total portfolio allocation
            if total_allocation + position_size > self.risk_params['max_portfolio_options_allocation']:
                print(f"   ⚠️  Skipping {strategy_type} on {symbol}: Portfolio allocation limit reached")
                continue
            
            # Check single position limit
            if position_size > self.risk_params['max_single_position']:
                strategy['position_size'] = self.risk_params['max_single_position']
                position_size = self.risk_params['max_single_position']
                print(f"   📉 Reduced {strategy_type} on {symbol} position size to {position_size:.1%}")
            
            # Check symbol concentration
            symbol_total = symbol_allocations.get(symbol, 0) + position_size
            if symbol_total > self.risk_params['max_single_position'] * 2:  # Max 2x single position per symbol
                print(f"   ⚠️  Skipping {strategy_type} on {symbol}: Symbol concentration limit")
                continue
            
            # Check strategy type diversification
            if strategy_type_counts.get(strategy_type, 0) >= 3:  # Max 3 of same strategy type
                print(f"   ⚠️  Skipping {strategy_type} on {symbol}: Strategy type limit")
                continue
            
            # VIX-based adjustment
            vix_level = strategy.get('vol_percentile', 20)  # Use vol_percentile as VIX proxy
            if vix_level > self.risk_params['vix_threshold']:
                strategy['position_size'] *= 0.7  # Reduce size in high vol environment
                print(f"   📉 Reduced {strategy_type} on {symbol} due to high volatility")
            
            # Add to selected strategies
            selected.append(strategy)
            total_allocation += strategy['position_size']
            symbol_allocations[symbol] = symbol_allocations.get(symbol, 0) + strategy['position_size']
            strategy_type_counts[strategy_type] = strategy_type_counts.get(strategy_type, 0) + 1
        
        print(f"   ✅ Strategies after filtering: {len(selected)}")
        print(f"   📊 Total portfolio allocation: {total_allocation:.1%}")
        
        return selected
    
    def print_options_portfolio_summary(self, strategies):
        """Print comprehensive portfolio summary"""
        print(f"\n📊 MOMENTUM OPTIONS PORTFOLIO SUMMARY")
        print(f"=" * 50)
        
        if not strategies:
            print("❌ No strategies selected")
            return
        
        # Basic stats
        total_allocation = sum(s['position_size'] for s in strategies)
        strategy_types = {}
        symbol_exposure = {}
        
        for strategy in strategies:
            stype = strategy['strategy_type']
            symbol = strategy['symbol']
            
            strategy_types[stype] = strategy_types.get(stype, 0) + 1
            symbol_exposure[symbol] = symbol_exposure.get(symbol, 0) + strategy['position_size']
        
        print(f"📈 Total Strategies: {len(strategies)}")
        print(f"💰 Total Allocation: {total_allocation:.1%}")
        print(f"🎯 Symbols Covered: {len(symbol_exposure)}")
        
        print(f"\n🎲 STRATEGY BREAKDOWN:")
        for stype, count in strategy_types.items():
            emoji_map = {
                'momentum_calls': '🚀',
                'momentum_spreads': '📊', 
                'momentum_straddles': '⚡',
                'momentum_iron_condors': '🔒'
            }
            emoji = emoji_map.get(stype, '📈')
            print(f"   {emoji} {stype.replace('_', ' ').title()}: {count}")
        
        print(f"\n💼 SYMBOL EXPOSURE:")
        sorted_exposure = sorted(symbol_exposure.items(), key=lambda x: x[1], reverse=True)
        for symbol, exposure in sorted_exposure:
            print(f"   📊 {symbol}: {exposure:.1%}")
        
        print(f"\n🏆 TOP 5 PRIORITY STRATEGIES:")
        for i, strategy in enumerate(strategies[:5], 1):
            emoji_map = {
                'momentum_calls': '🚀',
                'momentum_spreads': '📊', 
                'momentum_straddles': '⚡',
                'momentum_iron_condors': '🔒'
            }
            emoji = emoji_map.get(strategy['strategy_type'], '📈')
            
            print(f"   {i}. {emoji} {strategy['symbol']} {strategy['strategy_type'].replace('_', ' ').title()}")
            print(f"      💰 Size: {strategy['position_size']:.1%}, Priority: {strategy['priority']:.3f}")
            print(f"      📝 {strategy['rationale']}")
        
        # Risk metrics
        print(f"\n🛡️ RISK METRICS:")
        print(f"   📊 Portfolio Options Allocation: {total_allocation:.1%} / {self.risk_params['max_portfolio_options_allocation']:.1%}")
        print(f"   🎯 Max Single Position: {max(s['position_size'] for s in strategies):.1%} / {self.risk_params['max_single_position']:.1%}")
        print(f"   📈 Strategy Diversification: {len(strategy_types)} types")
        
        # Expected outcomes
        total_bullish = len([s for s in strategies if s['strategy_type'] in ['momentum_calls', 'momentum_spreads']])
        total_neutral = len([s for s in strategies if s['strategy_type'] in ['momentum_straddles', 'momentum_iron_condors']])
        
        print(f"\n🎯 DIRECTIONAL EXPOSURE:")
        print(f"   📈 Bullish Strategies: {total_bullish}")
        print(f"   ⚖️  Neutral Strategies: {total_neutral}")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. 📋 Review strategy details in saved JSON file")
        print(f"   2. 🔍 Check options chains for target strikes/expirations")
        print(f"   3. 💰 Implement position sizing based on portfolio allocation")
        print(f"   4. ⏰ Set up alerts for entry/exit criteria")
        print(f"   5. 📊 Monitor momentum portfolio correlation")

def main():
    """Main execution function"""
    print("📈 COMPREHENSIVE MOMENTUM OPTIONS MODEL")
    print("🏛️ Institutional Momentum + Advanced Options Strategies")
    print("=" * 70)
    
    # Initialize options model
    options_model = ComprehensiveMomentumOptionsModel()
    
    # Run comprehensive options analysis
    strategies = options_model.analyze_options_portfolio()
    
    if strategies:
        print(f"\n🎯 MOMENTUM OPTIONS ANALYSIS COMPLETE!")
        print(f"✅ {len(strategies)} high-priority options strategies identified")
        print(f"🎲 Strategies span across momentum portfolio stocks")
        print(f"🛡️ Portfolio risk management applied")
        print(f"🏆 Ready for deployment alongside momentum equity portfolio")
        
        print(f"\n🔄 INTEGRATION WITH MOMENTUM PORTFOLIO:")
        print(f"   • Options strategies complement equity momentum positions")
        print(f"   • Leverage institutional momentum signals for options timing")
        print(f"   • Risk-managed allocation prevents over-concentration")
        print(f"   • Performance attribution separate from core momentum returns")
    
    return options_model

if __name__ == "__main__":
    options_model = main()
