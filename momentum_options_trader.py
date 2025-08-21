#!/usr/bin/env python3
"""
MOMENTUM-ENHANCED OPTIONS TRADING MODEL
Uses institutional momentum features to time options trades
Integrates with deployed momentum portfolio for signal generation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MomentumOptionsTrader:
    def __init__(self, symbols=None):
        # Use same stocks as momentum portfolio
        if symbols:
            self.symbols = symbols
        else:
            self.symbols = [
                # From our deployed momentum portfolio
                "AMD", "GE", "PLTR", "MSFT", "NVDA", "JNJ", "CAT", "GOOGL"
            ]
        
        # Options strategy parameters
        self.strategies = {
            'momentum_calls': {
                'min_momentum_6m': 15.0,
                'min_momentum_3m': 8.0,
                'max_volatility': 40.0,
                'dte_range': (30, 45),
                'delta_target': 0.50
            },
            'momentum_spreads': {
                'min_momentum_6m': 8.0,
                'min_momentum_3m': 4.0,
                'max_volatility': 60.0,
                'dte_range': (30, 60),
                'delta_long': 0.70,
                'delta_short': 0.30
            },
            'momentum_straddles': {
                'min_momentum_acceleration': 5.0,
                'min_volatility': 25.0,
                'dte_range': (14, 30),
                'delta_target': 0.50
            }
        }
    
    def calculate_institutional_momentum_features(self, data):
        """Calculate same momentum features as our deployed portfolio"""
        
        # CORE MOMENTUM METRICS (from institutional research)
        data['momentum_6m'] = data['Close'].pct_change(126) * 100
        data['momentum_3m'] = data['Close'].pct_change(63) * 100
        data['momentum_1m'] = data['Close'].pct_change(21) * 100
        
        # RISK-ADJUSTED MOMENTUM
        data['volatility_20d'] = data['Close'].rolling(20).std() / data['Close'].rolling(20).mean() * 100
        data['momentum_6m_risk_adj'] = data['momentum_6m'] / (data['volatility_20d'] + 0.1)
        
        # MOMENTUM ACCELERATION
        data['momentum_acceleration'] = data['momentum_3m'].diff(5)
        
        # IMPLIED VOLATILITY PROXY (realized vol)
        data['realized_vol_20d'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
        
        # VOLUME MOMENTUM
        volume_ma = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / volume_ma
        
        # PRICE MOMENTUM SCORE (composite)
        data['momentum_score'] = (
            data['momentum_6m'] * 0.4 +
            data['momentum_3m'] * 0.4 +
            data['momentum_1m'] * 0.2
        ) / (data['volatility_20d'] + 1)
        
        return data
    
    def analyze_momentum_for_options(self, symbol):
        """Analyze momentum characteristics for options trading"""
        
        # Get 1 year of data
        data = yf.download(symbol, period='1y', progress=False, auto_adjust=True)
        
        if data.empty:
            return None
        
        # Calculate momentum features
        data = self.calculate_institutional_momentum_features(data)
        
        # Current metrics - handle NaN values safely
        current = data.iloc[-1]
        
        # Safe extraction with NaN handling
        def safe_float(value, default=0.0):
            try:
                if pd.isna(value):
                    return default
                return float(value)
            except:
                return default
        
        analysis = {
            'symbol': symbol,
            'current_price': safe_float(current['Close'], 100.0),
            'momentum_6m': safe_float(current['momentum_6m'], 0.0),
            'momentum_3m': safe_float(current['momentum_3m'], 0.0),
            'momentum_1m': safe_float(current['momentum_1m'], 0.0),
            'momentum_acceleration': safe_float(current['momentum_acceleration'], 0.0),
            'realized_volatility': safe_float(current['realized_vol_20d'], 20.0),
            'volume_ratio': safe_float(current['volume_ratio'], 1.0),
            'momentum_score': safe_float(current['momentum_score'], 0.0),
            'price_trend': 'UP' if safe_float(current['momentum_3m'], 0.0) > 0 else 'DOWN'
        }
        
        return analysis
    
    def generate_momentum_call_signals(self, analysis):
        """Generate signals for momentum call options"""
        
        if not analysis:
            return None
        
        strategy = self.strategies['momentum_calls']
        
        # Check momentum criteria
        strong_momentum = (
            analysis['momentum_6m'] > strategy['min_momentum_6m'] and
            analysis['momentum_3m'] > strategy['min_momentum_3m'] and
            analysis['realized_volatility'] < strategy['max_volatility']
        )
        
        if not strong_momentum:
            return None
        
        # Calculate signal strength
        momentum_strength = min(100, (
            analysis['momentum_6m'] / strategy['min_momentum_6m'] * 50 +
            analysis['momentum_3m'] / strategy['min_momentum_3m'] * 30 +
            (1 - analysis['realized_volatility'] / strategy['max_volatility']) * 20
        ))
        
        return {
            'strategy': 'MOMENTUM_CALLS',
            'signal': 'BUY_CALLS',
            'symbol': analysis['symbol'],
            'strength': momentum_strength,
            'target_dte': strategy['dte_range'][0],
            'target_delta': strategy['delta_target'],
            'reasoning': f"Strong momentum: 6M={analysis['momentum_6m']:.1f}%, 3M={analysis['momentum_3m']:.1f}%",
            'risk_level': 'MEDIUM',
            'expected_duration': '30-45 days'
        }
    
    def generate_momentum_spread_signals(self, analysis):
        """Generate signals for momentum spread options"""
        
        if not analysis:
            return None
        
        strategy = self.strategies['momentum_spreads']
        
        # Check moderate momentum criteria
        moderate_momentum = (
            analysis['momentum_6m'] > strategy['min_momentum_6m'] and
            analysis['momentum_3m'] > strategy['min_momentum_3m'] and
            analysis['realized_volatility'] < strategy['max_volatility']
        )
        
        if not moderate_momentum:
            return None
        
        # Determine spread type based on momentum direction
        if analysis['momentum_3m'] > 0:
            spread_type = 'BULL_CALL_SPREAD'
        else:
            spread_type = 'BEAR_PUT_SPREAD'
        
        # Calculate signal strength
        momentum_strength = min(100, (
            analysis['momentum_6m'] / strategy['min_momentum_6m'] * 40 +
            analysis['momentum_3m'] / strategy['min_momentum_3m'] * 40 +
            analysis['volume_ratio'] * 20
        ))
        
        return {
            'strategy': 'MOMENTUM_SPREADS',
            'signal': spread_type,
            'symbol': analysis['symbol'],
            'strength': momentum_strength,
            'target_dte': strategy['dte_range'][1],
            'long_delta': strategy['delta_long'],
            'short_delta': strategy['delta_short'],
            'reasoning': f"Moderate momentum trend: {analysis['price_trend']}",
            'risk_level': 'LOW',
            'expected_duration': '30-60 days'
        }
    
    def generate_momentum_straddle_signals(self, analysis):
        """Generate signals for momentum volatility plays"""
        
        if not analysis:
            return None
        
        strategy = self.strategies['momentum_straddles']
        
        # Check acceleration criteria (momentum change)
        high_acceleration = (
            abs(analysis['momentum_acceleration']) > strategy['min_momentum_acceleration'] and
            analysis['realized_volatility'] > strategy['min_volatility']
        )
        
        if not high_acceleration:
            return None
        
        # Determine straddle direction expectation
        if analysis['momentum_acceleration'] > 0:
            expectation = 'UPWARD_BREAKOUT'
        else:
            expectation = 'DOWNWARD_BREAKOUT'
        
        # Calculate signal strength
        volatility_strength = min(100, (
            abs(analysis['momentum_acceleration']) / strategy['min_momentum_acceleration'] * 60 +
            analysis['realized_volatility'] / 50 * 40
        ))
        
        return {
            'strategy': 'MOMENTUM_STRADDLES',
            'signal': 'BUY_STRADDLE',
            'symbol': analysis['symbol'],
            'strength': volatility_strength,
            'target_dte': strategy['dte_range'][0],
            'target_delta': strategy['delta_target'],
            'expectation': expectation,
            'reasoning': f"High momentum acceleration: {analysis['momentum_acceleration']:.1f}%",
            'risk_level': 'HIGH',
            'expected_duration': '14-30 days'
        }
    
    def generate_options_portfolio(self):
        """Generate complete options trading signals for momentum portfolio"""
        
        print("📈 GENERATING MOMENTUM OPTIONS SIGNALS")
        print("=" * 45)
        print("🏛️ Using institutional momentum features")
        print("🎯 Analyzing momentum portfolio stocks")
        print("=" * 45)
        
        all_signals = []
        
        for symbol in self.symbols:
            print(f"\n🔍 Analyzing {symbol}...")
            
            # Analyze momentum characteristics
            analysis = self.analyze_momentum_for_options(symbol)
            
            if not analysis:
                print(f"   ❌ No data for {symbol}")
                continue
            
            print(f"   📊 6M Momentum: {analysis['momentum_6m']:+.1f}%")
            print(f"   📊 3M Momentum: {analysis['momentum_3m']:+.1f}%")
            print(f"   📊 Volatility: {analysis['realized_volatility']:.1f}%")
            print(f"   📊 Score: {analysis['momentum_score']:.2f}")
            
            # Generate signals for each strategy
            strategies_signals = []
            
            # Momentum calls
            call_signal = self.generate_momentum_call_signals(analysis)
            if call_signal:
                strategies_signals.append(call_signal)
                print(f"   ✅ {call_signal['signal']} (Strength: {call_signal['strength']:.0f}%)")
            
            # Momentum spreads
            spread_signal = self.generate_momentum_spread_signals(analysis)
            if spread_signal:
                strategies_signals.append(spread_signal)
                print(f"   ✅ {spread_signal['signal']} (Strength: {spread_signal['strength']:.0f}%)")
            
            # Momentum straddles
            straddle_signal = self.generate_momentum_straddle_signals(analysis)
            if straddle_signal:
                strategies_signals.append(straddle_signal)
                print(f"   ✅ {straddle_signal['signal']} (Strength: {straddle_signal['strength']:.0f}%)")
            
            if not strategies_signals:
                print(f"   🟡 No options signals for {symbol}")
            
            all_signals.extend(strategies_signals)
        
        return all_signals
    
    def rank_options_opportunities(self, signals):
        """Rank and prioritize options opportunities"""
        
        if not signals:
            return []
        
        # Score each signal
        for signal in signals:
            # Base score from strength
            score = signal['strength']
            
            # Adjust for risk level
            risk_multipliers = {'LOW': 1.2, 'MEDIUM': 1.0, 'HIGH': 0.8}
            score *= risk_multipliers.get(signal['risk_level'], 1.0)
            
            # Adjust for strategy type preference
            strategy_preference = {'MOMENTUM_CALLS': 1.1, 'MOMENTUM_SPREADS': 1.0, 'MOMENTUM_STRADDLES': 0.9}
            score *= strategy_preference.get(signal['strategy'], 1.0)
            
            signal['final_score'] = score
        
        # Sort by final score
        return sorted(signals, key=lambda x: x['final_score'], reverse=True)
    
    def create_options_trading_plan(self):
        """Create comprehensive options trading plan"""
        
        # Generate all signals
        signals = self.generate_options_portfolio()
        
        # Rank opportunities
        ranked_signals = self.rank_options_opportunities(signals)
        
        print(f"\n🎯 OPTIONS TRADING PLAN")
        print("=" * 30)
        
        if not ranked_signals:
            print("❌ No options opportunities found")
            return None
        
        print(f"📊 Found {len(ranked_signals)} options opportunities")
        print(f"\n🏆 TOP 5 OPPORTUNITIES:")
        print("-" * 25)
        
        for i, signal in enumerate(ranked_signals[:5], 1):
            print(f"{i}. {signal['symbol']} - {signal['signal']}")
            print(f"   Strategy: {signal['strategy']}")
            print(f"   Score: {signal['final_score']:.1f}")
            print(f"   Risk: {signal['risk_level']}")
            print(f"   Reasoning: {signal['reasoning']}")
            print(f"   Duration: {signal['expected_duration']}")
            print()
        
        # Save to file
        options_plan = {
            'timestamp': datetime.now().isoformat(),
            'total_signals': len(ranked_signals),
            'top_opportunities': ranked_signals[:10],
            'strategy_breakdown': self._analyze_strategy_breakdown(ranked_signals)
        }
        
        filename = f"momentum_options_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(filename, 'w') as f:
            json.dump(options_plan, f, indent=2, default=str)
        
        print(f"💾 Options plan saved: {filename}")
        
        return options_plan
    
    def _analyze_strategy_breakdown(self, signals):
        """Analyze breakdown of strategies"""
        
        breakdown = {}
        for signal in signals:
            strategy = signal['strategy']
            if strategy not in breakdown:
                breakdown[strategy] = {'count': 0, 'avg_score': 0, 'symbols': []}
            
            breakdown[strategy]['count'] += 1
            breakdown[strategy]['avg_score'] += signal['final_score']
            breakdown[strategy]['symbols'].append(signal['symbol'])
        
        # Calculate averages
        for strategy in breakdown:
            breakdown[strategy]['avg_score'] /= breakdown[strategy]['count']
        
        return breakdown

def main():
    """Run momentum-enhanced options analysis"""
    
    print("🚀 MOMENTUM-ENHANCED OPTIONS TRADING")
    print("=" * 40)
    print("📈 Leveraging institutional momentum signals")
    print("🎯 Creating options strategies for momentum stocks")
    print("=" * 40)
    
    # Initialize options trader
    options_trader = MomentumOptionsTrader()
    
    # Create trading plan
    trading_plan = options_trader.create_options_trading_plan()
    
    print(f"\n✅ OPTIONS ANALYSIS COMPLETE!")
    print(f"🎯 Ready to execute momentum-based options trades")
    
    return options_trader, trading_plan

if __name__ == "__main__":
    options_trader, trading_plan = main()
