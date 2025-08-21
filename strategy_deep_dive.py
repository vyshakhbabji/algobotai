#!/usr/bin/env python3
"""
Strategy Performance Deep Dive Analysis
Understanding why returns are low and how to improve them
"""
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def analyze_market_conditions():
    """Analyze market conditions during the forward test period"""
    
    print("🔍 MARKET CONDITIONS ANALYSIS")
    print("=" * 60)
    
    # Forward test period: March 30, 2025 → August 12, 2025
    start_date = "2025-03-30"
    end_date = "2025-08-12"
    
    # Get SPY data for the period
    try:
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
        
        if not spy.empty:
            start_price = spy['Close'].iloc[0]
            end_price = spy['Close'].iloc[-1]
            market_return = (end_price - start_price) / start_price * 100
            
            # Calculate volatility
            spy['Daily_Return'] = spy['Close'].pct_change()
            volatility = spy['Daily_Return'].std() * (252**0.5) * 100  # Annualized
            
            # Max drawdown
            cumulative = (1 + spy['Daily_Return']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            print(f"📊 SPY Performance ({start_date} to {end_date}):")
            print(f"   Start Price: ${start_price:.2f}")
            print(f"   End Price: ${end_price:.2f}")
            print(f"   Market Return: {market_return:+.2f}%")
            print(f"   Volatility: {volatility:.1f}% (annualized)")
            print(f"   Max Drawdown: {max_drawdown:.2f}%")
            
            # Market characterization
            if abs(market_return) < 2:
                market_type = "SIDEWAYS/CHOPPY"
            elif market_return > 5:
                market_type = "BULL MARKET"
            elif market_return < -5:
                market_type = "BEAR MARKET"
            else:
                market_type = "TRENDING"
            
            print(f"   Market Type: {market_type}")
            
            return {
                'market_return': market_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'market_type': market_type
            }
        
    except Exception as e:
        print(f"❌ Could not fetch market data: {e}")
        return None

def analyze_strategy_limitations():
    """Analyze why the strategy is producing low returns"""
    
    print(f"\n🔧 STRATEGY LIMITATIONS ANALYSIS")
    print("=" * 60)
    
    # Load the latest results
    try:
        with open('two_year_batch/batch_results.json', 'r') as f:
            results = json.load(f)
        
        runs = results.get('runs', {})
        
        # Analyze exposure patterns
        exposures = []
        returns = []
        capture_ratios = []
        
        for symbol, metrics in runs.items():
            if 'error' not in metrics:
                exposure = metrics.get('exposure_avg', 0)
                return_pct = metrics.get('return_pct', 0)
                capture = metrics.get('capture', 0)
                
                exposures.append(exposure)
                returns.append(return_pct)
                capture_ratios.append(capture)
                
                print(f"   {symbol}: {exposure:.1%} exposure, {return_pct:+.3f}% return, {capture:.1%} capture")
        
        avg_exposure = sum(exposures) / len(exposures) if exposures else 0
        avg_capture = sum(capture_ratios) / len(capture_ratios) if capture_ratios else 0
        
        print(f"\n📈 Key Issues Identified:")
        print(f"   • Average Exposure: {avg_exposure:.1%} (target: >70%)")
        print(f"   • Average Capture Ratio: {avg_capture:.1%} (target: >50%)")
        
        if avg_exposure < 0.3:
            print(f"   🔴 CRITICAL: Very low market exposure")
            print(f"      → Strategy is barely trading")
            print(f"      → Thresholds may be too high")
        
        if avg_capture < 0.3:
            print(f"   🔴 CRITICAL: Poor market capture")
            print(f"      → Missing market moves")
            print(f"      → Entry/exit timing issues")
        
        return {
            'avg_exposure': avg_exposure,
            'avg_capture': avg_capture,
            'issues': []
        }
        
    except Exception as e:
        print(f"❌ Could not analyze strategy: {e}")
        return None

def recommend_fixes():
    """Recommend specific fixes to boost returns"""
    
    print(f"\n💡 RECOMMENDED FIXES FOR HIGHER RETURNS")
    print("=" * 60)
    
    print(f"🎯 IMMEDIATE ACTIONS:")
    print(f"   1. LOWER ENTRY THRESHOLDS:")
    print(f"      • Current tier_probs: (0.42,0.47,0.52,0.58)")
    print(f"      • Suggest: (0.35,0.40,0.45,0.52)")
    print(f"      • Current breakout_prob: 0.42")
    print(f"      • Suggest: 0.32")
    
    print(f"\n   2. INCREASE RISK TOLERANCE:")
    print(f"      • Current risk_per_trade: 12%")
    print(f"      • Suggest: 20%")
    print(f"      • Current risk_ceiling: 15%")
    print(f"      • Suggest: 25%")
    
    print(f"\n   3. FASTER SCALING:")
    print(f"      • Current fast_scale_threshold: 4%")
    print(f"      • Suggest: 2%")
    print(f"      • Enable immediate position doubling on 2% gains")
    
    print(f"\n   4. LONGER HOLDING PERIODS:")
    print(f"      • Current min_holding_days: 1")
    print(f"      • Suggest: 0 (same-day exits allowed)")
    print(f"      • Enable intraday scalping")
    
    print(f"\n   5. REMOVE PROFIT TAKING:")
    print(f"      • Current profit_ladder: (8%,18%,35%,60%)")
    print(f"      • Suggest: (15%,40%,80%) - let winners run")
    
    print(f"\n🔮 ALTERNATIVE STRATEGIES:")
    print(f"   • Momentum/Breakout Strategy:")
    print(f"     → Target stocks moving >5% in 1-2 days")
    print(f"     → Use 25% position sizes")
    print(f"     → 2-3% stop losses, no profit targets")
    
    print(f"\n   • Options Enhancement:")
    print(f"     → Add leveraged options trades")
    print(f"     → Target 50-200% returns per trade")
    print(f"     → Use 5-10% of capital per options trade")
    
    print(f"\n   • Sector Rotation:")
    print(f"     → Focus on strongest sectors each month")
    print(f"     → Ignore weak sectors completely")
    print(f"     → 30-40% concentration in hot sectors")

def create_extreme_config():
    """Create an extreme configuration for maximum returns"""
    
    extreme_config = {
        "name": "EXTREME Maximum Return Strategy",
        "warning": "VERY HIGH RISK - Use only small capital for testing",
        "parameters": {
            "risk_per_trade_pct": 0.25,           # 25% per trade
            "risk_ceiling": 0.30,                 # 30% max
            "tier_probs": [0.30, 0.35, 0.40, 0.47], # Much lower thresholds
            "breakout_prob": 0.28,                # 28% breakout
            "hard_exit_prob": 0.25,               # 25% exit
            "profit_ladder": [0.20, 0.50, 1.00], # Let winners run to 100%
            "profit_trim_fractions": [0.02, 0.05, 0.10], # Tiny trims
            "fast_scale_gain_threshold": 0.015,   # 1.5% for doubling
            "min_holding_days": 0,                # Same day exits
            "atr_initial_mult": 0.8,              # Very tight stops
            "atr_trail_mult": 1.2,                # Tight trailing
            "momentum_20d_threshold": 0.02,       # 2% momentum
            "enable_leverage": True,              # If available
            "max_portfolio_concentration": 0.90,   # 90% in best ideas
            "enable_aggressive_scaling": True,
            "double_down_on_gain": 0.02,          # Double position on 2% gain
            "max_position_size": 0.40             # 40% single position
        }
    }
    
    with open('extreme_config.json', 'w') as f:
        json.dump(extreme_config, f, indent=2)
    
    print(f"\n📁 EXTREME CONFIG SAVED: extreme_config.json")
    print(f"⚠️  WARNING: This config can lead to >50% drawdowns")
    print(f"🎯 TARGET: 2-10% returns in 3-month period")

def main():
    market_analysis = analyze_market_conditions()
    strategy_analysis = analyze_strategy_limitations()
    recommend_fixes()
    create_extreme_config()
    
    print(f"\n🏁 SUMMARY & NEXT STEPS")
    print("=" * 60)
    
    if market_analysis and abs(market_analysis['market_return']) < 2:
        print(f"📊 Market was {market_analysis['market_type']} ({market_analysis['market_return']:+.2f}%)")
        print(f"   → Sideways markets are challenging for momentum strategies")
        print(f"   → Consider market-neutral or volatility strategies")
    
    if strategy_analysis and strategy_analysis['avg_exposure'] < 0.3:
        print(f"🔧 Strategy barely traded ({strategy_analysis['avg_exposure']:.1%} exposure)")
        print(f"   → CRITICAL: Lower entry thresholds immediately")
        print(f"   → Target 60-80% market exposure")
    
    print(f"\n🎯 IMMEDIATE ACTION PLAN:")
    print(f"   1. Use extreme_config.json parameters")
    print(f"   2. Test on single high-momentum stock first")
    print(f"   3. Monitor for 50%+ exposure levels")
    print(f"   4. Expect higher volatility but better returns")
    
    print(f"\n⚡ QUICK TEST COMMAND:")
    print(f"   python3 ultra_aggressive_modifier.py --create")
    print(f"   # Manually edit with extreme parameters")
    print(f"   python -m algobot.portfolio.two_year_batch_runner --topk 1 --workers 1 --aggressive")

if __name__ == "__main__":
    main()
