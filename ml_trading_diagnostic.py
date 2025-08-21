#!/usr/bin/env python3
"""
ML Model Trading Diagnostic
Final diagnosis of why the ML model isn't generating trades
"""
import json
import os

def diagnose_ml_trading_issue():
    """Diagnose why ML model isn't trading despite aggressive settings"""
    
    print("🔍 ML MODEL TRADING DIAGNOSTIC")
    print("=" * 70)
    
    # Check if there are individual result files to understand what happened
    batch_dir = "two_year_batch"
    
    if os.path.exists(batch_dir):
        files = [f for f in os.listdir(batch_dir) if f.endswith('_result.json')]
        
        print(f"📁 Found {len(files)} individual result files")
        
        # Check a few key files
        key_stocks = ['t_result.json', 'nflx_result.json', 'pm_result.json']
        
        for stock_file in key_stocks:
            if stock_file in files:
                try:
                    with open(os.path.join(batch_dir, stock_file), 'r') as f:
                        stock_data = json.load(f)
                    
                    print(f"\n📊 {stock_file.replace('_result.json', '').upper()} Analysis:")
                    
                    # Look for key metrics
                    if 'trades' in stock_data:
                        print(f"   Trades executed: {len(stock_data['trades'])}")
                    
                    if 'daily_analysis' in stock_data:
                        daily_data = stock_data['daily_analysis']
                        print(f"   Daily analysis entries: {len(daily_data) if daily_data else 0}")
                    
                    if 'signal_stats' in stock_data:
                        signals = stock_data['signal_stats']
                        print(f"   Signal statistics: {signals}")
                    
                    if 'exposure_history' in stock_data:
                        exposure = stock_data['exposure_history']
                        non_zero_days = sum(1 for e in exposure if e > 0) if exposure else 0
                        print(f"   Days with exposure: {non_zero_days}/{len(exposure) if exposure else 0}")
                    
                    # Look for reasons why no trading occurred
                    if 'no_trade_reasons' in stock_data:
                        print(f"   No-trade reasons: {stock_data['no_trade_reasons']}")
                    
                    # Check confidence scores
                    if 'confidence_history' in stock_data:
                        conf_hist = stock_data['confidence_history']
                        if conf_hist:
                            avg_conf = sum(conf_hist) / len(conf_hist)
                            max_conf = max(conf_hist)
                            print(f"   Confidence: avg={avg_conf:.3f}, max={max_conf:.3f}")
                    
                    print(f"   Raw data keys: {list(stock_data.keys())[:10]}...")  # Show first 10 keys
                    
                except Exception as e:
                    print(f"   ❌ Error reading {stock_file}: {e}")

def check_configuration_conflicts():
    """Check if there are configuration conflicts preventing trading"""
    
    print(f"\n🔧 CONFIGURATION CONFLICT CHECK")
    print("=" * 70)
    
    # Check if the aggressive config files exist and are being used
    config_files = [
        'aggressive_ml_config.json',
        'aggressive_institutional_params.json', 
        'best_signal_config.json',
        'algobot/config.py'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file} exists")
            if config_file.endswith('.json'):
                try:
                    with open(config_file, 'r') as f:
                        data = json.load(f)
                    print(f"   Key parameters: {list(data.keys())[:5]}...")
                except:
                    print(f"   ❌ Could not read {config_file}")
        else:
            print(f"❌ {config_file} missing")

def suggest_emergency_fixes():
    """Suggest emergency fixes to force trading"""
    
    print(f"\n🚨 EMERGENCY FIXES TO FORCE TRADING")
    print("=" * 70)
    
    print(f"1. FORCE MINIMUM EXPOSURE:")
    print(f"   • Override ML model with guaranteed 50% exposure")
    print(f"   • Use buy-and-hold fallback if no signals")
    print(f"   • Implement minimum position size requirements")
    
    print(f"\n2. DISABLE ML MODEL TEMPORARILY:")
    print(f"   • Use pure technical analysis (RSI, MACD, momentum)")
    print(f"   • Simple strategy: buy if price > 20-day MA")
    print(f"   • This would guarantee some trading activity")
    
    print(f"\n3. LOWER ALL THRESHOLDS TO ZERO:")
    print(f"   • Set tier_probs to (0.05, 0.10, 0.15, 0.20)")
    print(f"   • Set breakout_prob to 0.05")
    print(f"   • This should force trading on any small signal")
    
    print(f"\n4. IMPLEMENT RANDOM TRADING:")
    print(f"   • If no ML signals for 5 days, buy randomly")
    print(f"   • Use 20% positions with 5% stop losses")
    print(f"   • At least this would capture market moves")

def create_forced_trading_config():
    """Create a configuration that forces trading regardless of signals"""
    
    forced_config = {
        "name": "FORCED_TRADING_CONFIG",
        "description": "Emergency config to force trading when ML fails",
        "tier_probs": [0.01, 0.05, 0.10, 0.15],  # Almost zero thresholds
        "breakout_prob": 0.01,                    # 1% breakout
        "hard_exit_prob": 0.01,                   # 1% exit
        "soft_exit_prob": 0.05,                   # 5% soft exit
        "risk_per_trade_pct": 0.20,               # 20% risk
        "early_core_fraction": 0.95,              # 95% initial position
        "force_min_exposure": 0.50,               # Force 50% minimum exposure
        "buy_hold_fallback": True,                # Buy and hold if no signals
        "random_trade_after_days": 3,             # Random trading after 3 days of no signals
        "override_ml_confidence": True,           # Ignore ML confidence requirements
    }
    
    with open('forced_trading_config.json', 'w') as f:
        json.dump(forced_config, f, indent=2)
    
    print(f"\n📁 CREATED: forced_trading_config.json")
    print(f"This config should force trading even if ML model fails")

def main():
    diagnose_ml_trading_issue()
    check_configuration_conflicts()
    suggest_emergency_fixes()
    create_forced_trading_config()
    
    print(f"\n🎯 DIAGNOSIS SUMMARY:")
    print("=" * 70)
    print(f"The ML model appears to be fundamentally not generating")
    print(f"signals that meet even the most aggressive thresholds.")
    print(f"This suggests one of these issues:")
    print(f"")
    print(f"1. 🧠 ML MODEL TRAINING ISSUE")
    print(f"   • Model may not be training properly")
    print(f"   • Predictions may be all neutral/low confidence")
    print(f"   • Need to debug the ML pipeline directly")
    print(f"")
    print(f"2. 📊 DATA QUALITY ISSUE")
    print(f"   • Training data may be insufficient or corrupted")
    print(f"   • Features may not be calculated correctly")
    print(f"   • Need to validate input data")
    print(f"")
    print(f"3. ⚙️ CONFIGURATION OVERRIDE ISSUE")
    print(f"   • Some other config may be overriding aggressive settings")
    print(f"   • Risk management may be blocking all trades")
    print(f"   • Need to trace through the execution logic")
    print(f"")
    print(f"🚨 IMMEDIATE RECOMMENDATION:")
    print(f"Implement a buy-and-hold fallback strategy that activates")
    print(f"when the ML model generates no trades for 2+ days.")
    print(f"This would at least capture market performance (0.48%)")
    print(f"while debugging the ML model issues.")

if __name__ == "__main__":
    main()
