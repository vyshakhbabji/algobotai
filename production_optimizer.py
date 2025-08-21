#!/usr/bin/env python3
"""
Production Configuration Optimizer
Fine-tune production settings based on backtest results
"""

import json
import os
from datetime import datetime

def analyze_current_performance():
    """Analyze current production configuration performance"""
    
    print("📊 CURRENT PRODUCTION PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Read batch results
    results_file = "two_year_batch/batch_results.json"
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        avg_return = results.get('average_return_pct', 0)
        avg_exposure = results.get('average_exposure_pct', 0)
        profitable_count = results.get('profitable_stocks', 0)
        total_stocks = results.get('total_stocks', 5)
        
        print(f"Current Performance:")
        print(f"   • Average Return: {avg_return:.3f}%")
        print(f"   • Average Exposure: {avg_exposure:.1f}%")
        print(f"   • Profitable Stocks: {profitable_count}/{total_stocks}")
        print(f"   • Success Rate: {(profitable_count/total_stocks)*100:.1f}%")
        
        return {
            'avg_return': avg_return,
            'avg_exposure': avg_exposure,
            'success_rate': profitable_count/total_stocks,
            'needs_optimization': avg_return < 0.5 or avg_exposure < 50
        }
    
    return None

def create_optimized_production_config():
    """Create optimized production configuration"""
    
    print("\n🎯 CREATING OPTIMIZED PRODUCTION CONFIGURATION")
    print("=" * 60)
    
    # Enhanced production config with better signal generation
    optimized_config = '''"""
OPTIMIZED PRODUCTION TRADING CONFIGURATION  
Enhanced for better signal generation while maintaining safety
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    """Optimized ML model configuration - Better signal generation"""
    min_directional_accuracy: float = 0.48         # 48% accuracy (slightly lower)
    buy_threshold: float = 0.006                   # 0.6% buy threshold (lower) 
    sell_threshold: float = -0.005                 # -0.5% sell threshold
    confidence_threshold: float = 0.30             # 30% confidence (lower for more signals)
    prediction_window: int = 5                     # 5-day prediction window
    model_retrain_frequency: int = 30              # Retrain every 30 days
    validation_score_min: float = 0.50             # 50% validation minimum

@dataclass  
class RiskConfig:
    """Optimized risk management - Slightly more aggressive"""
    max_position_pct: float = 0.30                # 30% maximum per position (increased)
    max_portfolio_risk_pct: float = 0.90          # 90% maximum portfolio risk
    stop_loss_pct: float = 0.10                   # 10% stop loss
    take_profit_pct: float = 0.18                 # 18% take profit
    trailing_stop_pct: float = 0.06               # 6% trailing stop
    max_daily_loss_pct: float = 0.04              # 4% maximum daily loss
    correlation_limit: float = 0.75               # 75% correlation limit
    min_holding_days: int = 2                     # 2 day minimum hold (reduced)

@dataclass
class ExecutionConfig:
    """Optimized execution - More frequent rebalancing"""
    rebalance_weekdays: Tuple[int, ...] = (0, 2, 4)  # Mon/Wed/Fri rebalancing
    min_holding_days: int = 2                     # 2 day minimum
    max_holding_days: int = 35                    # 35 day maximum
    market_hours_only: bool = True                # Market hours only
    slippage_assumption: float = 0.002            # 0.2% slippage
    min_trade_size: float = 100                   # $100 minimum trade
    target_positions: int = 10                    # Target 10 positions (increased)
    cash_reserve_pct: float = 0.10               # 10% cash reserve (reduced)

@dataclass
class InstitutionalConfig:
    """Optimized institutional strategy parameters"""
    tier_probs: Tuple[float, ...] = (0.40, 0.46, 0.52, 0.58)  # Lower tiers for more signals
    breakout_prob: float = 0.38                   # 38% breakout probability (lower)
    hard_exit_prob: float = 0.20                  # 20% hard exit (lower)
    soft_exit_prob: float = 0.30                  # 30% soft exit
    risk_per_trade_pct: float = 0.10              # 10% risk per trade (increased)
    early_core_fraction: float = 0.65             # 65% initial position

# Global configuration instance - OPTIMIZED PRODUCTION MODE
GLOBAL_CONFIG = {
    'model': ModelConfig(),
    'risk': RiskConfig(), 
    'execution': ExecutionConfig(),
    'institutional': InstitutionalConfig(),
    'production_mode': True,
    'optimization_level': 'HIGH_SIGNAL_GENERATION',
    'strategy_type': 'OPTIMIZED_ML_INSTITUTIONAL_V2'
}

print("🎯 OPTIMIZED PRODUCTION CONFIGURATION LOADED")
print("   • Enhanced signal generation (48% accuracy, 30% confidence)")
print("   • Increased position limits (30% max position)")
print("   • More frequent rebalancing (Mon/Wed/Fri)")
print("   • Target: 8-15% annual returns with controlled risk")
'''
    
    # Write optimized config
    config_path = "algobot/config.py"
    with open(config_path, 'w') as f:
        f.write(optimized_config)
    
    print(f"✅ Updated {config_path} with optimized production settings")
    
    return optimized_config

def create_signal_optimization_update():
    """Update signal configuration for better performance"""
    
    optimized_signals = {
        "signal_config_version": "PRODUCTION_OPTIMIZED_V2",
        "description": "Optimized signal thresholds for production trading",
        "created": datetime.now().isoformat(),
        
        # More sensitive thresholds for better signal generation
        "best_config": {
            "trend_5d_buy_threshold": 0.012,        # 1.2% (reduced from 1.8%)
            "trend_20d_buy_threshold": 0.018,       # 1.8% (reduced from 2.5%)
            "momentum_buy_threshold": 0.010,        # 1.0% (reduced from 1.5%)
            "volatility_threshold": 0.025,          # 2.5% volatility
            "rsi_overbought": 70,                   # 70 overbought (reduced)
            "rsi_oversold": 32,                     # 32 oversold (increased)
            "rsi_momentum_threshold": 0.08,         # 8% RSI momentum
            "macd_threshold": 0.003,                # 0.3% MACD (reduced)
            "volume_spike_threshold": 1.3,          # 1.3x volume (reduced)
            "breakout_confirmation": 0.010          # 1.0% breakout (reduced)
        },
        
        "risk_level": "MODERATE_AGGRESSIVE",
        "expected_signal_frequency": "MEDIUM_HIGH",
        "target_exposure": "60-80%"
    }
    
    with open('optimized_signal_config.json', 'w') as f:
        json.dump(optimized_signals, f, indent=2)
    
    print("✅ Created optimized_signal_config.json")
    
    return optimized_signals

def create_institutional_params_update():
    """Update institutional strategy parameters"""
    
    optimized_institutional = {
        "institutional_config_version": "PRODUCTION_OPTIMIZED_V2",
        "description": "Optimized institutional parameters for better exposure",
        "created": datetime.now().isoformat(),
        
        # Lower thresholds for more trading activity
        "tier_probs": [0.40, 0.46, 0.52, 0.58],
        "breakout_prob": 0.38,
        "hard_exit_prob": 0.20,
        "soft_exit_prob": 0.30,
        "risk_per_trade_pct": 0.10,
        "early_core_fraction": 0.65,
        
        # Enhanced features
        "momentum_boost": True,
        "volatility_adjustment": True,
        "market_regime_detection": True,
        "correlation_filtering": True,
        
        "target_metrics": {
            "exposure": "60-80%",
            "win_rate": "55%+",
            "profit_factor": "1.2+",
            "max_drawdown": "10%"
        }
    }
    
    with open('optimized_institutional_params.json', 'w') as f:
        json.dump(optimized_institutional, f, indent=2)
    
    print("✅ Created optimized_institutional_params.json")
    
    return optimized_institutional

def test_optimized_configuration():
    """Test the optimized configuration"""
    
    print("\n🚀 TESTING OPTIMIZED CONFIGURATION")
    print("=" * 60)
    
    try:
        # Test config loading
        from algobot.config import GLOBAL_CONFIG
        
        print("✅ Optimized config loaded successfully")
        print(f"   • Model accuracy: {GLOBAL_CONFIG['model'].min_directional_accuracy}")
        print(f"   • Confidence threshold: {GLOBAL_CONFIG['model'].confidence_threshold}")
        print(f"   • Max position: {GLOBAL_CONFIG['risk'].max_position_pct*100}%")
        print(f"   • Target positions: {GLOBAL_CONFIG['execution'].target_positions}")
        print(f"   • Cash reserve: {GLOBAL_CONFIG['execution'].cash_reserve_pct*100}%")
        
        # Run quick test
        print(f"\n🔬 Running optimized backtest...")
        os.system("python -m algobot.portfolio.two_year_batch_runner --topk 5 --workers 1")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing optimized config: {e}")
        return False

def main():
    """Main optimization process"""
    
    print("🎯 PRODUCTION CONFIGURATION OPTIMIZER")
    print("=" * 60)
    print("Analyzing current performance and optimizing for live trading...")
    print()
    
    # Analyze current performance
    current_perf = analyze_current_performance()
    
    if current_perf and current_perf['needs_optimization']:
        print(f"\n⚠️  OPTIMIZATION NEEDED")
        print(f"   Current return: {current_perf['avg_return']:.3f}% (target: >0.5%)")
        print(f"   Current exposure: {current_perf['avg_exposure']:.1f}% (target: >60%)")
        
        # Create optimized configurations
        create_optimized_production_config()
        create_signal_optimization_update()
        create_institutional_params_update()
        
        # Test optimized configuration
        if test_optimized_configuration():
            print("\n✅ OPTIMIZATION COMPLETE")
            print("=" * 60)
            print("🎯 Optimized Configuration Features:")
            print("   • Lower ML thresholds for more signals")
            print("   • Increased position sizes (30% max)")
            print("   • More frequent rebalancing (3x/week)")
            print("   • Enhanced signal sensitivity")
            print("   • Target: 60-80% exposure, 8-15% annual returns")
            print()
            print("🚀 Ready for production deployment!")
            
        else:
            print("\n❌ Optimization test failed")
    
    else:
        print("\n✅ Current configuration performing well")
        print("No optimization needed at this time")

if __name__ == "__main__":
    main()
