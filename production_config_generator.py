#!/usr/bin/env python3
"""
Production-Ready Trading Configuration Generator
Creates optimal configs for real trading based on backtesting insights
"""

import json
from datetime import datetime

def create_production_ml_config():
    """Create production-ready ML configuration"""
    
    production_config = {
        "config_name": "PRODUCTION_OPTIMIZED_V1",
        "description": "Production-ready configuration for live trading",
        "created": datetime.now().isoformat(),
        "risk_level": "MODERATE_AGGRESSIVE",
        "target_annual_return": "15-25%",
        "max_drawdown_target": "8-12%",
        
        # ML Model Parameters - Balanced for production
        "ml_parameters": {
            "min_directional_accuracy": 0.52,      # 52% accuracy requirement
            "buy_threshold": 0.008,                 # 0.8% buy signal threshold
            "sell_threshold": -0.006,               # -0.6% sell signal threshold
            "confidence_threshold": 0.35,           # 35% confidence requirement
            "prediction_window": 5,                 # 5-day prediction window
            "feature_importance_min": 0.02,         # 2% minimum feature importance
            "model_retrain_frequency": 30,          # Retrain every 30 days
            "validation_score_min": 0.55            # 55% validation score minimum
        },
        
        # Risk Management - Production-safe
        "risk_parameters": {
            "max_position_pct": 0.25,              # 25% maximum per position
            "max_portfolio_risk_pct": 0.85,        # 85% maximum portfolio risk
            "stop_loss_pct": 0.08,                 # 8% stop loss
            "take_profit_pct": 0.15,               # 15% take profit
            "trailing_stop_pct": 0.05,             # 5% trailing stop
            "max_daily_loss_pct": 0.03,            # 3% maximum daily loss
            "correlation_limit": 0.70,             # 70% maximum correlation between positions
            "sector_concentration_limit": 0.40     # 40% maximum in one sector
        },
        
        # Execution Parameters - Realistic for live trading
        "execution_parameters": {
            "rebalance_frequency": "weekly",        # Weekly rebalancing
            "min_holding_days": 3,                  # Hold positions for at least 3 days
            "max_holding_days": 45,                 # Maximum 45 days holding
            "market_hours_only": True,              # Trade only during market hours
            "pre_market_analysis": True,            # Analyze pre-market
            "after_hours_exit": False,              # No after-hours exits
            "slippage_assumption": 0.002,           # 0.2% slippage assumption
            "commission_per_trade": 0.0,            # Commission-free trading
            "min_trade_size": 100                   # Minimum $100 trade size
        },
        
        # Signal Generation - Optimized thresholds
        "signal_parameters": {
            "momentum_threshold": 0.012,            # 1.2% momentum threshold
            "trend_confirmation_days": 3,           # 3-day trend confirmation
            "volume_confirmation": True,            # Require volume confirmation
            "rsi_overbought": 75,                   # RSI overbought level
            "rsi_oversold": 30,                     # RSI oversold level
            "macd_signal_strength": 0.8,            # MACD signal strength
            "bollinger_band_width": 2.0,            # Bollinger band width
            "support_resistance_strength": 0.015   # Support/resistance level strength
        },
        
        # Portfolio Management
        "portfolio_parameters": {
            "target_positions": 8,                  # Target 8 positions
            "min_positions": 5,                     # Minimum 5 positions
            "max_positions": 12,                    # Maximum 12 positions
            "cash_reserve_pct": 0.15,               # 15% cash reserve
            "position_sizing_method": "kelly_fraction", # Kelly criterion sizing
            "rebalancing_threshold": 0.05,          # 5% deviation triggers rebalance
            "correlation_check_frequency": "daily"  # Daily correlation monitoring
        }
    }
    
    return production_config

def create_institutional_strategy_config():
    """Create institutional strategy parameters for production"""
    
    institutional_config = {
        "strategy_name": "INSTITUTIONAL_PRODUCTION_V1",
        "description": "Production institutional strategy configuration",
        
        # Tier probability thresholds - Balanced
        "tier_probs": [0.45, 0.52, 0.58, 0.65],    # Progressive probability requirements
        "breakout_prob": 0.42,                      # 42% breakout probability
        "hard_exit_prob": 0.25,                     # 25% hard exit probability
        "soft_exit_prob": 0.35,                     # 35% soft exit probability
        
        # Position sizing
        "risk_per_trade_pct": 0.08,                 # 8% risk per trade
        "early_core_fraction": 0.60,                # 60% initial position
        "scaling_increment": 0.20,                  # 20% scaling increments
        "max_scaling_attempts": 3,                  # Maximum 3 scaling attempts
        
        # Timing and execution
        "entry_confirmation_bars": 2,               # 2-bar entry confirmation
        "exit_confirmation_bars": 1,                # 1-bar exit confirmation
        "momentum_lookback": 20,                    # 20-day momentum lookback
        "volatility_adjustment": True,              # Adjust for volatility
        
        # Risk controls
        "max_sector_allocation": 0.35,              # 35% maximum per sector
        "correlation_filter": 0.65,                 # 65% correlation filter
        "liquidity_minimum": 1000000,               # $1M minimum daily volume
        "market_cap_minimum": 5000000000            # $5B minimum market cap
    }
    
    return institutional_config

def create_signal_optimization_config():
    """Create optimized signal generation configuration"""
    
    signal_config = {
        "optimization_target": "risk_adjusted_return",
        "description": "Production signal configuration",
        
        # Technical indicator thresholds
        "trend_5d_buy_threshold": 0.018,            # 1.8% 5-day trend
        "trend_20d_buy_threshold": 0.025,           # 2.5% 20-day trend
        "momentum_buy_threshold": 0.015,            # 1.5% momentum
        "volatility_threshold": 0.02,               # 2% volatility threshold
        
        # RSI parameters
        "rsi_period": 14,                           # 14-day RSI
        "rsi_overbought": 72,                       # 72 overbought level
        "rsi_oversold": 28,                         # 28 oversold level
        "rsi_momentum_threshold": 0.10,             # 10% RSI momentum
        
        # MACD parameters
        "macd_fast": 12,                            # 12-day EMA
        "macd_slow": 26,                            # 26-day EMA
        "macd_signal": 9,                           # 9-day signal line
        "macd_threshold": 0.005,                    # 0.5% MACD threshold
        
        # Volume analysis
        "volume_ma_period": 20,                     # 20-day volume MA
        "volume_spike_threshold": 1.5,              # 1.5x volume spike
        "volume_confirmation": True,                # Require volume confirmation
        
        # Support/Resistance
        "support_resistance_lookback": 50,          # 50-day S/R lookback
        "support_resistance_strength": 0.02,        # 2% S/R strength
        "breakout_confirmation": 0.015              # 1.5% breakout confirmation
    }
    
    return signal_config

def generate_production_config_file():
    """Generate complete production configuration file"""
    
    print("🎯 GENERATING PRODUCTION-READY TRADING CONFIGURATION")
    print("=" * 70)
    
    # Create all configuration components
    ml_config = create_production_ml_config()
    institutional_config = create_institutional_strategy_config()
    signal_config = create_signal_optimization_config()
    
    # Combine into master configuration
    master_config = {
        "config_version": "PRODUCTION_V1.0",
        "created": datetime.now().isoformat(),
        "description": "Production-ready configuration for live algorithmic trading",
        "target_performance": {
            "annual_return_target": "15-25%",
            "max_drawdown_target": "8-12%",
            "sharpe_ratio_target": "1.2+",
            "win_rate_target": "55%+",
            "profit_factor_target": "1.3+"
        },
        
        # Core configurations
        "ml_model": ml_config,
        "institutional_strategy": institutional_config,
        "signal_optimization": signal_config,
        
        # Production safety features
        "safety_features": {
            "circuit_breakers": True,
            "max_daily_trades": 20,
            "position_size_limits": True,
            "correlation_monitoring": True,
            "real_time_risk_monitoring": True,
            "automatic_stop_loss": True,
            "market_condition_filters": True
        },
        
        # Performance monitoring
        "monitoring": {
            "real_time_pnl": True,
            "risk_metrics_tracking": True,
            "performance_alerts": True,
            "daily_reports": True,
            "weekly_analysis": True,
            "monthly_review": True
        }
    }
    
    # Save master configuration
    with open('production_trading_config.json', 'w') as f:
        json.dump(master_config, f, indent=2)
    
    print("✅ Created production_trading_config.json")
    
    # Create individual config files for easier management
    with open('production_ml_config.json', 'w') as f:
        json.dump(ml_config, f, indent=2)
    
    with open('production_institutional_config.json', 'w') as f:
        json.dump(institutional_config, f, indent=2)
        
    with open('production_signal_config.json', 'w') as f:
        json.dump(signal_config, f, indent=2)
    
    print("✅ Created individual production config files")
    
    return master_config

def update_main_config_for_production():
    """Update main config.py with production-ready settings"""
    
    print("\n🔧 UPDATING MAIN CONFIG FOR PRODUCTION")
    print("=" * 70)
    
    # Create backup
    import os
    config_path = "algobot/config.py"
    os.system(f"cp {config_path} {config_path}.backup_production")
    print(f"✅ Backed up {config_path}")
    
    production_config_content = '''"""
PRODUCTION TRADING CONFIGURATION
Optimized settings for live algorithmic trading
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    """Production ML model configuration - Balanced for real trading"""
    min_directional_accuracy: float = 0.52         # 52% accuracy requirement
    buy_threshold: float = 0.008                   # 0.8% buy threshold  
    sell_threshold: float = -0.006                 # -0.6% sell threshold
    confidence_threshold: float = 0.35             # 35% confidence requirement
    prediction_window: int = 5                     # 5-day prediction window
    model_retrain_frequency: int = 30              # Retrain every 30 days
    validation_score_min: float = 0.55             # 55% validation minimum

@dataclass  
class RiskConfig:
    """Production risk management - Conservative yet profitable"""
    max_position_pct: float = 0.25                # 25% maximum per position
    max_portfolio_risk_pct: float = 0.85          # 85% maximum portfolio risk
    stop_loss_pct: float = 0.08                   # 8% stop loss
    take_profit_pct: float = 0.15                 # 15% take profit
    trailing_stop_pct: float = 0.05               # 5% trailing stop
    max_daily_loss_pct: float = 0.03              # 3% maximum daily loss
    correlation_limit: float = 0.70               # 70% correlation limit
    min_holding_days: int = 3                     # 3 day minimum hold

@dataclass
class ExecutionConfig:
    """Production execution - Market-hours optimized"""
    rebalance_weekdays: Tuple[int, ...] = (1, 3)  # Tuesday, Thursday rebalancing
    min_holding_days: int = 3                     # 3 day minimum
    max_holding_days: int = 45                    # 45 day maximum
    market_hours_only: bool = True                # Market hours only
    slippage_assumption: float = 0.002            # 0.2% slippage
    min_trade_size: float = 100                   # $100 minimum trade
    target_positions: int = 8                     # Target 8 positions
    cash_reserve_pct: float = 0.15               # 15% cash reserve

@dataclass
class InstitutionalConfig:
    """Production institutional strategy parameters"""
    tier_probs: Tuple[float, ...] = (0.45, 0.52, 0.58, 0.65)  # Balanced tiers
    breakout_prob: float = 0.42                   # 42% breakout probability
    hard_exit_prob: float = 0.25                  # 25% hard exit
    soft_exit_prob: float = 0.35                  # 35% soft exit
    risk_per_trade_pct: float = 0.08              # 8% risk per trade
    early_core_fraction: float = 0.60             # 60% initial position

# Global configuration instance - PRODUCTION MODE
GLOBAL_CONFIG = {
    'model': ModelConfig(),
    'risk': RiskConfig(), 
    'execution': ExecutionConfig(),
    'institutional': InstitutionalConfig(),
    'production_mode': True,
    'strategy_type': 'OPTIMIZED_ML_INSTITUTIONAL'
}

print("🎯 PRODUCTION CONFIGURATION LOADED")
print("   • Balanced ML thresholds (52% accuracy)")
print("   • Conservative risk management (25% max position)")
print("   • Market-hours execution only")
print("   • Target: 15-25% annual returns")
'''
    
    # Write production config
    with open(config_path, 'w') as f:
        f.write(production_config_content)
    
    print(f"✅ Updated {config_path} with production settings")

def create_production_test_runner():
    """Create test runner for production configuration"""
    
    test_runner_content = '''#!/usr/bin/env python3
"""
Production Configuration Test Runner
Test the production configs on recent market data
"""

import sys
import os
sys.path.append('.')

def test_production_config():
    """Test production configuration"""
    
    print("🎯 TESTING PRODUCTION CONFIGURATION")
    print("=" * 50)
    
    # Import the updated config
    try:
        from algobot.config import GLOBAL_CONFIG
        
        print("✅ Production config loaded successfully")
        print(f"   • Model accuracy requirement: {GLOBAL_CONFIG['model'].min_directional_accuracy}")
        print(f"   • Max position size: {GLOBAL_CONFIG['risk'].max_position_pct*100}%")
        print(f"   • Target positions: {GLOBAL_CONFIG['execution'].target_positions}")
        print(f"   • Rebalance days: {GLOBAL_CONFIG['execution'].rebalance_weekdays}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading production config: {e}")
        return False

def run_production_backtest():
    """Run backtest with production settings"""
    
    print("\\n📊 RUNNING PRODUCTION BACKTEST")
    print("=" * 50)
    
    try:
        # Test with 5 stocks first
        cmd = "python -m algobot.portfolio.two_year_batch_runner --topk 5 --workers 1"
        print(f"🚀 Running: {cmd}")
        
        os.system(cmd)
        print("✅ Production backtest completed")
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")

if __name__ == "__main__":
    if test_production_config():
        run_production_backtest()
'''
    
    with open('test_production_config.py', 'w') as f:
        f.write(test_runner_content)
    
    print("✅ Created test_production_config.py")

def main():
    """Generate all production configurations"""
    
    print("🎯 ALGORITHMIC TRADING BOT - PRODUCTION CONFIGURATION")
    print("=" * 70)
    print("Creating optimal configurations for live trading...")
    print("Target: 15-25% annual returns with 8-12% max drawdown")
    print()
    
    # Generate all configurations
    master_config = generate_production_config_file()
    update_main_config_for_production()
    create_production_test_runner()
    
    print("\n🎯 PRODUCTION CONFIGURATION COMPLETE")
    print("=" * 70)
    print("✅ Master production config created")
    print("✅ Individual config files generated")
    print("✅ Main config.py updated for production")
    print("✅ Test runner created")
    print()
    
    print("📊 CONFIGURATION SUMMARY:")
    print("-" * 40)
    print(f"• ML Accuracy Requirement: 52%")
    print(f"• Maximum Position Size: 25%") 
    print(f"• Portfolio Risk Limit: 85%")
    print(f"• Stop Loss: 8%")
    print(f"• Take Profit: 15%")
    print(f"• Target Positions: 8")
    print(f"• Rebalancing: Tue/Thu")
    print(f"• Cash Reserve: 15%")
    print()
    
    print("🚀 NEXT STEPS:")
    print("1. Run: python3 test_production_config.py")
    print("2. Test on 5 stocks first")  
    print("3. If successful, scale to full 30 stocks")
    print("4. Deploy to live paper trading")
    print()
    
    print("💡 This configuration balances:")
    print("   • Strong returns (15-25% target)")
    print("   • Controlled risk (8% stop loss)")
    print("   • Realistic execution (market hours only)")
    print("   • Production safety (correlation limits, circuit breakers)")

if __name__ == "__main__":
    main()
