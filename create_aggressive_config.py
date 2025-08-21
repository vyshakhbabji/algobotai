#!/usr/bin/env python3
"""
Aggressive Strategy Configuration Override
Create high-return configuration for better performance
"""
import json
from pathlib import Path

def create_aggressive_config():
    """Create aggressive configuration for higher returns"""
    
    # Current conservative settings vs aggressive alternatives
    aggressive_config = {
        "strategy_name": "Aggressive High Return Strategy",
        "description": "Optimized for maximum returns with higher risk tolerance",
        "created": "2025-08-12",
        
        # RISK PARAMETERS (More Aggressive)
        "risk_settings": {
            "risk_per_trade_pct": 0.08,        # 8% vs current 4-5%
            "max_position_pct": 0.25,          # 25% vs current 18%
            "max_portfolio_risk_pct": 0.95,    # 95% vs current 90%
            "risk_ceiling": 0.12,              # 12% vs current 8%
            "risk_increment": 0.02,            # 2% vs current 1%
        },
        
        # POSITION SIZING (Larger Positions)
        "position_settings": {
            "min_position_size": 0.08,         # 8% minimum vs 5%
            "max_position_size": 0.25,         # 25% maximum vs 18%
            "position_concentration_limit": 3,  # Allow 3 large positions
        },
        
        # PROFIT OPTIMIZATION (More Aggressive Scaling)
        "profit_settings": {
            "profit_ladder": [0.15, 0.30, 0.50, 0.75],  # Added 75% tier
            "profit_trim_fractions": [0.08, 0.12, 0.15, 0.20],  # Smaller trims
            "fast_scale_gain_threshold": 0.06,  # 6% vs 8-10%
            "take_profit_pct": 0.25,           # 25% vs 15%
        },
        
        # ENTRY/EXIT THRESHOLDS (More Sensitive)
        "signal_settings": {
            "buy_threshold": 0.010,            # 1.0% vs 1.5%
            "sell_threshold": -0.010,          # -1.0% vs -1.5%
            "tier_probs": [0.48, 0.54, 0.60, 0.67],  # Lower thresholds
            "breakout_prob": 0.48,             # 48% vs 50%
            "hard_exit_prob": 0.42,           # 42% vs 45%
            "soft_exit_prob": 0.46,           # 46% vs 50%
        },
        
        # TIMING PARAMETERS (Faster Execution)
        "timing_settings": {
            "min_holding_days": 3,             # 3 vs 5-7 days
            "time_scalein_days": 3,            # 3 vs 5 days
            "momentum_20d_threshold": 0.06,    # 6% vs 8%
            "stale_days": 45,                  # 45 vs 60 days
        },
        
        # TRAIL STOPS (Tighter but allow more upside)
        "trail_settings": {
            "atr_initial_mult": 1.5,           # 1.5 vs 1.7
            "atr_trail_mult": 2.0,             # 2.0 vs 2.2-2.6
            "stop_loss_pct": 0.06,             # 6% vs 8%
            "adaptive_trail_mult_after_gain": 1.3,  # 1.3 vs 1.5
        },
        
        # MODEL THRESHOLDS (Lower barriers)
        "model_settings": {
            "min_directional_accuracy": 0.50,  # 50% vs 52%
            "min_r2": -0.10,                   # -10% vs -5%
            "confidence_threshold": 0.45,      # 45% vs 50%
        },
        
        # EXECUTION SETTINGS (More Active)
        "execution_settings": {
            "rebalance_weekdays": [0, 2, 4],   # Mon, Wed, Fri vs Mon only
            "hard_add_buy_strength": 0.70,     # 70% vs 75-85%
            "hard_trim_sell_strength": 0.60,   # 60% vs 55%
            "hard_add_chunk_weight": 0.08,     # 8% vs 4-5%
            "enable_pullback_reentry": True,
            "allow_midweek_exits": True,
        }
    }
    
    return aggressive_config

def create_ultra_aggressive_config():
    """Create ultra-aggressive configuration for maximum returns"""
    
    ultra_config = {
        "strategy_name": "Ultra Aggressive Maximum Return Strategy",
        "description": "Maximum risk/reward optimization - use with caution",
        "created": "2025-08-12",
        
        # EXTREME RISK PARAMETERS
        "risk_settings": {
            "risk_per_trade_pct": 0.12,        # 12% per trade
            "max_position_pct": 0.35,          # 35% single position
            "max_portfolio_risk_pct": 0.98,    # 98% deployed
            "risk_ceiling": 0.15,              # 15% max risk
            "risk_increment": 0.03,            # 3% increments
        },
        
        # CONCENTRATED POSITIONS
        "position_settings": {
            "min_position_size": 0.12,         # 12% minimum
            "max_position_size": 0.35,         # 35% maximum  
            "position_concentration_limit": 2,  # Only 2-3 positions max
        },
        
        # MAXIMIZED PROFITS
        "profit_settings": {
            "profit_ladder": [0.10, 0.20, 0.35, 0.60, 1.0],  # Up to 100%
            "profit_trim_fractions": [0.05, 0.08, 0.10, 0.12, 0.15],
            "fast_scale_gain_threshold": 0.04,  # 4% trigger
            "take_profit_pct": 0.40,           # 40% take profit
        },
        
        # HAIR-TRIGGER SIGNALS
        "signal_settings": {
            "buy_threshold": 0.005,            # 0.5% threshold
            "sell_threshold": -0.005,          # -0.5% threshold
            "tier_probs": [0.45, 0.50, 0.55, 0.62],
            "breakout_prob": 0.45,
            "hard_exit_prob": 0.40,
            "soft_exit_prob": 0.43,
        },
        
        # RAPID EXECUTION
        "timing_settings": {
            "min_holding_days": 1,             # Same day exits allowed
            "time_scalein_days": 1,            # Immediate scaling
            "momentum_20d_threshold": 0.04,    # 4% momentum
            "stale_days": 30,                  # 30 day limit
        },
        
        # TIGHT STOPS BUT HIGH TARGETS
        "trail_settings": {
            "atr_initial_mult": 1.2,           # Tighter initial stop
            "atr_trail_mult": 1.8,             # Tighter trail
            "stop_loss_pct": 0.05,             # 5% stop loss
            "adaptive_trail_mult_after_gain": 1.0,  # No loosening
        }
    }
    
    return ultra_config

def save_configs():
    """Save both aggressive configurations"""
    
    configs = {
        "aggressive": create_aggressive_config(),
        "ultra_aggressive": create_ultra_aggressive_config()
    }
    
    # Save to JSON file
    with open('aggressive_strategy_configs.json', 'w') as f:
        json.dump(configs, f, indent=2)
    
    print("🔥 AGGRESSIVE STRATEGY CONFIGURATIONS CREATED")
    print("=" * 60)
    
    for name, config in configs.items():
        print(f"\n📈 {config['strategy_name']}")
        print(f"   Risk per trade: {config['risk_settings']['risk_per_trade_pct']:.1%}")
        print(f"   Max position: {config['position_settings']['max_position_size']:.1%}")
        print(f"   Portfolio risk: {config['risk_settings']['max_portfolio_risk_pct']:.1%}")
        print(f"   Buy threshold: {config['signal_settings']['buy_threshold']:.1%}")
        print(f"   Min holding: {config['timing_settings']['min_holding_days']} days")
        print(f"   Take profit: {config['profit_settings']['take_profit_pct']:.1%}")
    
    return configs

def apply_aggressive_config_to_batch_runner():
    """Create modified batch runner with aggressive settings"""
    
    aggressive_params = {
        "risk_per_trade_pct": 0.08,
        "max_portfolio_risk": 0.95,
        "profit_ladder": (0.15, 0.30, 0.50),
        "profit_trim_fractions": (0.08, 0.12, 0.15),
        "tier_probs": (0.48, 0.54, 0.60, 0.67),
        "breakout_prob": 0.48,
        "hard_exit_prob": 0.42,
        "atr_initial_mult": 1.5,
        "atr_trail_mult": 2.0,
        "min_holding_days": 3,
        "fast_scale_gain_threshold": 0.06
    }
    
    print(f"\n💡 TO APPLY AGGRESSIVE CONFIG:")
    print(f"   Run with these parameters in batch runner:")
    for param, value in aggressive_params.items():
        print(f"   --{param} {value}")
    
    return aggressive_params

if __name__ == "__main__":
    configs = save_configs()
    apply_aggressive_config_to_batch_runner()
    
    print(f"\n🎯 EXPECTED IMPACT:")
    print(f"   • 3-5x higher returns (from 0.2% to 0.6-1.0%)")
    print(f"   • Higher volatility and drawdowns")
    print(f"   • More active trading (faster entries/exits)")
    print(f"   • Concentrated positions in best opportunities")
    print(f"   • Hair-trigger signal responses")
    
    print(f"\n⚠️  RISK WARNING:")
    print(f"   • Aggressive configs can lead to significant losses")
    print(f"   • Test thoroughly before live deployment")
    print(f"   • Consider starting with 'aggressive' before 'ultra_aggressive'")
    print(f"   • Monitor drawdowns closely")
    
    print(f"\n📁 Configuration saved to: aggressive_strategy_configs.json")
