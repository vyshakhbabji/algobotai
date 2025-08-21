#!/usr/bin/env python3
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
    
    print("\n📊 RUNNING PRODUCTION BACKTEST")
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
