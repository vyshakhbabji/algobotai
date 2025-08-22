#!/usr/bin/env python3
"""Test the signal-based exit strategy"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realistic_live_trading_system import RealisticLiveTradingSystem
import pandas as pd
import json

def test_signal_exit_logic():
    print("🧪 Testing Signal-Based Exit Strategy")
    print("=" * 50)
    
    # Initialize the system
    system = RealisticLiveTradingSystem()
    
    # Set min holding days to something small for testing
    system.min_holding_days = 1
    system.config = {'signal_threshold': 0.5, 'max_position_size': 0.4}
    print("\n1. Setting up test scenario:")
    print("   - Mock position: AAPL (entry signal: 0.85 - Ultra strong)")
    print("   - Current signal: 0.45 - Weak")
    print("   - Should trigger exit: Strong→Weak signal degradation")
    
    # Set up mock data
    system.positions = {'AAPL': 100}  # 100 shares of AAPL
    system.position_entry_signals = {'AAPL': 0.85}  # Strong entry signal
    system.position_entry_dates = {'AAPL': pd.Timestamp('2024-01-01')}  # Old enough
    
    current_signals = {
        'AAPL': {
            'strength': 0.45,  # Weak current signal (degraded from 0.85)
            'price': 150.0,
            'base_strength': 0.45,
            'ml_multiplier': 1.0,
            'regime_boost': 1.0,
            'total_enhancement': 1.0,
            'ml_enhanced': False
        }
    }
    
    prices_today = {'AAPL': 150.0}
    current_date = pd.Timestamp('2024-02-01')  # 1 month later
    
    print(f"\n2. Test input:")
    print(f"   - Entry signal strength: {system.position_entry_signals['AAPL']:.3f}")
    print(f"   - Current signal strength: {current_signals['AAPL']['strength']:.3f}")
    print(f"   - Signal degradation: {system.position_entry_signals['AAPL'] - current_signals['AAPL']['strength']:.3f}")
    
    # Test the exit logic
    exit_orders = system._check_signal_based_exits(current_signals, prices_today, current_date)
    
    print(f"\n3. Exit decision:")
    print(f"   - Exit orders generated: {exit_orders}")
    print(f"   - Pending orders queue: {len(system.pending_orders)}")
    
    if exit_orders > 0:
        print("   ✅ PASS: Signal-based exit triggered correctly")
        for order in system.pending_orders:
            if 'symbol' in order:
                print(f"   📋 Order: {order['action']} {order['shares']} {order['symbol']} - {order['reason']}")
    else:
        print("   ❌ FAIL: Expected exit order but none generated")
    
    print("\n4. Testing edge cases:")
    
    # Test 2: Signal improvement (should NOT exit)
    system.pending_orders.clear()
    system.position_entry_signals = {'AAPL': 0.6}  # Moderate entry
    current_signals['AAPL']['strength'] = 0.8  # Improved signal
    
    exit_orders = system._check_signal_based_exits(current_signals, prices_today, current_date)
    print(f"   - Signal improvement test: {exit_orders} exits (should be 0)")
    
    # Test 3: Minimum holding period (should delay exit)
    system.pending_orders.clear()
    system.min_holding_days = 5  # Increase holding period requirement
    system.position_entry_signals = {'AAPL': 0.9}  # Ultra entry
    system.position_entry_dates = {'AAPL': pd.Timestamp('2024-01-30')}  # Too recent (2 days ago)
    current_signals['AAPL']['strength'] = 0.4  # Should trigger exit but blocked by holding period
    
    exit_orders = system._check_signal_based_exits(current_signals, prices_today, current_date)
    print(f"   - Holding period test: {exit_orders} exits (should be 0 - too soon)")
    
    print("\n🎯 Signal-based exit testing complete!")

if __name__ == "__main__":
    test_signal_exit_logic()
