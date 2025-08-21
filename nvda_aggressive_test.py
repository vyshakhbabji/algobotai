#!/usr/bin/env python3
"""
Test NVDA vs Cross-Sectional Momentum Strategy with more aggressive parameters
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from algobot.portfolio.cross_sectional_momentum import run_cross_sectional
from algobot.forward_test.advanced_forward_sim import advanced_forward


def run_aggressive_test():
    """
    Run more aggressive test to ensure trades are made
    """
    # Test period: 3 months from May 13, 2024 to Aug 13, 2024 (historical data)
    start_date = "2024-05-13"
    end_date = "2024-08-13"
    
    print(f"Running aggressive 3-month forward test from {start_date} to {end_date}")
    print(f"Capital: $10,000 (scaled from $100k)")
    print("=" * 60)
    
    # Test 1: NVDA single stock with aggressive parameters
    print("\n1. Testing NVDA as single stock (AGGRESSIVE)...")
    nvda_result = advanced_forward(
        symbols=['NVDA'],
        start=start_date,
        end=end_date,
        lookback_days=252,  # 1 year lookback
        retrain_interval='M',
        prob_buy=0.40,  # Lower threshold to buy more frequently
        prob_exit=0.35,  # Lower threshold to exit less frequently
        prob_hard_exit=0.25,  # Much lower hard exit
        smoothing_window=1,  # Less smoothing
        vol_target_annual=0.25,  # Higher vol target allows more aggressive positions
        min_holding_days=1,  # Shorter holding period
        max_symbol_weight=1.0,  # 100% allocation to NVDA
        transaction_cost_bps=5,
        rebalance_weekdays=(0,),
        allow_midweek_hard_exits=True,
        use_regime_filter=False,  # Disable regime filter
        out_dir='nvda_aggressive_test',
        chart=True,
        gross_target=1.0,
        allow_leverage=False,
    )
    
    # Test 2: Cross-sectional momentum strategy with aggressive parameters
    print("\n2. Testing Cross-Sectional Momentum Strategy (AGGRESSIVE)...")
    cross_sectional_result = run_cross_sectional(
        scan_path='two_year_batch/scan_summary.json',
        start=start_date,
        end=end_date,
        topk=10,  # Focus on top 10 stocks
        out_dir='cross_sectional_aggressive_test',
        prob_buy=0.40,  # Lower buy threshold
        prob_exit=0.35,  # Lower exit threshold
        prob_hard_exit=0.25,  # Lower hard exit
        smoothing_window=1,  # Less smoothing
        vol_target_annual=0.25,  # Higher volatility target
        min_holding_days=1,  # Shorter holding period
        max_symbol_weight=0.30,  # Higher max weight per stock
        transaction_cost_bps=5,
        rebalance_weekdays=(0,),
        allow_midweek_hard_exits=True,
        use_regime_filter=False,  # Disable regime filter
        gross_target=1.0,
        allow_leverage=False
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("AGGRESSIVE TEST RESULTS")
    print("=" * 60)
    
    print(f"\n1. NVDA Single Stock Strategy (Aggressive):")
    if nvda_result and hasattr(nvda_result, 'metrics'):
        total_ret = nvda_result.metrics.get('total_return', 0)
        num_trades = nvda_result.metrics.get('num_trades', 0)
        
        if isinstance(total_ret, (int, float)):
            print(f"   Total Return: {total_ret:.2%}")
            final_value_nvda = 10000 * (1 + total_ret)
            print(f"   Final Value ($10k start): ${final_value_nvda:,.2f}")
        else:
            print(f"   Total Return: {total_ret}")
            
        print(f"   Number of Trades: {num_trades}")
        print(f"   Sharpe Ratio: {nvda_result.metrics.get('sharpe_ratio', 'N/A')}")
    else:
        print("   No results available")
    
    print(f"\n2. Cross-Sectional Momentum Strategy (Aggressive):")
    if cross_sectional_result:
        total_ret = cross_sectional_result.get('total_return', 0)
        num_trades = cross_sectional_result.get('num_trades', 0)
        
        if isinstance(total_ret, (int, float)):
            print(f"   Total Return: {total_ret:.2%}")
            final_value_cs = 10000 * (1 + total_ret)
            print(f"   Final Value ($10k start): ${final_value_cs:,.2f}")
        else:
            print(f"   Total Return: {total_ret}")
            
        print(f"   Number of Trades: {num_trades}")
        print(f"   Sharpe Ratio: {cross_sectional_result.get('sharpe_ratio', 'N/A')}")
    else:
        print("   No results available")
    
    # NVDA Buy and Hold for comparison
    print(f"\n3. NVDA Buy and Hold (Baseline):")
    try:
        import yfinance as yf
        nvda_data = yf.download('NVDA', start=start_date, end=end_date, progress=False)
        
        if not nvda_data.empty:
            start_price = float(nvda_data['Close'].iloc[0])
            end_price = float(nvda_data['Close'].iloc[-1])
            
            shares = 10000 / start_price
            final_value = shares * end_price
            total_return = (final_value - 10000) / 10000
            
            print(f"   Total Return: {total_return:.2%}")
            print(f"   Final Value ($10k start): ${final_value:,.2f}")
        
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    return nvda_result, cross_sectional_result


if __name__ == '__main__':
    run_aggressive_test()
