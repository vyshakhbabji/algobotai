#!/usr/bin/env python3
"""
Test NVDA vs Cross-Sectional Momentum Strategy
Forward test for 3 months using 1-2 year historical data
Capital: $10,000
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from algobot.portfolio.cross_sectional_momentum import run_cross_sectional
from algobot.forward_test.advanced_forward_sim import advanced_forward


def run_nvda_vs_cross_sectional():
    """
    Compare NVDA buy-and-hold vs cross-sectional momentum strategy
    """
    # Test period: 3 months from May 13, 2024 to Aug 13, 2024 (historical data)
    start_date = "2024-05-13"
    end_date = "2024-08-13"
    
    print(f"Running 3-month forward test from {start_date} to {end_date}")
    print(f"Capital: $10,000")
    print("=" * 60)
    
    # Test 1: NVDA single stock
    print("\n1. Testing NVDA as single stock...")
    nvda_result = advanced_forward(
        symbols=['NVDA'],
        start=start_date,
        end=end_date,
        lookback_days=504,  # ~2 years of data for training
        retrain_interval='M',
        prob_buy=0.60,
        prob_exit=0.50,
        prob_hard_exit=0.45,
        smoothing_window=3,
        vol_target_annual=0.18,
        min_holding_days=5,
        max_symbol_weight=1.0,  # 100% allocation to NVDA
        transaction_cost_bps=5,
        rebalance_weekdays=(0,),
        allow_midweek_hard_exits=True,
        use_regime_filter=True,
        regime_symbol='SPY',
        regime_fast=20,
        regime_slow=100,
        out_dir='nvda_single_test',
        chart=True,
        gross_target=1.0,
        allow_leverage=False,
    )
    
    # Test 2: Cross-sectional momentum strategy
    print("\n2. Testing Cross-Sectional Momentum Strategy...")
    cross_sectional_result = run_cross_sectional(
        scan_path='two_year_batch/scan_summary.json',
        start=start_date,
        end=end_date,
        topk=20,  # Top 20 stocks
        out_dir='cross_sectional_test',
        prob_buy=0.60,
        prob_exit=0.50,
        prob_hard_exit=0.45,
        smoothing_window=3,
        vol_target_annual=0.18,
        min_holding_days=5,
        max_symbol_weight=0.20,  # Max 20% per stock
        transaction_cost_bps=5,
        rebalance_weekdays=(0,),
        allow_midweek_hard_exits=True,
        use_regime_filter=True,
        regime_symbol='SPY',
        regime_fast=20,
        regime_slow=100,
        gross_target=1.0,
        allow_leverage=False
    )
    
    # Test 3: NVDA Buy and Hold for comparison
    print("\n3. Testing NVDA Buy and Hold...")
    
    # Create a simple buy and hold test
    try:
        import yfinance as yf
        
        # Get NVDA data for the test period
        nvda_data = yf.download('NVDA', start=start_date, end=end_date, progress=False)
        
        if not nvda_data.empty:
            start_price = float(nvda_data['Close'].iloc[0])
            end_price = float(nvda_data['Close'].iloc[-1])
            
            # Calculate returns with $10k capital
            shares = 10000 / start_price
            final_value = shares * end_price
            total_return = (final_value - 10000) / 10000
            
            nvda_buy_hold = {
                'total_return': total_return,
                'final_value': final_value,
                'start_price': start_price,
                'end_price': end_price,
                'shares_bought': shares,
                'strategy': 'Buy and Hold'
            }
        else:
            nvda_buy_hold = {'error': 'No data available for NVDA buy and hold'}
            
    except Exception as e:
        nvda_buy_hold = {'error': f'Could not calculate NVDA buy and hold: {str(e)}'}
    
    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    print(f"\n1. NVDA Single Stock Strategy:")
    if nvda_result and hasattr(nvda_result, 'metrics'):
        total_ret = nvda_result.metrics.get('total_return', 0)
        sharpe_rat = nvda_result.metrics.get('sharpe_ratio', 0)
        max_dd = nvda_result.metrics.get('max_drawdown', 0)
        win_rt = nvda_result.metrics.get('win_rate', 0)
        
        if isinstance(total_ret, (int, float)):
            print(f"   Total Return: {total_ret:.2%}")
        else:
            print(f"   Total Return: {total_ret}")
            
        if isinstance(sharpe_rat, (int, float)):
            print(f"   Sharpe Ratio: {sharpe_rat:.2f}")
        else:
            print(f"   Sharpe Ratio: {sharpe_rat}")
            
        if isinstance(max_dd, (int, float)):
            print(f"   Max Drawdown: {max_dd:.2%}")
        else:
            print(f"   Max Drawdown: {max_dd}")
            
        if isinstance(win_rt, (int, float)):
            print(f"   Win Rate: {win_rt:.2%}")
        else:
            print(f"   Win Rate: {win_rt}")
        
        # Calculate final value with $10k (scale from 100k)
        if isinstance(total_ret, (int, float)):
            final_value_nvda = 10000 * (1 + total_ret)
            print(f"   Final Value ($10k start): ${final_value_nvda:,.2f}")
            print(f"   [Note: Strategy uses $100k capital, scaled to $10k for comparison]")
    else:
        print("   No results available")
    
    print(f"\n2. Cross-Sectional Momentum Strategy:")
    if cross_sectional_result:
        total_ret = cross_sectional_result.get('total_return', 0)
        sharpe_rat = cross_sectional_result.get('sharpe_ratio', 0)
        max_dd = cross_sectional_result.get('max_drawdown', 0)
        win_rt = cross_sectional_result.get('win_rate', 0)
        
        if isinstance(total_ret, (int, float)):
            print(f"   Total Return: {total_ret:.2%}")
        else:
            print(f"   Total Return: {total_ret}")
            
        if isinstance(sharpe_rat, (int, float)):
            print(f"   Sharpe Ratio: {sharpe_rat:.2f}")
        else:
            print(f"   Sharpe Ratio: {sharpe_rat}")
            
        if isinstance(max_dd, (int, float)):
            print(f"   Max Drawdown: {max_dd:.2%}")
        else:
            print(f"   Max Drawdown: {max_dd}")
            
        if isinstance(win_rt, (int, float)):
            print(f"   Win Rate: {win_rt:.2%}")
        else:
            print(f"   Win Rate: {win_rt}")
        
        # Calculate final value with $10k (scale from 100k)
        if isinstance(total_ret, (int, float)):
            final_value_cs = 10000 * (1 + total_ret)
            print(f"   Final Value ($10k start): ${final_value_cs:,.2f}")
            print(f"   [Note: Strategy uses $100k capital, scaled to $10k for comparison]")
    else:
        print("   No results available")
    
    print(f"\n3. NVDA Buy and Hold:")
    if 'error' not in nvda_buy_hold:
        total_ret = nvda_buy_hold['total_return']
        final_val = nvda_buy_hold['final_value']
        start_pr = nvda_buy_hold['start_price']
        end_pr = nvda_buy_hold['end_price']
        shares = nvda_buy_hold['shares_bought']
        
        if isinstance(total_ret, (int, float)):
            print(f"   Total Return: {total_ret:.2%}")
        else:
            print(f"   Total Return: {total_ret}")
            
        if isinstance(final_val, (int, float)):
            print(f"   Final Value ($10k start): ${final_val:,.2f}")
        else:
            print(f"   Final Value ($10k start): {final_val}")
            
        if isinstance(start_pr, (int, float)):
            print(f"   Start Price: ${start_pr:.2f}")
        else:
            print(f"   Start Price: {start_pr}")
            
        if isinstance(end_pr, (int, float)):
            print(f"   End Price: ${end_pr:.2f}")
        else:
            print(f"   End Price: {end_pr}")
            
        if isinstance(shares, (int, float)):
            print(f"   Shares Bought: {shares:.4f}")
        else:
            print(f"   Shares Bought: {shares}")
    else:
        print(f"   Error: {nvda_buy_hold['error']}")
    
    # Save comparison results
    comparison_results = {
        'test_period': f"{start_date} to {end_date}",
        'capital': 10000,
        'nvda_single_strategy': nvda_result.metrics if nvda_result and hasattr(nvda_result, 'metrics') else None,
        'cross_sectional_strategy': cross_sectional_result,
        'nvda_buy_hold': nvda_buy_hold
    }
    
    with open('nvda_vs_cross_sectional_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to:")
    print(f"- NVDA single: nvda_single_test/")
    print(f"- Cross-sectional: cross_sectional_test/")
    print(f"- Comparison summary: nvda_vs_cross_sectional_comparison.json")
    
    return comparison_results


if __name__ == '__main__':
    run_nvda_vs_cross_sectional()
