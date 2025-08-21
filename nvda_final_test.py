#!/usr/bin/env python3
"""
Final NVDA vs Cross-Sectional Momentum Test with working parameters
"""

import json
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

from algobot.portfolio.cross_sectional_momentum import run_cross_sectional
from algobot.forward_test.advanced_forward_sim import advanced_forward


def run_final_test():
    """
    Final working test of NVDA vs cross-sectional momentum
    """
    # Test period: 3 months from May 13, 2024 to Aug 13, 2024 (historical data)
    start_date = "2024-05-13"
    end_date = "2024-08-13"
    
    print(f"NVDA vs Cross-Sectional Momentum Strategy")
    print(f"Test period: {start_date} to {end_date} (3 months)")
    print(f"Capital: $10,000")
    print("=" * 70)
    
    # Test 1: NVDA single stock with reduced lookback
    print("\n1. Testing NVDA Single Stock Strategy...")
    nvda_result = advanced_forward(
        symbols=['NVDA'],
        start=start_date,
        end=end_date,
        lookback_days=126,  # Reduced from 504 to 126 (6 months)
        retrain_interval='M',
        prob_buy=0.50,  # Reduced from 0.60 to allow more trades
        prob_exit=0.45,  # Reduced from 0.50
        prob_hard_exit=0.35,  # Reduced from 0.45
        smoothing_window=1,  # Reduced smoothing
        vol_target_annual=0.20,
        min_holding_days=1,  # Reduced from 5
        max_symbol_weight=1.0,  # 100% allocation to NVDA
        transaction_cost_bps=5,
        rebalance_weekdays=(0,1,2,3,4),  # Allow trading every day
        allow_midweek_hard_exits=True,
        use_regime_filter=False,  # Disable regime filter
        out_dir='nvda_final_test',
        chart=True,
        gross_target=1.0,
        allow_leverage=False,
    )
    
    # Test 2: Cross-sectional momentum strategy with same parameters
    print("\n2. Testing Cross-Sectional Momentum Strategy...")
    cross_sectional_result = run_cross_sectional(
        scan_path='two_year_batch/scan_summary.json',
        start=start_date,
        end=end_date,
        topk=15,  # Top 15 stocks
        out_dir='cross_sectional_final_test',
        prob_buy=0.50,  # Reduced threshold
        prob_exit=0.45,
        prob_hard_exit=0.35,
        smoothing_window=1,
        vol_target_annual=0.20,
        min_holding_days=1,
        max_symbol_weight=0.25,  # Max 25% per stock
        transaction_cost_bps=5,
        rebalance_weekdays=(0,1,2,3,4),  # Daily rebalancing
        allow_midweek_hard_exits=True,
        use_regime_filter=False,  # Disable regime filter
        gross_target=1.0,
        allow_leverage=False
    )
    
    # Calculate NVDA Buy and Hold
    print("\n3. Calculating NVDA Buy and Hold...")
    try:
        nvda_data = yf.download('NVDA', start=start_date, end=end_date, progress=False)
        
        if not nvda_data.empty:
            start_price = nvda_data['Close'].iloc[0]
            end_price = nvda_data['Close'].iloc[-1]
            
            # For pandas Series, extract the actual float values
            if hasattr(start_price, 'iloc'):
                start_price = float(start_price.iloc[0])
            else:
                start_price = float(start_price)
                
            if hasattr(end_price, 'iloc'):
                end_price = float(end_price.iloc[0])
            else:
                end_price = float(end_price)
            
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
            nvda_buy_hold = {'error': 'No data available'}
            
    except Exception as e:
        nvda_buy_hold = {'error': f'Error calculating buy and hold: {str(e)}'}
    
    # Print Results
    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)
    
    # Helper function to safely format percentages
    def safe_format_pct(value, default="N/A"):
        try:
            if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                return f"{value:.2%}"
            else:
                return default
        except:
            return default
    
    def safe_format_num(value, decimals=2, default="N/A"):
        try:
            if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                return f"{value:.{decimals}f}"
            else:
                return default
        except:
            return default
    
    print(f"\n1. NVDA Single Stock Strategy:")
    if nvda_result and hasattr(nvda_result, 'metrics'):
        metrics = nvda_result.metrics
        total_ret = metrics.get('total_return', 0)
        num_trades = metrics.get('num_trades', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)
        
        print(f"   Total Return: {safe_format_pct(total_ret)}")
        print(f"   Sharpe Ratio: {safe_format_num(sharpe)}")
        print(f"   Max Drawdown: {safe_format_pct(max_dd)}")
        print(f"   Number of Trades: {num_trades}")
        
        if isinstance(total_ret, (int, float)):
            final_value_nvda = 10000 * (1 + total_ret)
            print(f"   Final Value ($10k start): ${final_value_nvda:,.2f}")
        
    else:
        print("   No results available")
    
    print(f"\n2. Cross-Sectional Momentum Strategy:")
    if cross_sectional_result:
        total_ret = cross_sectional_result.get('total_return', 0)
        num_trades = cross_sectional_result.get('num_trades', 0)
        sharpe = cross_sectional_result.get('sharpe_ratio', 0)
        max_dd = cross_sectional_result.get('max_drawdown', 0)
        
        print(f"   Total Return: {safe_format_pct(total_ret)}")
        print(f"   Sharpe Ratio: {safe_format_num(sharpe)}")
        print(f"   Max Drawdown: {safe_format_pct(max_dd)}")
        print(f"   Number of Trades: {num_trades}")
        
        if isinstance(total_ret, (int, float)):
            final_value_cs = 10000 * (1 + total_ret)
            print(f"   Final Value ($10k start): ${final_value_cs:,.2f}")
        
    else:
        print("   No results available")
    
    print(f"\n3. NVDA Buy and Hold (Baseline):")
    if 'error' not in nvda_buy_hold:
        total_ret = nvda_buy_hold['total_return']
        final_val = nvda_buy_hold['final_value']
        
        print(f"   Total Return: {safe_format_pct(total_ret)}")
        print(f"   Final Value ($10k start): ${final_val:,.2f}")
        print(f"   Start Price: ${nvda_buy_hold['start_price']:.2f}")
        print(f"   End Price: ${nvda_buy_hold['end_price']:.2f}")
        
    else:
        print(f"   Error: {nvda_buy_hold['error']}")
    
    # Summary comparison
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    strategies = []
    
    if nvda_result and hasattr(nvda_result, 'metrics'):
        nvda_ret = nvda_result.metrics.get('total_return', 0)
        if isinstance(nvda_ret, (int, float)):
            strategies.append(("NVDA Single Stock", nvda_ret, 10000 * (1 + nvda_ret)))
    
    if cross_sectional_result:
        cs_ret = cross_sectional_result.get('total_return', 0)
        if isinstance(cs_ret, (int, float)):
            strategies.append(("Cross-Sectional Momentum", cs_ret, 10000 * (1 + cs_ret)))
    
    if 'error' not in nvda_buy_hold:
        bh_ret = nvda_buy_hold['total_return']
        strategies.append(("NVDA Buy & Hold", bh_ret, nvda_buy_hold['final_value']))
    
    if strategies:
        strategies.sort(key=lambda x: x[1], reverse=True)
        print(f"\nRanking by Total Return:")
        for i, (name, ret, final_val) in enumerate(strategies, 1):
            print(f"   {i}. {name}: {ret:.2%} (${final_val:,.2f})")
    
    # Save detailed results
    comparison_results = {
        'test_period': f"{start_date} to {end_date}",
        'capital': 10000,
        'nvda_single_strategy': nvda_result.metrics if nvda_result and hasattr(nvda_result, 'metrics') else None,
        'cross_sectional_strategy': cross_sectional_result,
        'nvda_buy_hold': nvda_buy_hold
    }
    
    with open('final_nvda_vs_cross_sectional_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to:")
    print(f"- NVDA single: nvda_final_test/")
    print(f"- Cross-sectional: cross_sectional_final_test/")
    print(f"- Comparison summary: final_nvda_vs_cross_sectional_comparison.json")
    
    return comparison_results


if __name__ == '__main__':
    run_final_test()
