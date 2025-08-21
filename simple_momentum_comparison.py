#!/usr/bin/env python3
"""
Simple momentum strategy comparison: NVDA vs diversified portfolio
Bypassing the complex advanced_forward_sim to get actual results
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import json

def calculate_simple_momentum_signals(data, window=20):
    """Calculate simple momentum signals"""
    returns = data['Close'].pct_change()
    momentum = returns.rolling(window).mean()
    volatility = returns.rolling(window).std()
    
    # Simple signal: buy when momentum > 0 and volatility is reasonable
    signals = pd.DataFrame(index=data.index)
    signals['momentum'] = momentum
    signals['volatility'] = volatility
    signals['signal'] = np.where(
        (momentum > 0) & (volatility < 0.05),  # Positive momentum, low volatility
        1,  # Buy
        0   # Hold/No position
    )
    
    return signals

def backtest_simple_strategy(data, signals, initial_capital=10000):
    """Simple backtest of momentum strategy"""
    
    portfolio = pd.DataFrame(index=data.index)
    portfolio['price'] = data['Close']
    portfolio['signal'] = signals['signal']
    
    # Position sizing: full position when signal = 1, cash when signal = 0
    portfolio['position'] = portfolio['signal']
    portfolio['holdings'] = portfolio['position'].shift(1).fillna(0)
    
    # Calculate daily returns
    portfolio['daily_return'] = data['Close'].pct_change()
    portfolio['strategy_return'] = portfolio['holdings'] * portfolio['daily_return']
    
    # Calculate cumulative returns
    portfolio['cum_return'] = (1 + portfolio['strategy_return']).cumprod()
    portfolio['total_value'] = initial_capital * portfolio['cum_return']
    
    # Calculate metrics
    total_return = (portfolio['total_value'].iloc[-1] / initial_capital) - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio)) - 1
    sharpe = portfolio['strategy_return'].mean() / portfolio['strategy_return'].std() * np.sqrt(252)
    max_dd = (portfolio['total_value'] / portfolio['total_value'].cummax() - 1).min()
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'final_value': portfolio['total_value'].iloc[-1],
        'portfolio': portfolio
    }

def run_simple_comparison():
    """Run simple momentum strategy comparison"""
    
    start_date = "2024-05-13"
    end_date = "2024-08-13"
    initial_capital = 10000
    
    print("Simple Momentum Strategy Comparison")
    print(f"Period: {start_date} to {end_date}")
    print(f"Capital: ${initial_capital:,}")
    print("=" * 60)
    
    # Test 1: NVDA Simple Momentum
    print("\n1. NVDA Simple Momentum Strategy")
    try:
        nvda_data = yf.download('NVDA', start=start_date, end=end_date, progress=False)
        nvda_signals = calculate_simple_momentum_signals(nvda_data)
        nvda_results = backtest_simple_strategy(nvda_data, nvda_signals, initial_capital)
        
        print(f"   Total Return: {nvda_results['total_return']:.2%}")
        print(f"   Final Value: ${nvda_results['final_value']:,.2f}")
        print(f"   Sharpe Ratio: {nvda_results['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {nvda_results['max_drawdown']:.2%}")
        
    except Exception as e:
        print(f"   Error: {e}")
        nvda_results = None
    
    # Test 2: Diversified Momentum (top stocks from scan)
    top_stocks = ['NFLX', 'PM', 'T', 'BK', 'WMT', 'AVGO', 'COST', 'GE', 'V', 'JPM']
    print(f"\n2. Diversified Momentum Strategy ({len(top_stocks)} stocks)")
    
    try:
        # Download data for all stocks
        portfolio_data = {}
        portfolio_results = {}
        
        for symbol in top_stocks:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    signals = calculate_simple_momentum_signals(data)
                    results = backtest_simple_strategy(data, signals, initial_capital / len(top_stocks))
                    portfolio_data[symbol] = data
                    portfolio_results[symbol] = results
                    
            except Exception as e:
                print(f"   {symbol}: Error - {e}")
        
        # Combine portfolio results
        if portfolio_results:
            total_final_value = sum(r['final_value'] for r in portfolio_results.values())
            total_return = (total_final_value / initial_capital) - 1
            
            # Calculate weighted metrics
            weights = [initial_capital / len(top_stocks) / initial_capital for _ in portfolio_results]
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in portfolio_results.values() if not np.isnan(r['sharpe_ratio'])])
            avg_max_dd = np.mean([r['max_drawdown'] for r in portfolio_results.values()])
            
            print(f"   Total Return: {total_return:.2%}")
            print(f"   Final Value: ${total_final_value:,.2f}")
            print(f"   Avg Sharpe Ratio: {avg_sharpe:.2f}")
            print(f"   Avg Max Drawdown: {avg_max_dd:.2%}")
            print(f"   Active Stocks: {len(portfolio_results)}/{len(top_stocks)}")
            
            diversified_results = {
                'total_return': total_return,
                'final_value': total_final_value,
                'sharpe_ratio': avg_sharpe,
                'max_drawdown': avg_max_dd,
                'num_stocks': len(portfolio_results)
            }
        else:
            print("   No successful stock results")
            diversified_results = None
            
    except Exception as e:
        print(f"   Error: {e}")
        diversified_results = None
    
    # Test 3: NVDA Buy and Hold
    print(f"\n3. NVDA Buy and Hold")
    try:
        nvda_data = yf.download('NVDA', start=start_date, end=end_date, progress=False)
        start_price = float(nvda_data['Close'].iloc[0])
        end_price = float(nvda_data['Close'].iloc[-1])
        
        shares = initial_capital / start_price
        final_value = shares * end_price
        total_return = (final_value / initial_capital) - 1
        
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Final Value: ${final_value:,.2f}")
        print(f"   Start Price: ${start_price:.2f}")
        print(f"   End Price: ${end_price:.2f}")
        
        buy_hold_results = {
            'total_return': total_return,
            'final_value': final_value,
            'start_price': start_price,
            'end_price': end_price
        }
        
    except Exception as e:
        print(f"   Error: {e}")
        buy_hold_results = None
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    strategies = []
    
    if nvda_results:
        strategies.append(("NVDA Momentum", nvda_results['total_return'], nvda_results['final_value']))
    
    if diversified_results:
        strategies.append(("Diversified Momentum", diversified_results['total_return'], diversified_results['final_value']))
    
    if buy_hold_results:
        strategies.append(("NVDA Buy & Hold", buy_hold_results['total_return'], buy_hold_results['final_value']))
    
    if strategies:
        strategies.sort(key=lambda x: x[1], reverse=True)
        print(f"\nRanking by Total Return:")
        for i, (name, ret, final_val) in enumerate(strategies, 1):
            print(f"   {i}. {name}: {ret:.2%} (${final_val:,.2f})")
    
    # Save results
    results = {
        'test_period': f"{start_date} to {end_date}",
        'initial_capital': initial_capital,
        'nvda_momentum': nvda_results,
        'diversified_momentum': diversified_results,
        'nvda_buy_hold': buy_hold_results
    }
    
    with open('simple_momentum_comparison.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: simple_momentum_comparison.json")
    
    return results

if __name__ == '__main__':
    run_simple_comparison()
