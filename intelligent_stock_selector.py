#!/usr/bin/env python3
"""
Intelligent Stock Selection for $100K Investment
Use AI model to select only the best stocks from the 30 analyzed
"""
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def load_analysis_results():
    """Load the 30-stock analysis results"""
    with open('two_year_batch/batch_results.json', 'r') as f:
        data = json.load(f)
    return data

def select_best_stocks(results_data, max_stocks=5, min_confidence_threshold=75):
    """
    Intelligently select the best stocks to invest in
    Based on AI model performance, returns, and risk metrics
    """
    print("🤖 AI-POWERED STOCK SELECTION")
    print("=" * 50)
    
    # Extract stock metrics
    stocks = []
    for symbol, metrics in results_data['runs'].items():
        if 'error' not in metrics:
            # Calculate composite score
            return_pct = metrics.get('return_pct', 0)
            alpha_pct = metrics.get('alpha_pct', 0)
            capture_ratio = metrics.get('capture', 0)
            exposure_avg = metrics.get('exposure_avg', 0)
            
            # AI Selection Criteria
            score = 0
            
            # Return performance (40% weight)
            if return_pct > 0:
                score += return_pct * 40
            
            # Alpha generation (30% weight) 
            if alpha_pct > 0:
                score += alpha_pct * 30
            
            # Market capture efficiency (20% weight)
            if capture_ratio > 0.5:  # Good capture ratio
                score += capture_ratio * 20
            
            # Risk-adjusted exposure (10% weight)
            if 0.3 <= exposure_avg <= 0.7:  # Optimal exposure range
                score += 10
            
            stocks.append({
                'Symbol': symbol,
                'AI_Score': score,
                'Return (%)': return_pct,
                'Alpha (%)': alpha_pct,
                'Market Capture': capture_ratio,
                'Avg Exposure': exposure_avg,
                'Classification': metrics.get('class', 'N/A')
            })
    
    # Sort by AI score
    stocks_df = pd.DataFrame(stocks).sort_values('AI_Score', ascending=False)
    
    # Apply selection filters
    selected = []
    
    print(f"\n📊 STOCK SCREENING RESULTS:")
    print("-" * 70)
    
    for i, row in stocks_df.iterrows():
        symbol = row['Symbol']
        score = row['AI_Score']
        
        # Selection criteria
        reasons = []
        
        # Must be profitable
        if row['Return (%)'] <= 0:
            reasons.append("Non-profitable")
        
        # Must have reasonable exposure
        if row['Avg Exposure'] < 0.1:
            reasons.append("Too low exposure")
        
        # Prefer strong classified stocks
        if row['Classification'] == 'strong':
            score += 5  # Bonus for strong classification
        
        if len(reasons) == 0 and len(selected) < max_stocks:
            selected.append(row)
            print(f"✅ {symbol:>6} | Score: {score:>6.1f} | Return: {row['Return (%)']:>6.2f}% | Alpha: {row['Alpha (%)']:>6.2f}% | {row['Classification']}")
        else:
            reason_str = ", ".join(reasons) if reasons else "Not in top selection"
            print(f"❌ {symbol:>6} | Score: {score:>6.1f} | Reason: {reason_str}")
    
    return selected

def calculate_portfolio_allocation(selected_stocks, total_capital=100000):
    """
    Calculate optimal allocation for selected stocks
    """
    print(f"\n💰 PORTFOLIO ALLOCATION ({total_capital:,} total)")
    print("=" * 50)
    
    if not selected_stocks:
        print("❌ No stocks selected for investment")
        return {}
    
    # Simple equal-weight allocation for now
    allocation_per_stock = total_capital / len(selected_stocks)
    
    allocations = {}
    total_allocated = 0
    
    for stock in selected_stocks:
        symbol = stock['Symbol']
        
        # Get current price for share calculation
        try:
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            
            shares = int(allocation_per_stock / current_price)
            actual_allocation = shares * current_price
            
            allocations[symbol] = {
                'shares': shares,
                'price': current_price,
                'allocation': actual_allocation,
                'weight': actual_allocation / total_capital,
                'expected_return': stock['Return (%)'],
                'alpha': stock['Alpha (%)']
            }
            
            total_allocated += actual_allocation
            
            print(f"{symbol:>6} | ${actual_allocation:>8,.0f} | {shares:>4} shares @ ${current_price:>7.2f} | Weight: {actual_allocation/total_capital:>5.1%}")
            
        except Exception as e:
            print(f"❌ Error getting price for {symbol}: {e}")
    
    cash_remaining = total_capital - total_allocated
    
    print("-" * 50)
    print(f"{'TOTAL':>6} | ${total_allocated:>8,.0f} | Invested: {total_allocated/total_capital:>5.1%}")
    print(f"{'CASH':>6} | ${cash_remaining:>8,.0f} | Remaining: {cash_remaining/total_capital:>5.1%}")
    
    return allocations

def simulate_nvda_only(total_capital=100000):
    """
    Simulate investing all $100K in NVDA only
    """
    print(f"\n🎯 NVDA-ONLY SIMULATION ({total_capital:,} investment)")
    print("=" * 50)
    
    try:
        # Get NVDA current price
        nvda = yf.Ticker('NVDA')
        current_price = nvda.history(period='1d')['Close'].iloc[-1]
        
        # Calculate shares
        shares = int(total_capital / current_price)
        actual_investment = shares * current_price
        cash_remaining = total_capital - actual_investment
        
        # Get NVDA results from our analysis
        with open('two_year_batch/batch_results.json', 'r') as f:
            data = json.load(f)
        
        nvda_results = data['runs'].get('NVDA', {})
        
        if 'error' not in nvda_results:
            expected_return = nvda_results.get('return_pct', 0)
            alpha = nvda_results.get('alpha_pct', 0)
            
            # Project final value
            final_value = actual_investment * (1 + expected_return/100)
            profit = final_value - actual_investment
            
            print(f"Current NVDA Price: ${current_price:.2f}")
            print(f"Shares Purchased: {shares:,}")
            print(f"Investment Amount: ${actual_investment:,.0f}")
            print(f"Cash Remaining: ${cash_remaining:.0f}")
            print(f"Expected Return: {expected_return:.2f}%")
            print(f"Alpha vs Market: {alpha:.2f}%")
            print(f"Projected Value: ${final_value:,.0f}")
            print(f"Projected Profit: ${profit:,.0f}")
            
            return {
                'symbol': 'NVDA',
                'shares': shares,
                'investment': actual_investment,
                'expected_return': expected_return,
                'projected_profit': profit
            }
        else:
            print("❌ NVDA analysis not available")
            return None
            
    except Exception as e:
        print(f"❌ Error simulating NVDA: {e}")
        return None

def main():
    """Main execution"""
    print("🚀 INTELLIGENT PORTFOLIO CONSTRUCTION")
    print("=" * 80)
    
    # Load analysis results
    results = load_analysis_results()
    
    # AI-powered stock selection
    selected_stocks = select_best_stocks(results, max_stocks=5)
    
    if selected_stocks:
        print(f"\n🎯 SELECTED {len(selected_stocks)} STOCKS FOR INVESTMENT")
        
        # Calculate portfolio allocation
        allocations = calculate_portfolio_allocation(selected_stocks)
        
        # Calculate expected portfolio performance
        total_expected_return = sum(
            alloc['expected_return'] * alloc['weight'] 
            for alloc in allocations.values()
        )
        
        total_alpha = sum(
            alloc['alpha'] * alloc['weight'] 
            for alloc in allocations.values()
        )
        
        print(f"\n📈 PORTFOLIO PROJECTIONS:")
        print(f"Expected Return: {total_expected_return:.2f}%")
        print(f"Expected Alpha: {total_alpha:.2f}%")
        print(f"Number of Positions: {len(allocations)}")
    
    else:
        print("\n❌ No stocks met selection criteria!")
    
    # Compare with NVDA-only strategy
    print("\n" + "="*80)
    nvda_simulation = simulate_nvda_only()
    
    if selected_stocks and nvda_simulation:
        print(f"\n⚖️ STRATEGY COMPARISON:")
        print("-" * 40)
        diversified_return = sum(
            alloc['expected_return'] * alloc['weight'] 
            for alloc in allocations.values()
        )
        
        print(f"Diversified Portfolio: {diversified_return:.2f}% expected return")
        print(f"NVDA Only: {nvda_simulation['expected_return']:.2f}% expected return")
        
        if nvda_simulation['expected_return'] > diversified_return:
            print("🎯 NVDA-only strategy shows higher expected returns!")
        else:
            print("🔗 Diversified portfolio shows better risk-adjusted returns!")

if __name__ == "__main__":
    main()
