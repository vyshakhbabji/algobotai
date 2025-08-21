#!/usr/bin/env python3
"""
Aggressive vs Conservative Strategy Comparison
Compare the returns from aggressive vs conservative configurations
"""
import json
import pandas as pd
from datetime import datetime

def load_results(results_file='two_year_batch/batch_results.json'):
    """Load batch analysis results"""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return None

def analyze_aggressive_results():
    """Analyze the aggressive strategy results"""
    
    print("🔥 AGGRESSIVE STRATEGY RESULTS ANALYSIS")
    print("=" * 70)
    
    # Load aggressive results (latest run)
    aggressive_results = load_results('two_year_batch/batch_results.json')
    
    if not aggressive_results:
        return
    
    runs = aggressive_results.get('runs', {})
    
    if not runs:
        print("❌ No results found in aggressive analysis")
        return
    
    print(f"📊 Analyzed {len(runs)} stocks")
    print(f"📅 Analysis Period: {aggressive_results.get('period', 'Unknown')}")
    
    # Calculate statistics
    profitable_stocks = []
    losing_stocks = []
    total_return = 0
    total_alpha = 0
    
    stock_performance = []
    
    for symbol, metrics in runs.items():
        if 'error' in metrics:
            continue
            
        return_pct = metrics.get('return_pct', 0)
        alpha_pct = metrics.get('alpha_pct', 0)
        capture_ratio = metrics.get('capture', 0)
        exposure_avg = metrics.get('exposure_avg', 0)
        classification = metrics.get('class', 'unknown')
        
        stock_performance.append({
            'symbol': symbol,
            'return_pct': return_pct,
            'alpha_pct': alpha_pct,
            'capture_ratio': capture_ratio,
            'exposure_avg': exposure_avg,
            'classification': classification
        })
        
        total_return += return_pct
        total_alpha += alpha_pct
        
        if return_pct > 0:
            profitable_stocks.append((symbol, return_pct))
        else:
            losing_stocks.append((symbol, return_pct))
    
    # Sort by performance
    stock_performance.sort(key=lambda x: x['return_pct'], reverse=True)
    profitable_stocks.sort(key=lambda x: x[1], reverse=True)
    losing_stocks.sort(key=lambda x: x[1])
    
    # Summary statistics
    avg_return = total_return / len(runs) if runs else 0
    avg_alpha = total_alpha / len(runs) if runs else 0
    win_rate = len(profitable_stocks) / len(runs) * 100 if runs else 0
    
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print(f"   Average Return: {avg_return:.3f}%")
    print(f"   Average Alpha: {avg_alpha:.3f}%")
    print(f"   Win Rate: {win_rate:.1f}% ({len(profitable_stocks)}/{len(runs)})")
    print(f"   Best Performer: {profitable_stocks[0][0]} (+{profitable_stocks[0][1]:.3f}%)" if profitable_stocks else "   No profitable trades")
    
    if losing_stocks:
        print(f"   Worst Performer: {losing_stocks[0][0]} ({losing_stocks[0][1]:.3f}%)")
    
    # Detailed breakdown
    print(f"\n🏆 TOP PERFORMERS:")
    for i, stock in enumerate(stock_performance[:5]):
        print(f"   {i+1}. {stock['symbol']}: {stock['return_pct']:+.3f}% return, {stock['alpha_pct']:+.3f}% alpha")
        print(f"      Exposure: {stock['exposure_avg']:.1%}, Capture: {stock['capture_ratio']:.1%}, Class: {stock['classification']}")
    
    # Portfolio analysis with $100K
    initial_capital = 100000
    equal_weight_per_stock = initial_capital / len(runs)
    
    portfolio_value = initial_capital
    total_profit = 0
    
    print(f"\n💰 PORTFOLIO IMPACT ($100K INVESTMENT):")
    print(f"   Equal weight per stock: ${equal_weight_per_stock:,.0f}")
    
    for stock in stock_performance:
        stock_investment = equal_weight_per_stock
        stock_profit = stock_investment * (stock['return_pct'] / 100)
        total_profit += stock_profit
        
        print(f"   {stock['symbol']}: ${stock_investment:,.0f} → {stock['return_pct']:+.3f}% = ${stock_profit:+,.0f}")
    
    final_portfolio_value = initial_capital + total_profit
    portfolio_return = (total_profit / initial_capital) * 100
    
    print(f"\n🎯 AGGRESSIVE STRATEGY RESULTS:")
    print(f"   Initial Capital: ${initial_capital:,}")
    print(f"   Final Portfolio Value: ${final_portfolio_value:,.0f}")
    print(f"   Total Profit: ${total_profit:+,.0f}")
    print(f"   Portfolio Return: {portfolio_return:+.3f}%")
    
    # Compare to previous conservative results
    print(f"\n📊 COMPARISON TO CONSERVATIVE STRATEGY:")
    
    # Load previous conservative results if available
    try:
        with open('batch_results.json', 'r') as f:
            conservative_data = json.load(f)
        
        conservative_runs = conservative_data.get('runs', {})
        
        if conservative_runs:
            conservative_return = sum(metrics.get('return_pct', 0) for metrics in conservative_runs.values()) / len(conservative_runs)
            conservative_profit = initial_capital * (conservative_return / 100)
            
            improvement = portfolio_return - conservative_return
            profit_improvement = total_profit - conservative_profit
            
            print(f"   Conservative Return: {conservative_return:.3f}%")
            print(f"   Conservative Profit: ${conservative_profit:+,.0f}")
            print(f"   Improvement: {improvement:+.3f}% ({profit_improvement:+,.0f})")
            
            if improvement > 0:
                multiplier = portfolio_return / conservative_return if conservative_return != 0 else float('inf')
                print(f"   🚀 Aggressive strategy is {multiplier:.1f}x better!")
            else:
                print(f"   ⚠️ Aggressive strategy underperformed")
        else:
            print(f"   No conservative results found for comparison")
            
    except FileNotFoundError:
        print(f"   No conservative baseline found (batch_results.json)")
    
    # Risk analysis
    print(f"\n⚖️ RISK ANALYSIS:")
    exposures = [stock['exposure_avg'] for stock in stock_performance]
    avg_exposure = sum(exposures) / len(exposures) if exposures else 0
    
    returns = [stock['return_pct'] for stock in stock_performance]
    return_volatility = pd.Series(returns).std() if len(returns) > 1 else 0
    
    print(f"   Average Market Exposure: {avg_exposure:.1%}")
    print(f"   Return Volatility: {return_volatility:.3f}%")
    print(f"   Risk-Adjusted Return: {portfolio_return/max(return_volatility, 0.001):.2f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if portfolio_return > 0.5:
        print(f"   ✅ Aggressive strategy is delivering strong returns")
        print(f"   • Consider increasing position sizes for top performers")
        print(f"   • Monitor drawdowns during volatile periods")
    elif portfolio_return > 0.1:
        print(f"   ✅ Moderate improvement over conservative approach")
        print(f"   • Fine-tune parameters for better performance")
        print(f"   • Consider sector concentration")
    else:
        print(f"   ⚠️ Returns still modest - consider ultra-aggressive config")
        print(f"   • May need longer time horizon")
        print(f"   • Check if market conditions favor this approach")
    
    if win_rate < 60:
        print(f"   • Win rate below 60% - review signal quality")
    
    if avg_exposure < 0.7:
        print(f"   • Low market exposure - consider more aggressive entry thresholds")
    
    return {
        'portfolio_return': portfolio_return,
        'total_profit': total_profit,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'best_performer': profitable_stocks[0] if profitable_stocks else None,
        'top_stocks': stock_performance[:3]
    }

if __name__ == "__main__":
    result = analyze_aggressive_results()
    
    if result:
        print(f"\n🏁 SUMMARY:")
        print(f"Aggressive strategy returned {result['portfolio_return']:+.3f}% vs target >0.5%")
        
        if result['portfolio_return'] > 0.5:
            print(f"🎉 SUCCESS: Strategy exceeded expectations!")
        elif result['portfolio_return'] > 0.2:
            print(f"📈 IMPROVEMENT: Better than conservative baseline")
        else:
            print(f"🔧 NEEDS WORK: Consider ultra-aggressive configuration")
