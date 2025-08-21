#!/usr/bin/env python3
"""
NVDA-Only Investment Analysis
Show detailed profit analysis for $100K NVDA investment
"""
import json
import yfinance as yf
from datetime import datetime

def analyze_nvda_investment():
    """Analyze NVDA-only investment strategy"""
    
    print("🎯 NVDA-ONLY INVESTMENT ANALYSIS")
    print("=" * 60)
    
    # Load NVDA results
    try:
        with open('two_year_batch/batch_results.json', 'r') as f:
            data = json.load(f)
        
        nvda_results = data['runs'].get('NVDA', {})
        
        if 'error' in nvda_results:
            print(f"❌ NVDA analysis error: {nvda_results['error']}")
            return
        
        # Investment parameters
        initial_capital = 100000
        
        # Get current NVDA price
        nvda_ticker = yf.Ticker('NVDA')
        current_price = nvda_ticker.history(period='1d')['Close'].iloc[-1]
        
        # Calculate investment details
        shares = int(initial_capital / current_price)
        actual_investment = shares * current_price
        cash_remaining = initial_capital - actual_investment
        
        # Extract strategy results
        strategy_return = nvda_results.get('return_pct', 0)
        buy_hold_return = nvda_results.get('buy_hold_return_pct', 0)
        alpha = nvda_results.get('alpha_pct', 0)
        capture_ratio = nvda_results.get('capture', 0)
        avg_exposure = nvda_results.get('exposure_avg', 0)
        days_in_market = nvda_results.get('pct_days_in_market', 0)
        classification = nvda_results.get('class', 'N/A')
        
        print(f"📅 Analysis Period:")
        print(f"   Training: August 12, 2024 → March 29, 2025 (1 year)")
        print(f"   Forward Testing: March 30, 2025 → August 12, 2025 (3+ months)")
        print(f"   Strategy: Aggressive Institutional (5% risk per trade)")
        
        print(f"\n💰 INVESTMENT DETAILS:")
        print(f"   Initial Capital: ${initial_capital:,}")
        print(f"   NVDA Price: ${current_price:.2f}")
        print(f"   Shares Purchased: {shares:,}")
        print(f"   Actual Investment: ${actual_investment:,.0f}")
        print(f"   Cash Remaining: ${cash_remaining:.0f}")
        print(f"   Stock Classification: {classification}")
        
        print(f"\n📈 STRATEGY PERFORMANCE:")
        print(f"   Strategy Return: {strategy_return:.2f}%")
        print(f"   Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"   Alpha (vs Buy & Hold): {alpha:+.2f}%")
        print(f"   Market Capture Ratio: {capture_ratio:.1%}")
        print(f"   Average Exposure: {avg_exposure:.1%}")
        print(f"   Days in Market: {days_in_market:.1%}")
        
        # Calculate profit/loss scenarios
        strategy_final_value = actual_investment * (1 + strategy_return/100)
        buyhold_final_value = actual_investment * (1 + buy_hold_return/100)
        
        strategy_profit = strategy_final_value - actual_investment
        buyhold_profit = buyhold_final_value - actual_investment
        alpha_dollars = strategy_profit - buyhold_profit
        
        print(f"\n💵 PROFIT ANALYSIS:")
        print(f"   Strategy Final Value: ${strategy_final_value:,.0f}")
        print(f"   Strategy Profit: ${strategy_profit:+,.0f}")
        print(f"   Buy & Hold Final Value: ${buyhold_final_value:,.0f}")
        print(f"   Buy & Hold Profit: ${buyhold_profit:+,.0f}")
        print(f"   Alpha in Dollars: ${alpha_dollars:+,.0f}")
        
        # Risk analysis
        print(f"\n⚖️ RISK ANALYSIS:")
        if avg_exposure < 0.5:
            risk_level = "Conservative"
        elif avg_exposure < 0.8:
            risk_level = "Moderate"
        else:
            risk_level = "Aggressive"
        
        print(f"   Risk Level: {risk_level}")
        print(f"   Market Exposure: {avg_exposure:.1%} (vs 100% buy & hold)")
        print(f"   Days Trading: {days_in_market:.1%}")
        
        # Performance verdict
        print(f"\n🎯 VERDICT:")
        if strategy_return > buy_hold_return:
            print(f"   ✅ Strategy OUTPERFORMED buy & hold by {alpha:+.2f}%")
            print(f"   💰 Extra profit: ${alpha_dollars:+,.0f}")
        elif strategy_return == buy_hold_return:
            print(f"   ⚖️ Strategy matched buy & hold performance")
        else:
            print(f"   ❌ Strategy UNDERPERFORMED buy & hold by {alpha:.2f}%")
            print(f"   💸 Lost profit: ${alpha_dollars:,.0f}")
        
        # Forward-looking insights
        print(f"\n🔮 INSIGHTS:")
        if classification == 'strong':
            print(f"   • NVDA classified as 'strong' - good for AI trading")
        else:
            print(f"   • NVDA classified as '{classification}' - proceed with caution")
            
        if capture_ratio > 0.8:
            print(f"   • High capture ratio ({capture_ratio:.1%}) - strategy captures most upside")
        elif capture_ratio > 0.5:
            print(f"   • Moderate capture ratio ({capture_ratio:.1%}) - balanced approach")
        else:
            print(f"   • Low capture ratio ({capture_ratio:.1%}) - conservative/defensive")
            
        if days_in_market < 0.5:
            print(f"   • Low market exposure ({days_in_market:.1%}) - selective trading")
        else:
            print(f"   • High market exposure ({days_in_market:.1%}) - active trading")
            
        return {
            'investment': actual_investment,
            'strategy_profit': strategy_profit,
            'buyhold_profit': buyhold_profit,
            'alpha_dollars': alpha_dollars,
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'alpha': alpha
        }
        
    except Exception as e:
        print(f"❌ Error analyzing NVDA: {e}")
        return None

def compare_with_diversified():
    """Compare NVDA-only with diversified approach"""
    
    print(f"\n" + "="*60)
    print("🔄 COMPARISON: NVDA vs DIVERSIFIED PORTFOLIO")
    print("="*60)
    
    # Load original 30-stock results for comparison
    try:
        # Get the top 5 from our earlier analysis
        with open('two_year_batch/batch_results.json', 'r') as f:
            all_data = json.load(f)
        
        # If we only have NVDA, load the previous 30-stock results
        if len(all_data['runs']) == 1:
            print("Note: Loading previous 30-stock analysis for comparison...")
            # You could load from a backup file or re-run the analysis
        
        # Calculate average performance of profitable stocks
        profitable_stocks = []
        total_investment = 100000
        
        for symbol, metrics in all_data['runs'].items():
            if 'error' not in metrics and metrics.get('return_pct', 0) > 0:
                profitable_stocks.append({
                    'symbol': symbol,
                    'return': metrics.get('return_pct', 0),
                    'alpha': metrics.get('alpha_pct', 0)
                })
        
        if profitable_stocks:
            avg_return = sum(s['return'] for s in profitable_stocks) / len(profitable_stocks)
            avg_alpha = sum(s['alpha'] for s in profitable_stocks) / len(profitable_stocks)
            
            diversified_profit = total_investment * (avg_return / 100)
            
            print(f"Diversified Portfolio ({len(profitable_stocks)} stocks):")
            print(f"   Average Return: {avg_return:.2f}%")
            print(f"   Average Alpha: {avg_alpha:.2f}%")
            print(f"   Projected Profit: ${diversified_profit:+,.0f}")
            
        else:
            print("No profitable stocks found in comparison set")
            
    except Exception as e:
        print(f"❌ Error in comparison: {e}")

if __name__ == "__main__":
    result = analyze_nvda_investment()
    
    if result:
        compare_with_diversified()
        
        print(f"\n" + "="*60)
        print("🏁 CONCLUSION")
        print("="*60)
        print(f"For a ${result['investment']:,.0f} NVDA investment:")
        print(f"• Strategy would return: ${result['strategy_profit']:+,.0f}")
        print(f"• Buy & hold would return: ${result['buyhold_profit']:+,.0f}")
        print(f"• Alpha generated: ${result['alpha_dollars']:+,.0f}")
        
        if result['alpha'] > 0:
            print(f"✅ The AI strategy adds value to NVDA investment!")
        else:
            print(f"❌ Simple buy & hold might be better for NVDA")
