#!/usr/bin/env python3
"""
Create visual profit analysis charts for the 30-stock analysis
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_profit_charts():
    # Load the results
    with open('two_year_batch/batch_results.json', 'r') as f:
        data = json.load(f)

    # Extract results and create summary
    results = []
    for symbol, metrics in data['runs'].items():
        if 'error' not in metrics:
            results.append({
                'Symbol': symbol,
                'Total Return (%)': metrics.get('return_pct', 0),
                'Buy & Hold (%)': metrics.get('buy_hold_return_pct', 0), 
                'Alpha (%)': metrics.get('alpha_pct', 0),
                'Market Capture (%)': round(metrics.get('capture', 0) * 100, 1),
                'Avg Exposure (%)': round(metrics.get('exposure_avg', 0) * 100, 1),
                'Days in Market (%)': round(metrics.get('pct_days_in_market', 0) * 100, 1),
                'Classification': metrics.get('class', 'N/A'),
                'Alpha ($)': metrics.get('alpha_dollars', 0)
            })

    # Sort by Total Return descending
    results_df = pd.DataFrame(results).sort_values('Total Return (%)', ascending=False)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('🎯 30-Stock Trading Strategy Analysis\nTraining: 1 Year (2024-08-12 to 2025-03-29) | Forward Testing: 3+ Months (2025-03-30 to 2025-08-12)', 
                 fontsize=16, fontweight='bold')

    # Chart 1: Strategy Returns vs Buy & Hold
    ax1.scatter(results_df['Buy & Hold (%)'], results_df['Total Return (%)'], 
                c=results_df['Avg Exposure (%)'], s=100, alpha=0.7, cmap='viridis')
    ax1.plot([-0.5, 0.5], [-0.5, 0.5], 'r--', alpha=0.5, label='Equal Performance Line')
    ax1.set_xlabel('Buy & Hold Return (%)')
    ax1.set_ylabel('Strategy Return (%)')
    ax1.set_title('Strategy vs Buy & Hold Performance\n(Color = Average Exposure %)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add colorbar for exposure
    scatter = ax1.scatter(results_df['Buy & Hold (%)'], results_df['Total Return (%)'], 
                         c=results_df['Avg Exposure (%)'], s=100, alpha=0.7, cmap='viridis')
    plt.colorbar(scatter, ax=ax1, label='Average Exposure (%)')

    # Chart 2: Alpha Distribution
    colors = ['green' if x > 0 else 'red' for x in results_df['Alpha (%)']]
    bars = ax2.bar(range(len(results_df)), results_df['Alpha (%)'], color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Stock Rank (by Total Return)')
    ax2.set_ylabel('Alpha (%)')
    ax2.set_title('Alpha Distribution Across All Stocks')
    ax2.grid(True, alpha=0.3)
    
    # Add stock symbols on bars
    for i, (bar, symbol) in enumerate(zip(bars, results_df['Symbol'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                symbol, ha='center', va='bottom' if height >= 0 else 'top', fontsize=8, rotation=45)

    # Chart 3: Top 10 Performance Comparison
    top_10 = results_df.head(10)
    x_pos = np.arange(len(top_10))
    
    width = 0.35
    bars1 = ax3.bar(x_pos - width/2, top_10['Total Return (%)'], width, label='Strategy Return', alpha=0.8, color='blue')
    bars2 = ax3.bar(x_pos + width/2, top_10['Buy & Hold (%)'], width, label='Buy & Hold', alpha=0.8, color='orange')
    
    ax3.set_xlabel('Top 10 Stocks')
    ax3.set_ylabel('Return (%)')
    ax3.set_title('Top 10 Performers: Strategy vs Buy & Hold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(top_10['Symbol'], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Chart 4: Risk-Return Analysis
    # Calculate hypothetical $10K investment outcomes
    investment_amount = 10000
    final_values = [(investment_amount * (1 + ret/100)) for ret in results_df['Total Return (%)']]
    profits = [fv - investment_amount for fv in final_values]
    
    colors = ['green' if p > 0 else 'red' for p in profits]
    bars = ax4.bar(range(len(results_df)), profits, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_xlabel('Stock Rank (by Total Return)')
    ax4.set_ylabel('Profit/Loss on $10K Investment ($)')
    ax4.set_title('Profit/Loss Analysis ($10K per Stock)')
    ax4.grid(True, alpha=0.3)
    
    # Add total portfolio summary text
    total_profit = sum(profits)
    profitable_count = sum(1 for p in profits if p > 0)
    avg_profit = total_profit / len(profits)
    
    summary_text = f"""Portfolio Summary (30 stocks × $10K each):
Total Investment: ${len(profits) * investment_amount:,}
Total Profit: ${total_profit:,.0f}
Average per Stock: ${avg_profit:.0f}
Profitable Stocks: {profitable_count}/{len(profits)} ({profitable_count/len(profits)*100:.1f}%)"""
    
    ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('30_stock_analysis_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Charts saved as '30_stock_analysis_charts.png'")
    
    # Print detailed profit analysis
    print("\n" + "="*80)
    print("💰 DETAILED PROFIT ANALYSIS")
    print("="*80)
    
    print(f"\n📊 If you invested $10,000 in each of the 30 stocks:")
    print(f"Total Investment: ${len(results_df) * 10000:,}")
    print(f"Total Final Value: ${sum(final_values):,.0f}")
    print(f"Total Profit: ${sum(profits):,.0f}")
    print(f"Overall Return: {(sum(profits)/(len(results_df)*10000))*100:.2f}%")
    
    print(f"\n🏆 Top 5 Most Profitable:")
    for i in range(5):
        row = results_df.iloc[i]
        profit = profits[i]
        print(f"  {i+1}. {row['Symbol']:>6}: ${profit:>+6.0f} ({row['Total Return (%)']:+.2f}%)")
    
    print(f"\n📉 Bottom 5 Performers:")
    for i in range(5):
        idx = len(results_df) - 5 + i
        row = results_df.iloc[idx]
        profit = profits[idx]
        print(f"  {i+1}. {row['Symbol']:>6}: ${profit:>+6.0f} ({row['Total Return (%)']:+.2f}%)")
    
    print(f"\n🎯 Strategy Effectiveness:")
    print(f"Stocks beating Buy & Hold: {sum(1 for _, row in results_df.iterrows() if row['Alpha (%)'] > 0)}/{len(results_df)}")
    print(f"Average Alpha: {results_df['Alpha (%)'].mean():.3f}%")
    print(f"Average Market Capture: {results_df['Market Capture (%)'].mean():.1f}%")
    print(f"Average Exposure: {results_df['Avg Exposure (%)'].mean():.1f}%")

if __name__ == "__main__":
    create_profit_charts()
