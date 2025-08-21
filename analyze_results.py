#!/usr/bin/env python3
"""
Analyze the 30-stock batch results and show comprehensive profit analysis
"""
import json
import pandas as pd

def analyze_results():
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

    print('🎯 COMPREHENSIVE 30-STOCK ANALYSIS RESULTS')
    print('=' * 80)
    print(f'Training Period: 2024-08-12 to 2025-03-29 (1 year)')
    print(f'Forward Testing: 2025-03-30 to 2025-08-12 (3+ months)')
    print(f'Strategy: Aggressive Institutional with 5% risk per trade')
    print('=' * 80)

    print('\n📊 TOP PERFORMING STOCKS:')
    print('-' * 80)
    top_10 = results_df.head(10)
    for i, row in top_10.iterrows():
        print(f'{row["Symbol"]:>6} | Return: {row["Total Return (%)"]:>7.1f}% | Alpha: {row["Alpha (%)"]:>6.1f}% | B&H: {row["Buy & Hold (%)"]:>6.1f}% | Exposure: {row["Avg Exposure (%)"]:>5.1f}%')

    print('\n📈 PORTFOLIO SUMMARY:')
    print('-' * 40)
    total_returns = results_df['Total Return (%)'].tolist()
    buy_hold_returns = results_df['Buy & Hold (%)'].tolist()
    alpha_values = results_df['Alpha (%)'].tolist()

    profitable_count = sum(1 for r in total_returns if r > 0)
    avg_return = sum(total_returns) / len(total_returns)
    avg_buy_hold = sum(buy_hold_returns) / len(buy_hold_returns)
    avg_alpha = sum(alpha_values) / len(alpha_values)
    best_stock = results_df.iloc[0]
    worst_stock = results_df.iloc[-1]

    print(f'Profitable Stocks: {profitable_count}/{len(results_df)} ({profitable_count/len(results_df)*100:.1f}%)')
    print(f'Average Return: {avg_return:.1f}%')
    print(f'Average Buy & Hold: {avg_buy_hold:.1f}%') 
    print(f'Average Alpha: {avg_alpha:.1f}%')
    print(f'Best Performer: {best_stock["Symbol"]} ({best_stock["Total Return (%)"]:+.1f}%)')
    print(f'Worst Performer: {worst_stock["Symbol"]} ({worst_stock["Total Return (%)"]:+.1f}%)')

    # Calculate portfolio value if invested equally across top 10
    initial_capital = 100000  # $100K
    position_size = initial_capital / 10  # Equal weight
    top_10_total_return = sum(top_10['Total Return (%)']) / 10
    portfolio_final_value = initial_capital * (1 + top_10_total_return/100)
    portfolio_profit = portfolio_final_value - initial_capital

    print(f'\n💰 HYPOTHETICAL PORTFOLIO PERFORMANCE (Top 10 Equal Weight):')
    print(f'Initial Capital: ${initial_capital:,}')
    print(f'Final Value: ${portfolio_final_value:,.0f}')
    print(f'Total Profit: ${portfolio_profit:,.0f}')
    print(f'Portfolio Return: {top_10_total_return:.1f}%')

    print('\n🏆 CLASSIFICATION BREAKDOWN:')
    classifications = results_df['Classification'].value_counts()
    for class_name, count in classifications.items():
        print(f'{class_name}: {count} stocks')

    print('\n📋 DETAILED RESULTS (All 30 Stocks):')
    print('-' * 100)
    print(f'{"Symbol":>6} | {"Return%":>8} | {"Alpha%":>7} | {"B&H%":>7} | {"Capture%":>9} | {"Exposure%":>10} | {"Class":>8}')
    print('-' * 100)
    for i, row in results_df.iterrows():
        print(f'{row["Symbol"]:>6} | {row["Total Return (%)"]:>7.1f}% | {row["Alpha (%)"]:>6.1f}% | {row["Buy & Hold (%)"]:>6.1f}% | {row["Market Capture (%)"]:>8.1f}% | {row["Avg Exposure (%)"]:>9.1f}% | {row["Classification"]:>8}')

    return results_df

if __name__ == "__main__":
    analyze_results()
