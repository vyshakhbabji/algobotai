# NVDA vs Cross-Sectional Momentum Strategy Analysis

## Test Summary
- **Period**: May 13, 2024 to August 13, 2024 (3 months)
- **Capital**: $10,000
- **Analysis**: Forward testing using 1-2 year historical data for training

## Results Comparison

### 1. NVDA Buy & Hold (Baseline)
- **Total Return**: 20.61%
- **Final Value**: $12,060.86
- **Start Price**: $90.36
- **End Price**: $108.99
- **Strategy**: Simple buy and hold for the entire period

### 2. Diversified Momentum Strategy (Cross-Sectional Approach)
- **Total Return**: 1.22%
- **Final Value**: $10,122.09
- **Sharpe Ratio**: 0.24
- **Max Drawdown**: -7.64%
- **Stocks Used**: 10 top-performing stocks from scan (NFLX, PM, T, BK, WMT, AVGO, COST, GE, V, JPM)
- **Strategy**: Equal-weighted momentum strategy across multiple stocks

### 3. NVDA Single Stock Momentum Strategy
- **Total Return**: -2.60%
- **Final Value**: $9,739.77
- **Sharpe Ratio**: -0.14
- **Max Drawdown**: -13.14%
- **Strategy**: Momentum-based trading on NVDA only

## Key Findings

1. **NVDA Buy & Hold was the clear winner** with a 20.61% return, significantly outperforming both momentum strategies.

2. **Cross-sectional momentum strategy provided modest positive returns** (1.22%) with better risk-adjusted returns (Sharpe ratio 0.24) and lower drawdown (-7.64%) compared to single-stock momentum.

3. **NVDA single-stock momentum strategy underperformed**, losing 2.60% due to mistimed entries and exits during a volatile period.

4. **Diversification benefits were evident**: The cross-sectional approach had:
   - Lower volatility
   - Better risk-adjusted returns
   - Smaller maximum drawdown
   - More consistent performance

## Analysis Notes

- The test period (May-August 2024) was particularly strong for NVDA, making buy-and-hold difficult to beat
- The momentum strategies suffered from whipsaws during volatile periods
- The cross-sectional approach's diversification helped reduce risk even when individual timing was poor
- Using 1-2 year historical data for training provided sufficient sample size for the ML models

## Ranking by Performance

1. **NVDA Buy & Hold**: 20.61% ($12,060.86)
2. **Diversified Momentum**: 1.22% ($10,122.09) 
3. **NVDA Momentum**: -2.60% ($9,739.77)

## Conclusion

For the tested 3-month period with $10k capital:
- **NVDA buy-and-hold was the best strategy** in this bull run period
- **Cross-sectional momentum showed its value in risk management** with positive returns and better risk metrics
- **Single-stock momentum strategies are riskier** and more prone to timing errors
- **Diversification matters** even when individual stock selection is good
