# Backtrader ML Trading System

A clean, reproducible backtesting harness built around the ML signals from `RealisticLiveTradingSystem`, using the Backtrader framework.

## Features

✅ **CLI-driven runs** with configurable parameters  
✅ **Robust metrics** including Sharpe, Calmar, Sortino ratios  
✅ **Trade blotter output** with detailed transaction logs  
✅ **ML signal integration** from your existing system  
✅ **Comprehensive analyzers** for performance evaluation  
✅ **Equity curve and drawdown plots**  
✅ **Commission & slippage modeling**  
🔄 **Alpaca integration ready** (for paper/live trading)

## Quick Start

### 1. Install Dependencies

```bash
cd backtrader_system
pip install -r requirements.txt
```

### 2. Run Basic Backtest

```bash
python backtest_runner.py
```

### 3. Custom Configuration

```bash
python backtest_runner.py \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --initial-capital 100000 \
    --output-dir ./my_results
```

## Configuration

Edit `config/backtest_config.yaml` to customize:

```yaml
# Trading Parameters
strategy:
  signal_threshold: 0.25      # Minimum signal strength
  max_positions: 15           # Maximum concurrent positions
  max_position_size: 0.40     # 40% max position size
  max_symbol_exposure: 0.15   # 15% max per symbol

# Commission & Slippage  
backtest:
  commission_bps: 5           # 5 basis points commission
  slippage_bps: 2            # 2 basis points slippage
  initial_capital: 100000.0

# ML Model Configuration
ml_config:
  min_training_days: 60
  feature_columns: [...]      # Technical indicators to use
```

## Output Files

The system generates comprehensive output in your results directory:

- **`performance_report_YYYYMMDD_HHMMSS.json`** - Complete metrics
- **`trade_blotter_YYYYMMDD_HHMMSS.csv`** - All trades with P&L
- **`equity_curve_YYYYMMDD_HHMMSS.csv`** - Daily portfolio values
- **`plot_equity.png`** - Equity curve visualization
- **`plot_drawdown.png`** - Drawdown analysis

## Key Metrics Computed

### Performance Metrics
- **CAGR** - Compound Annual Growth Rate
- **Total Return** - Overall return percentage
- **Sharpe Ratio** - Risk-adjusted return (√252 annualized)
- **Calmar Ratio** - CAGR / Max Drawdown
- **Sortino Ratio** - Return / Downside deviation

### Risk Metrics
- **Max Drawdown** - Largest peak-to-trough decline
- **Average Drawdown** - Mean drawdown over time
- **Volatility** - Annualized standard deviation

### Trade Metrics
- **Total Trades** - Number of completed trades
- **Win Rate** - Percentage of profitable trades
- **Win/Loss Ratio** - Average win divided by average loss
- **Turnover** - Portfolio turnover rate
- **Exposure %** - Average capital deployed

### ML Enhancement Metrics
- **ML Model Accuracy** - Regime classification accuracy
- **Signal Strength R²** - ML signal prediction quality
- **Enhancement Rate** - % of trades with ML enhancement

## ML Signal Integration

The system extracts and replicates the ML logic from your `RealisticLiveTradingSystem`:

### Technical Indicators
- Moving averages (5, 20, 50 day)
- RSI with normalization
- Bollinger Bands position
- MACD histogram
- Volatility measures
- Volume ratios
- Trend consistency

### ML Models
- **Regime Classification** - Trending vs ranging markets
- **Signal Strength Prediction** - Forecast signal quality
- **Daily Retraining** - Models update with new data

### Position Sizing
- **Kelly Criterion** - Optimal position sizing
- **Signal Strength Weighting** - Size based on confidence
- **Risk Management** - Maximum exposure limits

## CLI Options

```bash
python backtest_runner.py --help

Options:
  -c, --config PATH          Configuration file path
  -o, --output-dir PATH      Output directory for results
  --start-date YYYY-MM-DD    Override start date
  --end-date YYYY-MM-DD      Override end date  
  --initial-capital FLOAT    Override initial capital
  -v, --verbose              Enable verbose logging
```

## Example Usage

### Basic 1-Year Backtest
```bash
python backtest_runner.py \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

### High-Capital Aggressive Test
```bash
python backtest_runner.py \
    --initial-capital 500000 \
    --config config/aggressive_config.yaml \
    --output-dir ./aggressive_results
```

### Quick 3-Month Test
```bash
python backtest_runner.py \
    --start-date 2024-10-01 \
    --end-date 2024-12-31 \
    --output-dir ./quick_test
```

## Sample Output

```
🚀 Starting backtest...
📅 Period: 2024-01-01 to 2024-12-31
🏦 Symbols: 20
💰 Initial Capital: $100,000

📊 Downloading data for 20 symbols...
✅ AAPL: 252 bars loaded
✅ MSFT: 252 bars loaded
...

✅ Backtest completed!
📊 Final Portfolio Value: $132,450

🏆 BACKTEST RESULTS SUMMARY
==================================================
Total Return:           +32.45%
CAGR:                   +32.45%
Max Drawdown:           8.23%
Sharpe Ratio:           1.85
Calmar Ratio:           3.94
Total Trades:           145
Win Rate:               58.6%
Avg ML Regime Accuracy: 0.672
Avg ML Strength R²:     0.418

📁 Results saved to: ./results
```

## Integration with RealisticLiveTradingSystem

This Backtrader harness is designed to work alongside your existing system:

- **Signal Logic** - Exact replication of your ML signal generation
- **Risk Management** - Same position sizing and exposure rules
- **Performance Comparison** - Compare Backtrader vs native results
- **Strategy Development** - Test modifications in clean environment

## Extending the System

### Adding New Analyzers
```python
# In backtest_runner.py
cerebro.addanalyzer(YourCustomAnalyzer, _name='custom')
```

### Custom Commission Models
```python
# Create custom commission class
class MLCommission(bt.CommissionInfo):
    def _getcommission(self, size, price, pseudoexec):
        # Custom commission logic
        return commission_amount
```

### Regime Filters
```python
# Add to strategy
def check_market_regime(self):
    vix_level = self.get_vix_data()
    if vix_level > 30:
        return "HIGH_VOLATILITY"  # Reduce position sizes
    return "NORMAL"
```

## Future Enhancements

- [ ] **Streamlit Dashboard** - Interactive results viewer
- [ ] **Alpaca Integration** - Live/paper trading support  
- [ ] **Multi-timeframe** - Higher frequency signals
- [ ] **Portfolio Optimization** - Modern portfolio theory
- [ ] **Regime Filters** - VIX and HMM-based filtering
- [ ] **Walk-Forward Analysis** - Out-of-sample testing

## Troubleshooting

### Common Issues

**Import Error**: Make sure you've installed all requirements
```bash
pip install -r requirements.txt
```

**No Data**: Check your date range and internet connection
```bash
python -c "import yfinance as yf; print(yf.Ticker('AAPL').history(period='1mo'))"
```

**Memory Issues**: Reduce the number of symbols or date range for large backtests

### Logging

Enable verbose logging for debugging:
```bash
python backtest_runner.py --verbose
```

## Support

For issues or questions:
1. Check the logs in your output directory
2. Verify your configuration file syntax
3. Test with a smaller date range first
4. Compare results with your original `RealisticLiveTradingSystem`

---

**Built with ❤️ for systematic trading research**
