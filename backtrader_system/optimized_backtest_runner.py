"""
Optimized Backtest Runner - Implements analysis recommendations

Key Improvements:
1. Conservative Kelly position sizing (15-25% max instead of 35-60%)
2. Stop-loss protection (8% stop loss)
3. Portfolio heat limits (25% max risk)
4. Higher conviction signals only (threshold 0.35 vs 0.25)
5. Reduced turnover with min hold periods
6. Daily trade limits to control activity
"""

import argparse
import os
import sys
import logging
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
import backtrader.analyzers as btanalyzers
import yfinance as yf
import pandas as pd

from strategies.optimized_ml_strategy import OptimizedMLTradingStrategy
from analyzers.performance_plotter import PerformancePlotter


class OptimizedBacktestRunner:
    """Run optimized backtests with improved risk management"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_filename = f"optimized_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*50)
        self.logger.info("OPTIMIZED BACKTRADER ML SYSTEM")
        self.logger.info("="*50)
        self.logger.info("🎯 Key Improvements:")
        self.logger.info("   • Conservative Kelly sizing (12% max position)")
        self.logger.info("   • 8% stop-loss protection")
        self.logger.info("   • 25% portfolio heat limit")
        self.logger.info("   • Higher signal threshold (0.35)")
        self.logger.info("   • Minimum 3-day hold periods")
        self.logger.info("   • Maximum 8 trades per day")
        self.logger.info("="*50)
    
    def download_data(self, symbols: list, start_date: str, end_date: str) -> dict:
        """Download price data for all symbols"""
        self.logger.info(f"📊 Downloading data for {len(symbols)} symbols...")
        
        data_dict = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                self.logger.info(f"   Downloading {symbol}...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                
                if df.empty or len(df) < 100:
                    self.logger.warning(f"   ⚠️  Insufficient data for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                # Ensure we have required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_columns):
                    self.logger.warning(f"   ⚠️  Missing required columns for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                data_dict[symbol] = df
                self.logger.info(f"   ✅ {symbol}: {len(df)} bars")
                
            except Exception as e:
                self.logger.error(f"   ❌ Failed to download {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            self.logger.warning(f"⚠️  Failed symbols: {failed_symbols}")
        
        self.logger.info(f"✅ Successfully downloaded {len(data_dict)} symbols")
        return data_dict
    
    def create_config(self) -> dict:
        """Create optimized configuration"""
        return {
            'ml_config': {
                'regime_features': [
                    'sma_10', 'sma_50', 'rsi_14', 'bb_upper', 'bb_lower',
                    'macd', 'macd_signal', 'volume_sma_20', 'atr_14'
                ],
                'strength_features': [
                    'rsi_14', 'macd_histogram', 'bb_position', 'volume_ratio',
                    'price_momentum', 'volatility_ratio'
                ],
                'regime_model': 'random_forest',
                'strength_model': 'gradient_boosting',
                'regime_params': {
                    'n_estimators': 100,
                    'max_depth': 8,
                    'min_samples_split': 10,
                    'random_state': 42
                },
                'strength_params': {
                    'n_estimators': 150,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'signal_config': {
                'regime_weight': 0.6,
                'strength_weight': 0.4,
                'momentum_lookback': 20,
                'volume_threshold': 1.2,
                'rsi_overbought': 70,
                'rsi_oversold': 30
            }
        }
    
    def run_optimized_backtest(self, symbols: list, start_date: str = "2022-08-22", 
                              end_date: str = "2025-08-22") -> dict:\n        \"\"\"Run the optimized backtest\"\"\"\n        self.logger.info(f\"🚀 Starting OPTIMIZED backtest...\")\n        self.logger.info(f\"   Symbols: {len(symbols)}\")\n        self.logger.info(f\"   Period: {start_date} to {end_date}\")\n        self.logger.info(f\"   Capital: ${self.initial_capital:,.0f}\")\n        \n        # Download data\n        data_dict = self.download_data(symbols, start_date, end_date)\n        \n        if len(data_dict) < 5:\n            raise ValueError(f\"Insufficient data: only {len(data_dict)} symbols available\")\n        \n        # Create Cerebro instance\n        cerebro = bt.Cerebro()\n        \n        # Set optimized broker settings\n        cerebro.broker.setcash(self.initial_capital)\n        cerebro.broker.setcommission(commission=0.001)  # 0.1% commission\n        \n        # Add optimized strategy with configuration\n        config = self.create_config()\n        cerebro.addstrategy(OptimizedMLTradingStrategy, config=config)\n        \n        # Add data feeds\n        for symbol, df in data_dict.items():\n            data_feed = bt.feeds.PandasData(\n                dataname=df,\n                name=symbol,\n                plot=False\n            )\n            cerebro.adddata(data_feed)\n        \n        # Add analyzers\n        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')\n        cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')\n        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')\n        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')\n        cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')\n        \n        # Add custom performance analyzer\n        cerebro.addanalyzer(PerformancePlotter, _name='performance')\n        \n        self.logger.info(\"⚙️  Running optimized backtest...\")\n        \n        # Run the backtest\n        results = cerebro.run(maxcpus=1, tradehistory=True)\n        strategy = results[0]\n        \n        # Get final portfolio value\n        final_value = cerebro.broker.getvalue()\n        total_return = (final_value - self.initial_capital) / self.initial_capital\n        \n        self.logger.info(\"✅ Backtest completed!\")\n        self.logger.info(f\"📊 Final Portfolio Value: ${final_value:,.0f}\")\n        self.logger.info(f\"📈 Total Return: {total_return:.2%}\")\n        \n        # Extract trade blotter\n        trade_blotter = self.extract_trade_blotter(strategy)\n        \n        # Save results\n        results_data = self.save_results(strategy, final_value, total_return, trade_blotter)\n        \n        # Generate performance analysis\n        self.logger.info(\"🎯 GENERATING OPTIMIZED PERFORMANCE ANALYSIS...\")\n        performance_analyzer = strategy.analyzers.performance\n        if hasattr(performance_analyzer, 'create_complete_analysis'):\n            performance_analyzer.create_complete_analysis()\n        \n        return results_data\n    \n    def extract_trade_blotter(self, strategy) -> pd.DataFrame:\n        \"\"\"Extract detailed trade blotter from strategy\"\"\"\n        if not hasattr(strategy, 'trade_log') or not strategy.trade_log:\n            return pd.DataFrame()\n        \n        # Convert trade log to DataFrame\n        df = pd.DataFrame(strategy.trade_log)\n        \n        # Save trade blotter\n        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n        filename = f\"results/optimized_trade_blotter_{timestamp}.csv\"\n        \n        os.makedirs('results', exist_ok=True)\n        df.to_csv(filename, index=False)\n        \n        self.logger.info(f\"📋 Trade blotter saved: {filename}\")\n        self.logger.info(f\"📊 Total trades: {len(df)}\")\n        \n        return df\n    \n    def save_results(self, strategy, final_value: float, total_return: float, \n                    trade_blotter: pd.DataFrame) -> dict:\n        \"\"\"Save backtest results\"\"\"\n        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n        \n        # Create results directory\n        os.makedirs('results', exist_ok=True)\n        \n        # Extract daily portfolio values for equity curve\n        equity_data = []\n        if hasattr(strategy, 'daily_values'):\n            equity_data = strategy.daily_values\n        \n        # Calculate basic metrics\n        trades_df = trade_blotter[trade_blotter['action'] == 'BUY'] if not trade_blotter.empty else pd.DataFrame()\n        total_trades = len(trades_df)\n        \n        # Win rate calculation\n        if not trade_blotter.empty and 'pnl' in trade_blotter.columns:\n            profitable_trades = len(trade_blotter[trade_blotter['pnl'] > 0])\n            win_rate = profitable_trades / len(trade_blotter) if len(trade_blotter) > 0 else 0\n        else:\n            win_rate = 0\n        \n        # ML metrics\n        ml_metrics = getattr(strategy, 'ml_metrics', {})\n        avg_regime_accuracy = 0\n        avg_strength_r2 = 0\n        \n        if ml_metrics:\n            regime_accuracies = []\n            strength_r2s = []\n            \n            for symbol, metrics in ml_metrics.items():\n                if isinstance(metrics, dict):\n                    if 'regime_accuracy' in metrics:\n                        regime_accuracies.append(metrics['regime_accuracy'])\n                    if 'strength_r2' in metrics:\n                        strength_r2s.append(metrics['strength_r2'])\n            \n            avg_regime_accuracy = sum(regime_accuracies) / len(regime_accuracies) if regime_accuracies else 0\n            avg_strength_r2 = sum(strength_r2s) / len(strength_r2s) if strength_r2s else 0\n        \n        # Create results summary\n        results_summary = {\n            'timestamp': datetime.now().isoformat(),\n            'backtest_type': 'OPTIMIZED',\n            'initial_capital': self.initial_capital,\n            'final_value': final_value,\n            'total_return_pct': total_return * 100,\n            'total_trades': total_trades,\n            'win_rate_pct': win_rate * 100,\n            'avg_regime_accuracy': avg_regime_accuracy,\n            'avg_strength_r2': avg_strength_r2,\n            'optimizations_applied': [\n                'Conservative Kelly sizing (12% max)',\n                '8% stop-loss protection',\n                '25% portfolio heat limit', \n                'Higher signal threshold (0.35)',\n                'Minimum 3-day hold periods',\n                'Maximum 8 trades per day',\n                'Reduced max positions to 12',\n                'Profit targets at 15%'\n            ]\n        }\n        \n        # Save equity curve\n        if equity_data:\n            equity_df = pd.DataFrame(equity_data)\n            equity_filename = f\"results/optimized_equity_curve_{timestamp}.csv\"\n            equity_df.to_csv(equity_filename, index=False)\n            self.logger.info(f\"📈 Equity curve saved: {equity_filename}\")\n        \n        # Save results summary\n        results_filename = f\"results/optimized_performance_report_{timestamp}.json\"\n        with open(results_filename, 'w') as f:\n            json.dump(results_summary, f, indent=2)\n        \n        self.logger.info(f\"📊 Performance report: {results_filename}\")\n        \n        return results_summary\n\n\ndef main():\n    \"\"\"Main function for running optimized backtests\"\"\"\n    parser = argparse.ArgumentParser(description='Optimized Backtrader ML Trading System')\n    parser.add_argument('--initial-capital', type=float, default=1000000,\n                       help='Initial capital (default: 1000000)')\n    parser.add_argument('--symbols', nargs='+', \n                       default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', \n                               'ORCL', 'JPM', 'BAC', 'V', 'MA', 'UNH', 'JNJ', 'PG', \n                               'KO', 'XOM', 'HD', 'DIS', 'CRM'],\n                       help='Symbols to trade')\n    parser.add_argument('--start-date', default='2022-08-22',\n                       help='Start date (YYYY-MM-DD)')\n    parser.add_argument('--end-date', default='2025-08-22',\n                       help='End date (YYYY-MM-DD)')\n    \n    args = parser.parse_args()\n    \n    # Create and run optimized backtest\n    runner = OptimizedBacktestRunner(args.initial_capital)\n    \n    try:\n        results = runner.run_optimized_backtest(\n            symbols=args.symbols,\n            start_date=args.start_date,\n            end_date=args.end_date\n        )\n        \n        print(\"\\n\" + \"=\"*60)\n        print(\"🏆 OPTIMIZED BACKTEST RESULTS SUMMARY\")\n        print(\"=\"*60)\n        print(f\"Total Return:           {results['total_return_pct']:.2f}%\")\n        print(f\"Total Trades:           {results['total_trades']}\")\n        print(f\"Win Rate:               {results['win_rate_pct']:.1f}%\")\n        print(f\"Avg ML Regime Accuracy: {results['avg_regime_accuracy']:.3f}\")\n        print(f\"Avg ML Strength R²:     {results['avg_strength_r2']:.3f}\")\n        print(\"\\n🎯 Applied Optimizations:\")\n        for opt in results['optimizations_applied']:\n            print(f\"   • {opt}\")\n        print(f\"\\n📁 Results saved to: results\")\n        print(\"=\"*60)\n        \n    except Exception as e:\n        logging.error(f\"Backtest failed: {e}\")\n        raise\n\n\nif __name__ == '__main__':\n    main()"
