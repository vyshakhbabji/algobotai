#!/usr/bin/env python3
"""
Backtrader ML Trading System CLI
Clean, reproducible backtesting harness for RealisticLiveTradingSystem ML signals
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add parent directory to path to import realistic_live_trading_system
sys.path.append(str(Path(__file__).parent.parent))

try:
    import backtrader as bt
    import yfinance as yf
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Please install: pip install backtrader yfinance matplotlib pyyaml")
    sys.exit(1)

from strategies.ml_strategy import MLTradingStrategy
from strategies.signal_generator import MLSignalGenerator
from analyzers.custom_analyzers import (
    SharpeAnalyzer, CustomMetricsAnalyzer, 
    TurnoverAnalyzer, ExposureAnalyzer
)


def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        sys.exit(1)


def create_data_feeds(symbols: list, start_date: str, end_date: str):
    """Create Backtrader data feeds from Yahoo Finance"""
    data_feeds = []
    
    print(f"📊 Downloading data for {len(symbols)} symbols...")
    
    for symbol in symbols:
        try:
            # Download data with some buffer for indicators
            buffer_start = pd.to_datetime(start_date) - timedelta(days=100)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=buffer_start.strftime('%Y-%m-%d'),
                end=end_date,
                auto_adjust=True
            )
            
            if data.empty:
                print(f"⚠️  No data for {symbol}, skipping...")
                continue
            
            # Create Backtrader data feed
            data_feed = bt.feeds.PandasData(
                dataname=data,
                name=symbol,
                fromdate=pd.to_datetime(start_date),
                todate=pd.to_datetime(end_date)
            )
            
            data_feeds.append(data_feed)
            print(f"✅ {symbol}: {len(data)} bars loaded")
            
        except Exception as e:
            print(f"❌ Error loading {symbol}: {e}")
            continue
    
    print(f"📈 Successfully loaded {len(data_feeds)} data feeds")
    return data_feeds


def setup_broker(cerebro, config: dict):
    """Setup broker with commission and slippage"""
    # Set initial capital
    initial_capital = config['backtest']['initial_capital']
    cerebro.broker.setcash(initial_capital)
    
    # Set commission (convert bps to percentage)
    commission_pct = config['backtest']['commission_bps'] / 10000.0
    cerebro.broker.setcommission(commission=commission_pct)
    
    # Add slippage (approximate with spread)
    slippage_pct = config['backtest']['slippage_bps'] / 10000.0
    # Note: Backtrader doesn't have built-in slippage, we'll handle in strategy
    
    print(f"💰 Initial Capital: ${initial_capital:,.0f}")
    print(f"💸 Commission: {commission_pct*100:.3f}%")
    print(f"📉 Slippage: {slippage_pct*100:.3f}%")


def save_trade_blotter(strategy, output_dir: Path):
    """Save detailed trade blotter to CSV"""
    if not hasattr(strategy, 'trade_log') or not strategy.trade_log:
        print("⚠️  No trades to save")
        return
    
    trades_df = pd.DataFrame(strategy.trade_log)
    
    # Add additional columns
    trades_df['gross_pnl'] = trades_df['pnl']
    trades_df['net_pnl'] = trades_df['pnl'] - trades_df['fees']
    
    # Save to CSV
    blotter_file = output_dir / f"trade_blotter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    trades_df.to_csv(blotter_file, index=False)
    
    print(f"📋 Trade blotter saved: {blotter_file}")
    print(f"📊 Total trades: {len(trades_df)}")
    
    return blotter_file


def save_equity_curve(strategy, output_dir: Path):
    """Save equity curve data and plot"""
    if not hasattr(strategy, 'daily_values') or not strategy.daily_values:
        print("⚠️  No equity curve data to save")
        return
    
    equity_df = pd.DataFrame(strategy.daily_values)
    equity_df.set_index('date', inplace=True)
    
    # Save CSV
    equity_file = output_dir / f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    equity_df.to_csv(equity_file)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot equity curve
    plt.subplot(2, 1, 1)
    plt.plot(equity_df.index, equity_df['portfolio_value'], linewidth=2, color='blue')
    plt.title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Plot drawdown
    plt.subplot(2, 1, 2)
    peak = equity_df['portfolio_value'].expanding().max()
    drawdown = (equity_df['portfolio_value'] - peak) / peak * 100
    
    plt.fill_between(equity_df.index, drawdown, 0, color='red', alpha=0.3)
    plt.plot(equity_df.index, drawdown, color='red', linewidth=1)
    plt.title('Drawdown', fontsize=14, fontweight='bold')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / "plot_equity.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Equity curve saved: {equity_file}")
    print(f"📊 Equity plot saved: {plot_file}")
    
    return equity_file, plot_file


def save_drawdown_plot(analyzers, output_dir: Path):
    """Save standalone drawdown plot"""
    try:
        drawdown_analyzer = analyzers.get('drawdown')
        if not drawdown_analyzer:
            print("⚠️  No drawdown analyzer found")
            return
        
        # Get drawdown data
        dd_analysis = drawdown_analyzer.get_analysis()
        
        # Create simple drawdown plot
        plt.figure(figsize=(12, 6))
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Portfolio Drawdown Analysis', fontsize=14, fontweight='bold')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Time')
        
        # Add max drawdown annotation
        max_dd = dd_analysis.get('max', {}).get('drawdown', 0)
        plt.axhline(y=-max_dd*100, color='red', linestyle='--', alpha=0.7, 
                   label=f'Max Drawdown: {max_dd*100:.1f}%')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_file = output_dir / "plot_drawdown.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📉 Drawdown plot saved: {plot_file}")
        return plot_file
        
    except Exception as e:
        print(f"Error creating drawdown plot: {e}")
        return None


def format_performance_report(analyzers: dict, strategy) -> dict:
    """Format comprehensive performance report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'risk_metrics': {},
        'trade_metrics': {},
        'ml_metrics': {}
    }
    
    # Get analyzer results
    returns = analyzers.get('returns')
    drawdown = analyzers.get('drawdown') 
    trades = analyzers.get('trades')
    sharpe = analyzers.get('sharpe')
    custom = analyzers.get('custom_metrics')
    turnover = analyzers.get('turnover')
    exposure = analyzers.get('exposure')
    
    # Summary metrics
    if returns:
        ret_analysis = returns.get_analysis()
        report['summary'].update({
            'total_return_pct': ret_analysis.get('rtot', 0) * 100,
            'annualized_return_pct': ret_analysis.get('rnorm', 0) * 100
        })
    
    if custom:
        custom_analysis = custom.get_analysis()
        report['summary'].update(custom_analysis)
    
    # Risk metrics
    if sharpe:
        sharpe_analysis = sharpe.get_analysis()
        report['risk_metrics'].update(sharpe_analysis)
    
    if drawdown:
        dd_analysis = drawdown.get_analysis()
        max_dd = dd_analysis.get('max', {})
        report['risk_metrics'].update({
            'max_drawdown_pct': max_dd.get('drawdown', 0) * 100,
            'max_drawdown_length': max_dd.get('len', 0)
        })
    
    # Trade metrics
    if trades:
        trade_analysis = trades.get_analysis()
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        
        report['trade_metrics'].update({
            'total_trades': total_trades,
            'winning_trades': won_trades,
            'losing_trades': total_trades - won_trades,
            'win_rate_pct': (won_trades / total_trades * 100) if total_trades > 0 else 0,
            'avg_win_pct': trade_analysis.get('won', {}).get('pnl', {}).get('average', 0),
            'avg_loss_pct': trade_analysis.get('lost', {}).get('pnl', {}).get('average', 0)
        })
        
        # Win/Loss ratio
        avg_win = trade_analysis.get('won', {}).get('pnl', {}).get('average', 0)
        avg_loss = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('average', 0))
        report['trade_metrics']['win_loss_ratio'] = avg_win / avg_loss if avg_loss > 0 else 0
    
    # Portfolio metrics
    if turnover:
        turnover_analysis = turnover.get_analysis()
        report['trade_metrics'].update(turnover_analysis)
    
    if exposure:
        exposure_analysis = exposure.get_analysis()
        report['trade_metrics'].update(exposure_analysis)
    
    # ML metrics
    if hasattr(strategy, 'signal_generator') and hasattr(strategy.signal_generator, 'model_evolution'):
        ml_stats = {}
        for symbol, evolution in strategy.signal_generator.model_evolution.items():
            if evolution:
                final_metrics = evolution[-1]
                ml_stats[symbol] = {
                    'final_regime_accuracy': final_metrics.get('regime_accuracy', 0),
                    'final_strength_r2': final_metrics.get('strength_r2', 0),
                    'training_sessions': len(evolution)
                }
        
        if ml_stats:
            # Calculate averages
            avg_regime_acc = np.mean([s['final_regime_accuracy'] for s in ml_stats.values()])
            avg_strength_r2 = np.mean([s['final_strength_r2'] for s in ml_stats.values()])
            
            report['ml_metrics'] = {
                'avg_regime_accuracy': avg_regime_acc,
                'avg_strength_r2': avg_strength_r2,
                'symbols_with_models': len(ml_stats),
                'symbol_stats': ml_stats
            }
    
    return report


def run_backtest(config: dict) -> dict:
    """Main backtest execution"""
    
    # Setup
    cerebro = bt.Cerebro()
    
    # Get symbols from config or use elite stocks
    if 'symbols' in config['backtest'] and config['backtest']['symbols']:
        symbols = config['backtest']['symbols']
    else:
        signal_gen = MLSignalGenerator(config.get('ml_config', {}))
        symbols = signal_gen.get_elite_stocks()
    
    # Create data feeds
    data_feeds = create_data_feeds(
        symbols, 
        config['backtest']['start_date'],
        config['backtest']['end_date']
    )
    
    if not data_feeds:
        raise ValueError("No data feeds created. Check symbols and date range.")
    
    # Add data to cerebro
    for data_feed in data_feeds:
        cerebro.adddata(data_feed)
    
    # Setup broker
    setup_broker(cerebro, config)
    
    # Add strategy
    strategy_params = config.get('strategy', {})
    strategy_params['config'] = config.get('ml_config', {})
    
    cerebro.addstrategy(MLTradingStrategy, **strategy_params)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(SharpeAnalyzer, _name='sharpe')
    cerebro.addanalyzer(CustomMetricsAnalyzer, _name='custom_metrics')
    cerebro.addanalyzer(TurnoverAnalyzer, _name='turnover')
    cerebro.addanalyzer(ExposureAnalyzer, _name='exposure')
    
    print(f"\n🚀 Starting backtest...")
    print(f"📅 Period: {config['backtest']['start_date']} to {config['backtest']['end_date']}")
    print(f"🏦 Symbols: {len(data_feeds)}")
    print(f"💰 Initial Capital: ${config['backtest']['initial_capital']:,.0f}")
    
    # Run backtest
    results = cerebro.run()
    strategy = results[0]
    
    # Extract analyzer results
    analyzer_results = {}
    for name in ['returns', 'drawdown', 'trades', 'sharpe', 'custom_metrics', 'turnover', 'exposure']:
        if hasattr(strategy.analyzers, name):
            analyzer_results[name] = getattr(strategy.analyzers, name)
    
    print(f"\n✅ Backtest completed!")
    print(f"📊 Final Portfolio Value: ${cerebro.broker.getvalue():,.0f}")
    
    return {
        'strategy': strategy,
        'analyzers': analyzer_results,
        'cerebro': cerebro
    }


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Backtrader ML Trading System - Clean backtesting harness'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/backtest_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='./results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Stock symbols to backtest (overrides config file)'
    )
    
    parser.add_argument(
        '--start-date',
        help='Override start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date', 
        help='Override end date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        help='Override initial capital'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)
    
    # Load configuration
    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)
    
    # Apply CLI overrides
    if args.symbols:
        config['backtest']['symbols'] = args.symbols
    if args.start_date:
        config['backtest']['start_date'] = args.start_date
    if args.end_date:
        config['backtest']['end_date'] = args.end_date
    if args.initial_capital:
        config['backtest']['initial_capital'] = args.initial_capital
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Run backtest
        results = run_backtest(config)
        strategy = results['strategy']
        analyzers = results['analyzers']
        
        # Generate performance report
        performance_report = format_performance_report(analyzers, strategy)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save performance report
        report_file = output_dir / f"performance_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(performance_report, f, indent=2, default=str)
        
        # Save trade blotter
        if config['output'].get('save_trades', True):
            save_trade_blotter(strategy, output_dir)
        
        # Save equity curve
        if config['output'].get('save_equity_curve', True):
            save_equity_curve(strategy, output_dir)
        
        # Save plots
        if config['output'].get('save_plots', True):
            save_drawdown_plot(analyzers, output_dir)
        
        # Print summary
        print(f"\n🏆 BACKTEST RESULTS SUMMARY")
        print(f"=" * 50)
        
        summary = performance_report['summary']
        print(f"Total Return:           {summary.get('total_return_pct', 0):+.2f}%")
        print(f"CAGR:                   {summary.get('cagr_pct', 0):+.2f}%")
        print(f"Max Drawdown:           {summary.get('max_drawdown_pct', 0):.2f}%")
        print(f"Sharpe Ratio:           {performance_report['risk_metrics'].get('sharpe_ratio', 0):.2f}")
        print(f"Calmar Ratio:           {summary.get('calmar_ratio', 0):.2f}")
        print(f"Total Trades:           {performance_report['trade_metrics'].get('total_trades', 0)}")
        print(f"Win Rate:               {performance_report['trade_metrics'].get('win_rate_pct', 0):.1f}%")
        
        if 'ml_metrics' in performance_report:
            ml_metrics = performance_report['ml_metrics']
            print(f"Avg ML Regime Accuracy: {ml_metrics.get('avg_regime_accuracy', 0):.3f}")
            print(f"Avg ML Strength R²:     {ml_metrics.get('avg_strength_r2', 0):.3f}")
        
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"📊 Performance report: {report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
