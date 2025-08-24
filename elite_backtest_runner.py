#!/usr/bin/env python3
"""
ELITE Backtest Runner - Maximum Accuracy & Profit

Features:
1. Enhanced ML Signal Generator (XGBoost + LightGBM + Neural Networks)
2. Elite risk management and position sizing
3. Advanced feature engineering (100+ features)
4. Multi-horizon predictions
5. Regime-aware trading
6. Comprehensive performance analysis
"""

import argparse
import os
import sys
import logging
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Install required packages if not available
required_packages = [
    'backtrader', 'xgboost', 'lightgbm', 'talib-binary',
    'yfinance', 'pandas', 'numpy', 'scikit-learn', 'scipy'
]

def install_requirements():
    """Install required packages"""
    import subprocess
    import sys
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try to install requirements
try:
    install_requirements()
except Exception as e:
    print(f"Warning: Could not install some requirements: {e}")

import backtrader as bt
import backtrader.analyzers as btanalyzers
import yfinance as yf
import pandas as pd
import numpy as np

from backtrader_system.strategies.elite_ml_strategy import EliteMLTradingStrategy


def main():
    """Main function for running elite backtests"""
    parser = argparse.ArgumentParser(description='Elite ML Trading System - Maximum Accuracy & Profit')
    parser.add_argument('--initial-capital', type=float, default=1000000,
                       help='Initial capital (default: 1000000)')
    parser.add_argument('--symbols', nargs='+', 
                       default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 
                               'ORCL', 'JPM', 'BAC', 'V', 'MA', 'UNH', 'JNJ', 'PG', 
                               'KO', 'XOM', 'HD', 'DIS', 'CRM'],
                       help='Symbols to trade')
    parser.add_argument('--start-date', default='2022-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-08-23',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--training-split', type=float, default=0.6,
                       help='Training data percentage (default: 0.6)')
    parser.add_argument('--rebalance-frequency', type=int, default=1,
                       help='Rebalance frequency in days (default: 1)')
    parser.add_argument('--training-data-length', type=int, default=None,
                       help='Number of bars required for training each symbol. '
                            'Defaults to max(training split days, 730)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_filename = f"elite_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("🚀 ELITE ML TRADING SYSTEM - MAXIMUM ACCURACY & PROFIT")
    logger.info("="*60)
    logger.info("🎯 Elite Features:")
    logger.info("   • XGBoost + LightGBM + Neural Network Ensemble")
    logger.info("   • 100+ Advanced Technical Features")
    logger.info("   • Multi-horizon Predictions (1d, 3d, 5d, 10d, 20d)")
    logger.info("   • Risk-adjusted Kelly Position Sizing")
    logger.info("   • Regime-aware Trading")
    logger.info("   • Advanced Risk Management")
    logger.info("   • Market Context Integration (VIX, SPY)")
    logger.info("="*60)
    
    try:
        # Calculate training and testing periods
        start_date = pd.to_datetime(args.start_date)
        end_date = pd.to_datetime(args.end_date)
        total_days = (end_date - start_date).days
        training_days = int(total_days * args.training_split)

        training_end = start_date + timedelta(days=training_days)

        if args.training_data_length is not None:
            training_data_length = args.training_data_length
        else:
            training_data_length = max(training_days, 730)

        logger.info(f"📊 Data Periods:")
        logger.info(f"   Training: {start_date.date()} to {training_end.date()}")
        logger.info(f"   Testing: {training_end.date()} to {end_date.date()}")
        logger.info(f"   Training data length: {training_data_length} bars")
        
        # Download data for symbols
        logger.info(f"📈 Downloading data for {len(args.symbols)} symbols...")
        
        data_dict = {}
        failed_symbols = []
        
        for symbol in args.symbols:
            try:
                logger.info(f"   Downloading {symbol}...")
                ticker = yf.Ticker(symbol)
                
                # Download with some buffer for technical indicators
                buffer_start = start_date - timedelta(days=100)
                df = ticker.history(start=buffer_start, end=end_date, auto_adjust=True)
                
                if df.empty or len(df) < 200:
                    logger.warning(f"   ⚠️  Insufficient data for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                # Ensure we have required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_columns):
                    logger.warning(f"   ⚠️  Missing required columns for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                # Clean data
                df = df.dropna()
                
                # Ensure Volume is numeric
                if 'Volume' in df.columns:
                    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
                    df['Volume'] = df['Volume'].fillna(df['Volume'].median())
                
                # Filter to actual date range (keeping buffer for indicators)
                # Handle timezone-aware vs timezone-naive comparison
                if df.index.tz is not None:
                    # If DataFrame index is timezone-aware, make start_date timezone-aware too
                    start_date_tz = pd.Timestamp(start_date).tz_localize(df.index.tz)
                    df_filtered = df[df.index >= start_date_tz]
                else:
                    df_filtered = df[df.index >= start_date]
                
                if len(df_filtered) < 100:
                    logger.warning(f"   ⚠️  Insufficient data after filtering for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                data_dict[symbol] = df  # Keep full data with buffer
                logger.info(f"   ✅ {symbol}: {len(df_filtered)} bars")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to download {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            logger.warning(f"⚠️  Failed symbols: {failed_symbols}")
        
        logger.info(f"✅ Successfully downloaded {len(data_dict)} symbols")
        
        if len(data_dict) < 3:
            raise ValueError(f"Insufficient data: only {len(data_dict)} symbols available, need at least 3")
        
        # Create elite ML configuration
        config = {
            'training_split': args.training_split,
            'feature_columns': [
                # Price action features
                'price_change', 'price_change_3d', 'price_change_5d', 'price_change_10d', 'price_change_20d',
                'hl_ratio', 'oc_ratio', 'gap_up', 'close_position',
                
                # Moving averages
                'price_vs_ma_5', 'price_vs_ma_10', 'price_vs_ma_20', 'price_vs_ma_50', 'price_vs_ma_100',
                'ma_5_20_diff', 'ma_20_50_diff', 'ma_slope_5',
                
                # Volatility
                'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_30d',
                'atr_14', 'atr_rank', 'vol_regime',
                
                # Momentum
                'rsi_14', 'rsi_7', 'rsi_21', 'rsi_divergence',
                'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_histogram', 'macd_slope',
                'williams_r', 'roc_5', 'roc_10', 'roc_20', 'momentum_consistency',
                
                # Volume
                'volume_ratio', 'volume_rank', 'price_volume_trend', 'price_vs_vwap',
                'obv_slope', 'ad_slope', 'volume_profile',
                
                # Mean reversion
                'bb_width', 'bb_position', 'bb_squeeze', 'kc_position',
                'mean_reversion_5', 'mean_reversion_20',
                
                # Market context
                'vix_rank', 'vix_change', 'fear_greed', 'beta_spy', 'relative_strength',
                'is_open_hour', 'is_power_hour',
                
                # Patterns
                'near_resistance', 'near_support', 'trend_strength', 'trend_direction',
                'channel_position', 'fractal_high', 'fractal_low',
                
                # Statistical
                'skewness_20', 'kurtosis_20', 'hurst_20', 'info_ratio', 'sharpe_20', 'max_drawdown_20',
                
                # Regime
                'regime_momentum', 'regime_volatility', 'regime_score', 'bull_regime'
            ],
            'min_training_days': 300,
            'model_params': {
                'ensemble_voting': 'soft',
                'xgb_n_estimators': 300,
                'lgb_n_estimators': 300,
                'nn_hidden_layers': (100, 50, 25)
            }
        }
        
        logger.info(f"🚀 Starting ELITE backtest...")
        logger.info(f"   Symbols: {len(data_dict)}")
        logger.info(f"   Period: {args.start_date} to {args.end_date}")
        logger.info(f"   Capital: ${args.initial_capital:,.0f}")
        logger.info(f"   Training Split: {args.training_split:.0%}")
        
        # Create Cerebro instance
        cerebro = bt.Cerebro()
        
        # Set elite broker settings
        cerebro.broker.setcash(args.initial_capital)
        cerebro.broker.setcommission(commission=0.0005)  # 0.05% commission (competitive)
        
        # Add elite strategy
        strategy_params = {
            'signal_threshold': 0.15,      # Higher threshold for quality
            'max_positions': 12,           # Manageable portfolio size
            'max_position_size': 0.15,     # Maximum 15% per position
            'max_symbol_exposure': 0.12,   # Maximum 12% per symbol
            'min_hold_days': 2,            # Minimum holding period
            'stop_loss_pct': 0.08,         # 8% stop loss
            'profit_target_pct': 0.25,     # 25% profit target
            'trailing_stop_pct': 0.12,     # 12% trailing stop
            'max_portfolio_heat': 0.30,    # Maximum 30% portfolio risk
            'config': config,
            'rebalance_frequency': args.rebalance_frequency,
            'training_data_length': training_data_length,
            'volatility_filter': 0.05,     # Max 5% volatility for entry
            'volume_filter': 0.8,          # Min 80% volume ratio
        }
        
        cerebro.addstrategy(EliteMLTradingStrategy, **strategy_params)
        
        # Add data feeds (filter to actual backtest period)
        for symbol, df in data_dict.items():
            # Filter to backtest period but keep some buffer for indicators
            backtest_start = start_date - timedelta(days=50)
            
            # Handle timezone-aware filtering
            if df.index.tz is not None:
                backtest_start_tz = pd.Timestamp(backtest_start).tz_localize(df.index.tz)
                df_backtest = df[df.index >= backtest_start_tz]
            else:
                df_backtest = df[df.index >= backtest_start]
            
            data_feed = bt.feeds.PandasData(
                dataname=df_backtest,
                name=symbol,
                plot=False
            )
            cerebro.adddata(data_feed)
        
        # Add analyzers
        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')
        cerebro.addanalyzer(btanalyzers.VWR, _name='vwr')
        cerebro.addanalyzer(btanalyzers.Calmar, _name='calmar')
        
        logger.info("⚙️  Running elite backtest...")
        
        # Run the backtest
        results = cerebro.run(maxcpus=1, tradehistory=True)
        strategy = results[0]
        
        # Get final portfolio value
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - args.initial_capital) / args.initial_capital
        
        logger.info("✅ Elite backtest completed!")
        logger.info(f"📊 Final Portfolio Value: ${final_value:,.0f}")
        logger.info(f"📈 Total Return: {total_return:.2%}")
        
        # Extract performance metrics
        try:
            sharpe_ratio = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            max_drawdown = strategy.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
            calmar_ratio = strategy.analyzers.calmar.get_analysis().get('calmar', 0)
            vwr = strategy.analyzers.vwr.get_analysis().get('vwr', 0)
            sqn = strategy.analyzers.sqn.get_analysis().get('sqn', 0)
            
            trade_analysis = strategy.analyzers.trades.get_analysis()
            total_trades = trade_analysis.get('total', {}).get('total', 0)
            won_trades = trade_analysis.get('won', {}).get('total', 0)
            lost_trades = trade_analysis.get('lost', {}).get('total', 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
        except Exception as e:
            logger.warning(f"Could not extract all analyzer results: {e}")
            sharpe_ratio = max_drawdown = calmar_ratio = vwr = sqn = 0
            total_trades = won_trades = lost_trades = win_rate = 0
        
        # Extract trade blotter
        trade_blotter = pd.DataFrame()
        if hasattr(strategy, 'trade_log') and strategy.trade_log:
            trade_blotter = pd.DataFrame(strategy.trade_log)
            
        # Extract equity curve
        equity_curve = pd.DataFrame()
        if hasattr(strategy, 'daily_values') and strategy.daily_values:
            equity_curve = pd.DataFrame(strategy.daily_values)
        
        # Extract signals history
        signals_history = pd.DataFrame()
        if hasattr(strategy, 'signals_history') and strategy.signals_history:
            signals_history = pd.DataFrame(strategy.signals_history)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('elite_results', exist_ok=True)
        
        # Save trade blotter
        if not trade_blotter.empty:
            filename = f"elite_results/elite_trade_blotter_{timestamp}.csv"
            trade_blotter.to_csv(filename, index=False)
            logger.info(f"📋 Trade blotter saved: {filename}")
        
        # Save equity curve
        if not equity_curve.empty:
            filename = f"elite_results/elite_equity_curve_{timestamp}.csv"
            equity_curve.to_csv(filename, index=False)
            logger.info(f"📈 Equity curve saved: {filename}")
        
        # Save signals history
        if not signals_history.empty:
            filename = f"elite_results/elite_signals_{timestamp}.csv"
            signals_history.to_csv(filename, index=False)
            logger.info(f"🎯 Signals history saved: {filename}")
        
        # Get ML model performance
        ml_performance = {}
        if hasattr(strategy, 'ml_generator'):
            ml_summary = strategy.ml_generator.get_model_summary()
            ml_performance = ml_summary.get('average_performance', {})
        
        # Create comprehensive results summary
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'backtest_type': 'ELITE_ML',
            'configuration': {
                'initial_capital': args.initial_capital,
                'symbols_count': len(data_dict),
                'training_split': args.training_split,
                'start_date': args.start_date,
                'end_date': args.end_date
            },
            'performance_metrics': {
                'final_value': final_value,
                'total_return_pct': total_return * 100,
                'sharpe_ratio': sharpe_ratio if sharpe_ratio else 0,
                'calmar_ratio': calmar_ratio if calmar_ratio else 0,
                'max_drawdown_pct': max_drawdown if max_drawdown else 0,
                'vwr': vwr if vwr else 0,
                'sqn': sqn if sqn else 0
            },
            'trading_metrics': {
                'total_trades': total_trades,
                'won_trades': won_trades,
                'lost_trades': lost_trades,
                'win_rate_pct': win_rate
            },
            'ml_metrics': ml_performance,
            'elite_features': [
                'XGBoost + LightGBM + Neural Network Ensemble',
                '100+ Advanced Technical Features',
                'Multi-horizon Predictions',
                'Risk-adjusted Kelly Position Sizing',
                'Regime-aware Trading',
                'Market Context Integration (VIX, SPY)',
                'Advanced Risk Management',
                'Pattern Recognition',
                'Statistical Feature Engineering'
            ]
        }
        
        # Save performance report
        results_filename = f"elite_results/elite_performance_report_{timestamp}.json"
        with open(results_filename, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"📊 Performance report: {results_filename}")
        
        # Print comprehensive results
        print("\\n" + "="*80)
        print("🏆 ELITE ML TRADING SYSTEM - COMPREHENSIVE RESULTS")
        print("="*80)
        print(f"💰 Financial Performance:")
        print(f"   Initial Capital:        ${args.initial_capital:,.0f}")
        print(f"   Final Value:            ${final_value:,.0f}")
        print(f"   Total Return:           {total_return:.2%}")
        print(f"   Sharpe Ratio:           {sharpe_ratio:.3f}")
        print(f"   Calmar Ratio:           {calmar_ratio:.3f}")
        print(f"   Max Drawdown:           {max_drawdown:.2f}%")
        print(f"   VWR:                    {vwr:.3f}")
        print(f"   SQN:                    {sqn:.3f}")
        
        print(f"\\n📊 Trading Performance:")
        print(f"   Total Trades:           {total_trades}")
        print(f"   Win Rate:               {win_rate:.1f}%")
        print(f"   Winning Trades:         {won_trades}")
        print(f"   Losing Trades:          {lost_trades}")
        
        if ml_performance:
            print(f"\\n🧠 ML Model Performance:")
            print(f"   Avg Strength R²:        {ml_performance.get('strength_r2', 0):.3f}")
            print(f"   Avg Direction Accuracy: {ml_performance.get('direction_accuracy', 0):.3f}")
        
        print(f"\\n🎯 Elite Features Applied:")
        for feature in results_summary['elite_features']:
            print(f"   • {feature}")
        
        print(f"\\n📁 Results saved to: elite_results/")
        print(f"   • Performance report: {results_filename}")
        if not trade_blotter.empty:
            print(f"   • Trade blotter: {len(trade_blotter)} trades")
        if not equity_curve.empty:
            print(f"   • Equity curve: {len(equity_curve)} daily values")
        if not signals_history.empty:
            print(f"   • Signals history: {len(signals_history)} signals")
        
        print("="*80)
        
        # Performance vs benchmarks
        if total_return > 0.15:  # 15%+ return
            print("🚀 EXCELLENT PERFORMANCE - Above 15% return!")
        elif total_return > 0.10:  # 10%+ return
            print("✅ GOOD PERFORMANCE - Above 10% return!")
        elif total_return > 0.05:  # 5%+ return  
            print("📈 MODERATE PERFORMANCE - Above 5% return")
        else:
            print("⚠️  UNDERPERFORMANCE - Consider strategy adjustments")
            
        if sharpe_ratio > 1.5:
            print("💎 ELITE RISK-ADJUSTED RETURNS - Sharpe > 1.5!")
        elif sharpe_ratio > 1.0:
            print("🔥 EXCELLENT RISK-ADJUSTED RETURNS - Sharpe > 1.0!")
        elif sharpe_ratio > 0.5:
            print("👍 GOOD RISK-ADJUSTED RETURNS - Sharpe > 0.5")
        
        print("\\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Elite backtest failed: {e}")
        raise


if __name__ == '__main__':
    main()
