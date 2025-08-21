#!/usr/bin/env python3
"""
Unified Live Trading Runner
Consolidates all sophisticated components for live execution

Features:
- Elite stock selection with AI screening
- Ensemble ML models (RandomForest, GradientBoosting, LightGBM)
- Technical indicator signals (RSI, MA, momentum, volatility)
- Options trading recommendations (50-200% return strategies)
- Real-time Alpaca paper trading execution
- Comprehensive risk management
- Daily performance monitoring

Target: 30%+ Annual Returns with institutional-grade risk management

Author: Unified ML Trading System
"""

import os
import sys
import argparse
import json
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Import unified trading system
from unified_ml_trading_system import UnifiedMLTradingSystem
from comprehensive_backtester import ComprehensiveBacktester

class LiveTradingRunner:
    """
    Live trading execution engine that consolidates all sophisticated components
    """
    
    def __init__(self, config_path: str = "unified_trading_config.json", 
                 account_size: float = 100000):
        """Initialize live trading runner"""
        
        self.config_path = config_path
        self.account_size = account_size
        
        # Initialize unified trading system
        self.trading_system = UnifiedMLTradingSystem(config_path)
        
        # Trading state
        self.is_running = False
        self.daily_stats = {}
        self.session_results = {}
        
        print("🚀 Live Trading Runner Initialized")
        print(f"💰 Account Size: ${account_size:,.2f}")
        print(f"🎯 Target: {self.trading_system.config['performance_targets']['annual_return_target']}% Annual Return")
        
    def pre_market_preparation(self):
        """Pre-market preparation and analysis"""
        print("\\n🌅 PRE-MARKET PREPARATION")
        print("=" * 50)
        
        try:
            # Update stock universe
            print("🔍 Updating elite stock universe...")
            universe = self.trading_system.get_elite_stock_universe()
            print(f"✅ Selected {len(universe)} elite stocks")
            
            # Train/update ML models for new or underperforming stocks
            print("🤖 Updating ML models...")
            models_updated = 0
            
            for symbol in universe[:20]:  # Limit for pre-market time
                try:
                    # Check if model needs update
                    if symbol not in self.trading_system.ensemble_models:
                        print(f"   Training new model for {symbol}")
                        
                        # Get training data
                        import yfinance as yf
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(period='1y')
                        
                        if len(data) > 100:
                            success = self.trading_system.train_ensemble_models(symbol, data)
                            if success:
                                models_updated += 1
                                
                except Exception as e:
                    print(f"   ⚠️ Error updating model for {symbol}: {e}")
                    continue
            
            print(f"✅ Updated {models_updated} ML models")
            
            # Generate pre-market watchlist
            print("📋 Generating pre-market watchlist...")
            watchlist = self.generate_premarket_watchlist(universe)
            
            self.daily_stats = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'universe_size': len(universe),
                'models_updated': models_updated,
                'watchlist_size': len(watchlist),
                'watchlist': watchlist
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Pre-market preparation failed: {e}")
            return False
    
    def generate_premarket_watchlist(self, universe: List[str]) -> List[Dict]:
        """Generate pre-market watchlist of high-potential stocks"""
        
        watchlist = []
        
        # Get pre-market data for universe stocks
        import yfinance as yf
        
        for symbol in universe[:50]:  # Limit for pre-market processing
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='5d')
                
                if len(data) < 3:
                    continue
                
                current_price = float(data['Close'].iloc[-1])
                
                # Generate signals
                ml_prediction = self.trading_system.get_ml_prediction(symbol, data)
                tech_signal = self.trading_system.calculate_technical_signals(symbol, data)
                options_rec = self.trading_system.get_options_recommendations(
                    symbol, current_price, ml_prediction, tech_signal
                )
                
                composite_signal = self.trading_system.calculate_composite_signal(
                    symbol, ml_prediction, tech_signal, options_rec
                )
                
                # Add to watchlist if signal is strong
                if (composite_signal['signal'] in ['BUY', 'SELL'] and 
                    composite_signal['strength'] > 0.6 and
                    composite_signal['conviction'] > 0.65):
                    
                    watchlist.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'signal': composite_signal['signal'],
                        'strength': composite_signal['strength'],
                        'conviction': composite_signal['conviction'],
                        'ml_prediction': ml_prediction,
                        'rsi': tech_signal.get('rsi', 0),
                        'momentum': tech_signal.get('momentum', 0),
                        'options_strategy': options_rec.get('strategy', 'none')
                    })
                
            except Exception as e:
                continue
        
        # Sort by signal strength
        watchlist.sort(key=lambda x: x['strength'] * x['conviction'], reverse=True)
        
        print(f"📋 Generated watchlist with {len(watchlist)} high-potential stocks")
        
        return watchlist[:15]  # Top 15 for focused trading
    
    def market_open_execution(self):
        """Execute trades at market open"""
        print("\\n🔔 MARKET OPEN EXECUTION")
        print("=" * 50)
        
        try:
            # Check if market is open
            from datetime import datetime, time
            from zoneinfo import ZoneInfo
            
            now_et = datetime.now(tz=ZoneInfo('US/Eastern'))
            market_open = time(9, 30)
            market_close = time(16, 0)
            is_weekday = now_et.weekday() < 5
            is_market_hours = market_open <= now_et.time() <= market_close
            
            if not (is_weekday and is_market_hours):
                print(f"⏰ Market is closed. Current time: {now_et.strftime('%H:%M %Z')}")
                return False
            
            # Execute comprehensive trading session
            session_results = self.trading_system.execute_trading_session()
            
            self.session_results = session_results
            self.daily_stats.update({
                'market_open_time': now_et.strftime('%H:%M:%S %Z'),
                'signals_generated': session_results.get('signals_generated', 0),
                'trades_executed': session_results.get('trades_executed', 0),
                'errors': session_results.get('errors', 0)
            })
            
            print(f"✅ Market open execution completed")
            print(f"   Signals: {session_results.get('signals_generated', 0)}")
            print(f"   Trades: {session_results.get('trades_executed', 0)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Market open execution failed: {e}")
            return False
    
    def intraday_monitoring(self):
        """Continuous intraday monitoring and execution"""
        print("\\n📊 INTRADAY MONITORING")
        print("=" * 30)
        
        try:
            # Get current positions
            if self.trading_system.alpaca_connected:
                positions = self.trading_system.trading_client.get_all_positions()
                account = self.trading_system.trading_client.get_account()
                
                print(f"💼 Current Portfolio:")
                print(f"   Total Value: ${float(account.portfolio_value):,.2f}")
                print(f"   Day P&L: ${float(account.unrealized_pl):+,.2f}")
                print(f"   Positions: {len(positions)}")
                
                # Monitor positions for exit signals
                exit_trades = 0
                
                for position in positions:
                    symbol = position.symbol
                    current_shares = int(position.qty)
                    
                    if current_shares > 0:
                        # Check for exit signals
                        exit_signal = self.check_exit_signal(symbol, current_shares)
                        
                        if exit_signal:
                            # Execute exit trade
                            success = self.execute_exit_trade(symbol, current_shares, exit_signal)
                            if success:
                                exit_trades += 1
                
                print(f"   Exit trades executed: {exit_trades}")
                
                # Update daily stats
                self.daily_stats.update({
                    'current_portfolio_value': float(account.portfolio_value),
                    'day_pnl': float(account.unrealized_pl),
                    'positions_count': len(positions),
                    'exit_trades': exit_trades
                })
            
            return True
            
        except Exception as e:
            print(f"⚠️ Intraday monitoring error: {e}")
            return False
    
    def check_exit_signal(self, symbol: str, shares: int) -> Optional[Dict]:
        """Check if position should be exited"""
        try:
            import yfinance as yf
            
            # Get recent data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='5d')
            
            if len(data) < 3:
                return None
            
            current_price = float(data['Close'].iloc[-1])
            
            # Generate current signals
            ml_prediction = self.trading_system.get_ml_prediction(symbol, data)
            tech_signal = self.trading_system.calculate_technical_signals(symbol, data)
            
            composite_signal = self.trading_system.calculate_composite_signal(
                symbol, ml_prediction, tech_signal, {}
            )
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            # Signal-based exit
            if (composite_signal['signal'] == 'SELL' and 
                composite_signal['strength'] > 0.7):
                should_exit = True
                exit_reason = "Strong sell signal"
            
            # Risk-based exit (simplified)
            if tech_signal.get('rsi', 50) > 80:  # Very overbought
                should_exit = True
                exit_reason = "Overbought (RSI > 80)"
            
            # Momentum reversal
            if tech_signal.get('momentum', 0) < -0.05:  # Strong negative momentum
                should_exit = True
                exit_reason = "Momentum reversal"
            
            if should_exit:
                return {
                    'reason': exit_reason,
                    'signal_strength': composite_signal['strength'],
                    'current_price': current_price
                }
            
            return None
            
        except Exception as e:
            print(f"   Error checking exit signal for {symbol}: {e}")
            return None
    
    def execute_exit_trade(self, symbol: str, shares: int, exit_signal: Dict) -> bool:
        """Execute exit trade"""
        try:
            current_price = exit_signal['current_price']
            reason = exit_signal['reason']
            
            success = self.trading_system.execute_trade(symbol, 'SELL', shares, current_price)
            
            if success:
                print(f"   🔄 SOLD {shares} shares of {symbol} at ${current_price:.2f}")
                print(f"      Reason: {reason}")
                return True
            
            return False
            
        except Exception as e:
            print(f"   ❌ Error executing exit trade for {symbol}: {e}")
            return False
    
    def end_of_day_summary(self):
        """Generate end-of-day summary and reports"""
        print("\\n🌇 END-OF-DAY SUMMARY")
        print("=" * 50)
        
        try:
            # Generate performance report
            performance_report = self.trading_system.generate_performance_report()
            
            # Update daily stats
            if performance_report.get('portfolio_summary'):
                portfolio = performance_report['portfolio_summary']
                self.daily_stats.update({
                    'eod_portfolio_value': portfolio.get('total_value', 0),
                    'eod_cash': portfolio.get('cash', 0),
                    'eod_pnl': portfolio.get('day_pnl', 0),
                    'eod_positions': portfolio.get('positions_count', 0)
                })
            
            # Calculate daily performance
            initial_value = self.account_size
            current_value = self.daily_stats.get('eod_portfolio_value', initial_value)
            daily_return = (current_value - initial_value) / initial_value
            
            # Save daily results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            daily_results = {
                'timestamp': timestamp,
                'daily_stats': self.daily_stats,
                'session_results': self.session_results,
                'performance_report': performance_report,
                'daily_return': daily_return,
                'account_value': current_value
            }
            
            # Save to file
            with open(f'daily_trading_results_{timestamp}.json', 'w') as f:
                json.dump(daily_results, f, indent=2, default=str)
            
            # Print summary
            print(f"📊 Daily Trading Summary:")
            print(f"   Date: {self.daily_stats.get('date')}")
            print(f"   Account Value: ${current_value:,.2f}")
            print(f"   Daily Return: {daily_return*100:+.2f}%")
            print(f"   Signals Generated: {self.daily_stats.get('signals_generated', 0)}")
            print(f"   Trades Executed: {self.daily_stats.get('trades_executed', 0)}")
            print(f"   Watchlist Size: {self.daily_stats.get('watchlist_size', 0)}")
            print(f"   Models Updated: {self.daily_stats.get('models_updated', 0)}")
            print(f"\\n📁 Results saved: daily_trading_results_{timestamp}.json")
            
            return daily_results
            
        except Exception as e:
            print(f"❌ End-of-day summary failed: {e}")
            return {}
    
    def run_live_trading_day(self):
        """Execute complete live trading day"""
        print("🌟 STARTING LIVE TRADING DAY")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Pre-market preparation
        prep_success = self.pre_market_preparation()
        if not prep_success:
            print("❌ Pre-market preparation failed. Aborting trading day.")
            return False
        
        # Market open execution
        open_success = self.market_open_execution()
        if not open_success:
            print("⚠️ Market open execution had issues, but continuing...")
        
        # Set up intraday monitoring (would run throughout the day)
        print("\\n⏰ Setting up intraday monitoring...")
        
        # For demonstration, run a few monitoring cycles
        for i in range(3):
            print(f"\\n📊 Monitoring cycle {i+1}/3")
            self.intraday_monitoring()
            time.sleep(5)  # In real trading, this would be longer intervals
        
        # End-of-day summary
        daily_results = self.end_of_day_summary()
        
        print("\\n🎉 LIVE TRADING DAY COMPLETED!")
        
        return daily_results
    
    def run_backtest_validation(self):
        """Run backtest validation before live trading"""
        print("\\n🔬 RUNNING BACKTEST VALIDATION")
        print("=" * 50)
        
        try:
            # Initialize backtester
            backtester = ComprehensiveBacktester(self.config_path)
            
            # Run shorter backtest for validation
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year
            
            # Get limited universe for faster validation
            universe = self.trading_system.get_elite_stock_universe()[:20]  # Top 20 stocks
            
            print(f"🎯 Validation backtest: {len(universe)} stocks from {start_date} to {end_date}")
            
            # Run backtest
            results = backtester.run_walk_forward_backtest(universe, start_date, end_date)
            
            # Check if performance meets targets
            annual_return = results['performance_metrics'].get('annual_return', 0)
            sharpe_ratio = results['performance_metrics'].get('sharpe_ratio', 0)
            max_drawdown = results['performance_metrics'].get('max_drawdown', 0)
            
            target_return = self.trading_system.config['performance_targets']['annual_return_target'] / 100
            min_sharpe = self.trading_system.config['performance_targets']['min_sharpe_ratio']
            max_dd = self.trading_system.config['performance_targets']['max_drawdown_target']
            
            validation_passed = (
                annual_return >= target_return * 0.8 and  # 80% of target
                sharpe_ratio >= min_sharpe * 0.8 and
                abs(max_drawdown) <= max_dd * 1.2
            )
            
            print(f"\\n📊 Validation Results:")
            print(f"   Annual Return: {annual_return*100:.2f}% (Target: {target_return*100:.0f}%)")
            print(f"   Sharpe Ratio: {sharpe_ratio:.2f} (Target: {min_sharpe:.1f})")
            print(f"   Max Drawdown: {max_drawdown*100:.2f}% (Max: {max_dd*100:.0f}%)")
            print(f"   ✅ Validation {'PASSED' if validation_passed else 'FAILED'}")
            
            return validation_passed, results
            
        except Exception as e:
            print(f"❌ Backtest validation failed: {e}")
            return False, {}


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Unified Live Trading System")
    parser.add_argument('--account', type=float, default=100000.0, help='Account size')
    parser.add_argument('--mode', choices=['live', 'validation', 'single'], default='single',
                       help='Execution mode: live (scheduled), validation (backtest), single (one session)')
    parser.add_argument('--config', type=str, default='unified_trading_config.json',
                       help='Configuration file path')
    parser.add_argument('--validate-first', action='store_true',
                       help='Run backtest validation before live trading')
    
    args = parser.parse_args()
    
    print("🚀 UNIFIED ML TRADING SYSTEM")
    print("Consolidating Elite Components for 30%+ Annual Returns")
    print("=" * 80)
    print(f"💰 Account Size: ${args.account:,.2f}")
    print(f"🎛️ Mode: {args.mode}")
    print(f"⚙️ Config: {args.config}")
    
    # Initialize live trading runner
    runner = LiveTradingRunner(args.config, args.account)
    
    if args.mode == 'validation':
        # Run backtest validation only
        print("\\n🔬 Running backtest validation...")
        validation_passed, results = runner.run_backtest_validation()
        
        if validation_passed:
            print("\\n🎉 System validation PASSED! Ready for live trading.")
        else:
            print("\\n⚠️ System validation FAILED. Review configuration and models.")
        
    elif args.mode == 'single':
        # Run single trading session
        if args.validate_first:
            print("\\n🔬 Running validation first...")
            validation_passed, _ = runner.run_backtest_validation()
            
            if not validation_passed:
                print("\\n❌ Validation failed. Skipping live trading.")
                return
        
        print("\\n🎯 Running single live trading session...")
        daily_results = runner.run_live_trading_day()
        
        if daily_results:
            print("\\n✅ Single trading session completed successfully!")
        else:
            print("\\n❌ Trading session encountered issues.")
    
    elif args.mode == 'live':
        # Schedule live trading
        print("\\n⏰ Scheduling live trading sessions...")
        
        # Schedule pre-market preparation
        schedule.every().monday.at("08:00").do(runner.pre_market_preparation)
        schedule.every().tuesday.at("08:00").do(runner.pre_market_preparation)
        schedule.every().wednesday.at("08:00").do(runner.pre_market_preparation)
        schedule.every().thursday.at("08:00").do(runner.pre_market_preparation)
        schedule.every().friday.at("08:00").do(runner.pre_market_preparation)
        
        # Schedule market open execution
        schedule.every().monday.at("09:30").do(runner.market_open_execution)
        schedule.every().tuesday.at("09:30").do(runner.market_open_execution)
        schedule.every().wednesday.at("09:30").do(runner.market_open_execution)
        schedule.every().thursday.at("09:30").do(runner.market_open_execution)
        schedule.every().friday.at("09:30").do(runner.market_open_execution)
        
        # Schedule intraday monitoring
        schedule.every().monday.at("12:00").do(runner.intraday_monitoring)
        schedule.every().tuesday.at("12:00").do(runner.intraday_monitoring)
        schedule.every().wednesday.at("12:00").do(runner.intraday_monitoring)
        schedule.every().thursday.at("12:00").do(runner.intraday_monitoring)
        schedule.every().friday.at("12:00").do(runner.intraday_monitoring)
        
        schedule.every().monday.at("14:30").do(runner.intraday_monitoring)
        schedule.every().tuesday.at("14:30").do(runner.intraday_monitoring)
        schedule.every().wednesday.at("14:30").do(runner.intraday_monitoring)
        schedule.every().thursday.at("14:30").do(runner.intraday_monitoring)
        schedule.every().friday.at("14:30").do(runner.intraday_monitoring)
        
        # Schedule end-of-day summary
        schedule.every().monday.at("16:30").do(runner.end_of_day_summary)
        schedule.every().tuesday.at("16:30").do(runner.end_of_day_summary)
        schedule.every().wednesday.at("16:30").do(runner.end_of_day_summary)
        schedule.every().thursday.at("16:30").do(runner.end_of_day_summary)
        schedule.every().friday.at("16:30").do(runner.end_of_day_summary)
        
        print("✅ Trading schedule configured")
        print("🔄 Running continuous live trading...")
        
        runner.is_running = True
        
        while runner.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    print("\\n🎯 Unified ML Trading System execution completed!")


if __name__ == "__main__":
    main()
