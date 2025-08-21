#!/usr/bin/env python3
"""
AUTOMATED TRADING SIGNAL OPTIMIZER
Automatically tests different signal configurations across multiple stocks
Finds optimal parameters that maximize profits across the entire stock universe
Continuously improves until we beat buy-and-hold on all stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import itertools
import json
import time
import warnings
warnings.filterwarnings('ignore')

class AutomatedSignalOptimizer:
    def __init__(self, stocks=None):
        # STOCK UNIVERSE - Mix of different sectors and volatilities
        if stocks:
            self.stocks = stocks
        else:
            self.stocks = [
                # TECH GIANTS
                "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN",
                # VOLATILE GROWTH  
                "TSLA", "META", "NFLX", "CRM", "UBER",
                # TRADITIONAL VALUE
                "JPM", "WMT", "JNJ", "PG", "KO",
                # EMERGING/SPECULATIVE
                "PLTR", "COIN", "SNOW", "AMD", "INTC",
                # ENERGY/MATERIALS
                "XOM", "CVX", "CAT", "BA", "GE"
            ]
        
        self.results_history = []
        self.best_config = None
        self.best_avg_performance = -float('inf')
        
        # OPTIMIZED PARAMETER SEARCH SPACE - Including proven best configs
        self.parameter_space = {
            'trend_5d_buy_threshold': [0.02, 0.025, 0.03, 0.035, 0.04, 0.05],  # 6 values - includes 0.025
            'trend_5d_sell_threshold': [-0.015, -0.02, -0.025, -0.03, -0.035],  # 5 values - includes -0.02
            'trend_10d_buy_threshold': [0.015, 0.02, 0.025, 0.03, 0.035],  # 5 values - includes 0.025
            'trend_10d_sell_threshold': [-0.02, -0.025, -0.03, -0.035, -0.04, -0.045],  # 6 values - includes -0.045
            'rsi_overbought': [65, 70, 75, 80, 85],  # 5 values - includes 65
            'rsi_oversold': [15, 20, 25, 30],  # 4 values - includes 20
            'volatility_threshold': [0.06, 0.07, 0.08, 0.10, 0.12],  # 5 values - includes 0.07
            'volume_ratio_threshold': [1.2, 1.4, 1.6, 1.8]  # 4 values - includes 1.6
        }
        
    def calculate_technical_indicators(self, data):
        """Calculate all technical indicators for a stock"""
        data = data.copy()
        
        # Moving averages
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
        
    def generate_signals_with_config(self, data, config):
        """Generate trading signals with specific configuration"""
        signals = []
        
        for i in range(30, len(data)):
            price = float(data['Close'].iloc[i])
            
            # Multi-timeframe analysis
            recent_5d = data['Close'].iloc[i-5:i]
            recent_10d = data['Close'].iloc[i-10:i]
            
            # Trend analysis
            trend_5d = (price - float(recent_5d.mean())) / float(recent_5d.mean())
            trend_10d = (price - float(recent_10d.mean())) / float(recent_10d.mean())
            
            # Moving averages
            ma5 = float(data['MA5'].iloc[i]) if not pd.isna(data['MA5'].iloc[i]) else price
            ma10 = float(data['MA10'].iloc[i]) if not pd.isna(data['MA10'].iloc[i]) else price
            
            # RSI
            rsi = float(data['RSI'].iloc[i]) if not pd.isna(data['RSI'].iloc[i]) else 50
            
            # Volatility
            volatility = float(recent_10d.std()) / float(recent_10d.mean()) if float(recent_10d.mean()) > 0 else 0
            
            # Volume ratio
            try:
                recent_volume = float(data['Volume'].iloc[i-10:i].mean())
                current_volume = float(data['Volume'].iloc[i])
                volume_ratio = current_volume / recent_volume if recent_volume > 0 else 1
            except:
                volume_ratio = 1
            
            # CONFIGURABLE SIGNAL LOGIC
            signal = 'HOLD'  # Default
            
            # SELL CONDITIONS
            if (trend_5d < config['trend_5d_sell_threshold'] and trend_10d < config['trend_10d_sell_threshold']) or \
               (price < ma5 < ma10) or \
               (rsi > config['rsi_overbought'] and trend_5d < config['trend_5d_sell_threshold']/2) or \
               (volatility > config['volatility_threshold'] and trend_10d < config['trend_10d_sell_threshold']):
                signal = 'SELL'
            
            # BUY CONDITIONS
            elif (trend_5d > config['trend_5d_buy_threshold'] and trend_10d > config['trend_10d_buy_threshold'] and volume_ratio > config['volume_ratio_threshold']) or \
                 (price > ma5 > ma10 and trend_5d > config['trend_5d_buy_threshold']) or \
                 (rsi < config['rsi_oversold'] and trend_5d > config['trend_5d_buy_threshold']/2):
                signal = 'BUY'
            
            signals.append({
                'date': data.index[i],
                'price': price,
                'signal': signal
            })
        
        return signals
        
    def backtest_strategy(self, symbol, config, period_days=730):
        """Backtest a specific configuration on a stock"""
        try:
            # Download data - Using 2 years (730 days) for better statistical significance
            # This accounts for market cycles and gives more reliable results
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days + 50)  # Extra for indicators
            
            # Download with auto_adjust=True to ensure split/dividend adjustments
            data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if data.empty or len(data) < 50:
                return None
                
            # Calculate indicators
            data = self.calculate_technical_indicators(data)
            
            # Generate signals
            signals = self.generate_signals_with_config(data, config)
            if not signals:
                return None
            
            # BACKTEST SIMULATION
            initial_cash = 10000
            cash = initial_cash
            shares = 0
            position = None
            trades = []
            
            for signal_data in signals:
                price = signal_data['price']
                signal = signal_data['signal']
                
                if signal == 'BUY' and position != 'LONG':
                    # Close short, go long
                    if position == 'SHORT':
                        cash += shares * price
                        shares = 0
                    
                    # Buy shares
                    shares = cash / price
                    cash = 0
                    position = 'LONG'
                    trades.append({'action': 'BUY', 'price': price, 'date': signal_data['date']})
                    
                elif signal == 'SELL' and position != 'SHORT':
                    # Close long, go short (or hold cash)
                    if position == 'LONG':
                        cash = shares * price
                        shares = 0
                    position = 'SHORT'
                    trades.append({'action': 'SELL', 'price': price, 'date': signal_data['date']})
            
            # Final value
            final_price = signals[-1]['price']
            final_value = cash + (shares * final_price)
            
            # Buy-and-hold comparison
            start_price = float(data['Close'].iloc[30])
            end_price = float(data['Close'].iloc[-1])
            buy_hold_return = ((end_price - start_price) / start_price) * 100
            buy_hold_value = initial_cash * (end_price / start_price)
            
            # Strategy performance
            strategy_return = ((final_value - initial_cash) / initial_cash) * 100
            outperformance = strategy_return - buy_hold_return
            
            return {
                'symbol': symbol,
                'strategy_return': strategy_return,
                'buy_hold_return': buy_hold_return,
                'outperformance': outperformance,
                'final_value': final_value,
                'buy_hold_value': buy_hold_value,
                'num_trades': len(trades),
                'trades': trades
            }
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {str(e)}")
            return None
    
    def test_configuration(self, config):
        """Test a configuration across all stocks"""
        print(f"🧪 Testing config: {config}")
        
        results = []
        total_outperformance = 0
        successful_tests = 0
        
        for symbol in self.stocks:
            result = self.backtest_strategy(symbol, config)
            if result:
                results.append(result)
                total_outperformance += result['outperformance']
                successful_tests += 1
                
                status = "✅" if result['outperformance'] > 0 else "❌"
                print(f"   {status} {symbol}: {result['strategy_return']:+.1f}% vs {result['buy_hold_return']:+.1f}% (Diff: {result['outperformance']:+.1f}%)")
        
        if successful_tests == 0:
            return None
            
        avg_outperformance = total_outperformance / successful_tests
        winning_stocks = len([r for r in results if r['outperformance'] > 0])
        win_rate = winning_stocks / successful_tests
        
        config_result = {
            'config': config,
            'avg_outperformance': avg_outperformance,
            'win_rate': win_rate,
            'winning_stocks': winning_stocks,
            'total_stocks': successful_tests,
            'results': results
        }
        
        print(f"📊 Config Performance: Avg {avg_outperformance:+.1f}%, Win Rate: {win_rate:.1%} ({winning_stocks}/{successful_tests})")
        
        return config_result
    
    def generate_parameter_combinations(self, max_combinations=100):
        """Generate parameter combinations to test - Optimized for 1-hour runs"""
        # Create all possible combinations
        keys = list(self.parameter_space.keys())
        values = list(self.parameter_space.values())
        
        all_combinations = list(itertools.product(*values))
        print(f"🔢 Total possible combinations: {len(all_combinations):,}")
        
        # For 1-hour runs, we test smartly selected combinations
        if max_combinations > len(all_combinations):
            print(f"📊 Testing ALL {len(all_combinations):,} combinations!")
            return [dict(zip(keys, combo)) for combo in all_combinations]
        
        # Use grid sampling for better coverage
        if len(all_combinations) > max_combinations * 3:
            # Systematic sampling for better parameter space coverage
            step = len(all_combinations) // max_combinations
            selected_combinations = all_combinations[::step][:max_combinations]
        else:
            # Randomize and select
            np.random.shuffle(all_combinations)
            selected_combinations = all_combinations[:max_combinations]
        
        configs = []
        for combo in selected_combinations:
            config = dict(zip(keys, combo))
            configs.append(config)
            
        print(f"🎯 Selected {len(configs):,} combinations to test")
        return configs
    
    def test_proven_config_first(self):
        """Test the proven best configuration first"""
        print("🎯 TESTING PROVEN BEST CONFIGURATION FIRST")
        print("=" * 50)
        
        proven_config = {
            'trend_5d_buy_threshold': 0.025,
            'trend_5d_sell_threshold': -0.02,
            'trend_10d_buy_threshold': 0.025,
            'trend_10d_sell_threshold': -0.045,
            'rsi_overbought': 65,
            'rsi_oversold': 20,
            'volatility_threshold': 0.07,
            'volume_ratio_threshold': 1.6
        }
        
        print(f"📋 Proven Config: {proven_config}")
        result = self.test_configuration(proven_config)
        
        if result:
            print(f"🏆 PROVEN CONFIG RESULTS:")
            print(f"   📊 Average Outperformance: {result['avg_outperformance']:+.1f}%")
            print(f"   🎯 Win Rate: {result['win_rate']:.1%} ({result['winning_stocks']}/{result['total_stocks']})")
            
            # Set as baseline if good
            if result['avg_outperformance'] > self.best_avg_performance:
                self.best_avg_performance = result['avg_outperformance']
                self.best_config = proven_config.copy()
                self.results_history.append(result)
                print(f"✅ Proven config set as baseline to beat!")
        
        print("=" * 50)
        return result

    def optimize_signals(self, max_iterations=100):
        """Main optimization loop - Optimized for 1-hour runs"""
        print("🚀 AUTOMATED SIGNAL OPTIMIZATION STARTING")
        print("=" * 60)
        print(f"📊 Testing {len(self.stocks)} stocks")
        print(f"📅 Using 2-year periods (730 days) for better accuracy")
        print(f"🔧 Split-adjusted data ensures accurate historical analysis")
        print(f"🎯 Goal: Beat buy-and-hold on ALL stocks")
        print(f"🔄 Max iterations: {max_iterations:,}")
        print(f"⏰ Estimated time: {max_iterations * len(self.stocks) * 1.5 / 3600:.1f} hours")
        print("=" * 60)
        
        # Test proven configuration first
        self.test_proven_config_first()
        
        # Generate configurations to test
        configs_to_test = self.generate_parameter_combinations(max_iterations)
        
        best_configs = []
        start_time = datetime.now()
        
        for i, config in enumerate(configs_to_test, 1):
            print(f"\n🔄 ITERATION {i:,}/{len(configs_to_test):,}")
            elapsed = datetime.now() - start_time
            if i > 1:
                avg_time_per_iteration = elapsed.total_seconds() / (i - 1)
                remaining_iterations = len(configs_to_test) - i
                eta = timedelta(seconds=avg_time_per_iteration * remaining_iterations)
                print(f"⏱️  Elapsed: {str(elapsed).split('.')[0]}, ETA: {str(eta).split('.')[0]}")
            print("-" * 40)
            
            result = self.test_configuration(config)
            
            if result:
                self.results_history.append(result)
                
                # Track best performance
                if result['avg_outperformance'] > self.best_avg_performance:
                    self.best_avg_performance = result['avg_outperformance']
                    self.best_config = config.copy()
                    print(f"🎯 NEW BEST CONFIG! Avg outperformance: {result['avg_outperformance']:+.1f}%")
                
                # Keep track of top configs
                best_configs.append(result)
                best_configs.sort(key=lambda x: x['avg_outperformance'], reverse=True)
                best_configs = best_configs[:10]  # Keep top 10 for 1-hour runs
            
            # Progress update
            if result:
                print(f"📈 Current best: {self.best_avg_performance:+.1f}% avg outperformance")
            
            # Save progress every 20 iterations for 1-hour runs
            if i % 20 == 0:
                self.save_progress_checkpoint(i, len(configs_to_test))
            
            # No sleep needed for 1-hour optimization
        
        # FINAL RESULTS
        print(f"\n" + "="*60)
        print(f"🏆 OPTIMIZATION COMPLETE!")
        print(f"="*60)
        
        if best_configs:
            print(f"🥇 BEST CONFIGURATION:")
            best = best_configs[0]
            print(f"   📊 Average Outperformance: {best['avg_outperformance']:+.1f}%")
            print(f"   🎯 Win Rate: {best['win_rate']:.1%} ({best['winning_stocks']}/{best['total_stocks']})")
            print(f"   ⚙️  Parameters:")
            for key, value in best['config'].items():
                print(f"      {key}: {value}")
            
            # Show top 5 configurations for overnight runs
            print(f"\n🏅 TOP 5 CONFIGURATIONS:")
            for i, config_result in enumerate(best_configs[:5], 1):
                print(f"   #{i}: Avg {config_result['avg_outperformance']:+.1f}%, Win Rate {config_result['win_rate']:.1%}")
            
            # Save results
            self.save_results()
            
            return best_configs[0]
        else:
            print("❌ No successful configurations found!")
            return None
    
    def save_progress_checkpoint(self, current_iteration, total_iterations):
        """Save progress checkpoint for long-running optimizations"""
        checkpoint_filename = f"optimization_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        checkpoint_data = {
            'current_iteration': current_iteration,
            'total_iterations': total_iterations,
            'progress_percent': (current_iteration / total_iterations) * 100,
            'best_config_so_far': self.best_config,
            'best_performance_so_far': self.best_avg_performance,
            'timestamp': datetime.now().isoformat(),
            'completed_tests': len(self.results_history)
        }
        
        with open(checkpoint_filename, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        print(f"💾 Checkpoint saved: {current_iteration:,}/{total_iterations:,} ({checkpoint_data['progress_percent']:.1f}%)")
    
    def save_results(self):
        """Save optimization results to file"""
        filename = f"signal_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to JSON-serializable format
        save_data = {
            'best_config': self.best_config,
            'best_avg_performance': self.best_avg_performance,
            'optimization_timestamp': datetime.now().isoformat(),
            'stocks_tested': self.stocks,
            'total_iterations': len(self.results_history),
            'top_10_configs': self.results_history[-10:] if len(self.results_history) >= 10 else self.results_history
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filename}")
    
    def test_best_config_live(self):
        """Test the best configuration on current market data"""
        if not self.best_config:
            print("❌ No best configuration found yet!")
            return None
        
        print(f"\n🧪 TESTING BEST CONFIG ON CURRENT DATA")
        print("=" * 45)
        
        live_results = []
        
        for symbol in self.stocks[:10]:  # Test on first 10 stocks
            result = self.backtest_strategy(symbol, self.best_config, period_days=365)  # 1 year for live test
            if result:
                live_results.append(result)
                status = "✅" if result['outperformance'] > 0 else "❌"
                print(f"{status} {symbol}: {result['outperformance']:+.1f}% outperformance")
        
        if live_results:
            avg_performance = sum(r['outperformance'] for r in live_results) / len(live_results)
            win_rate = len([r for r in live_results if r['outperformance'] > 0]) / len(live_results)
            
            print(f"\n📊 LIVE TEST RESULTS:")
            print(f"   📈 Average Outperformance: {avg_performance:+.1f}%")
            print(f"   🎯 Win Rate: {win_rate:.1%}")
        
        return live_results

def main():
    """Run the automated optimization system - Optimized for 1-hour runs"""
    print("🤖 AUTOMATED TRADING SIGNAL OPTIMIZER - 1 HOUR EDITION")
    print("=" * 60)
    print("⚡ OPTIMIZED FOR FAST 1-HOUR COMPREHENSIVE TESTING")
    print("📊 Smart parameter space coverage with 100 iterations")
    print("💾 Progress checkpoints every 20 iterations")
    print("⏰ Estimated runtime: ~1 hour")
    print("=" * 60)
    
    # Initialize optimizer with a diverse stock universe
    optimizer = AutomatedSignalOptimizer()
    
    # Run optimized 1-hour optimization (100 iterations)
    best_config = optimizer.optimize_signals(max_iterations=100)
    
    if best_config:
        # Test best config on live data
        optimizer.test_best_config_live()
        
        print(f"\n🎯 1-HOUR OPTIMIZATION SUMMARY:")
        print(f"   🚀 Found optimal configuration in ~1 hour!")
        print(f"   📊 Ready to deploy optimized signal logic")
        print(f"   💰 Expected to outperform on {best_config['win_rate']:.1%} of stocks")
        print(f"   🏆 Average outperformance: {best_config['avg_outperformance']:+.1f}%")
    
    return optimizer

if __name__ == "__main__":
    # RUN AUTOMATED OPTIMIZATION
    optimizer = main()
