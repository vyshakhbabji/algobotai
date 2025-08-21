#!/usr/bin/env python3
"""
OPTIMIZED PORTFOLIO TRADER - FIND BEST PARAMETERS FOR $100K
Runs portfolio optimization to find the best signal parameters
for real portfolio management with dynamic allocation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
import json
import warnings
warnings.filterwarnings('ignore')

class OptimizedPortfolioTrader:
    def __init__(self, starting_capital=100000):
        self.starting_capital = starting_capital
        self.stocks = ['AMD', 'GE', 'PLTR', 'MSFT', 'NVDA', 'JNJ', 'CAT', 'GOOGL']
        
        # Real trader parameters
        self.min_position_size = 0.05  # 5% minimum position
        self.max_position_size = 0.25  # 25% maximum position
        self.max_positions = 5         # Max 5 positions at once
        self.cash_reserve = 0.10       # Keep 10% cash reserve
        
        # EXPANDED parameter search space for portfolio optimization
        self.parameter_space = {
            'trend_5d_buy_threshold': [0.015, 0.02, 0.025, 0.03, 0.035, 0.04],
            'trend_5d_sell_threshold': [-0.01, -0.015, -0.02, -0.025, -0.03],
            'trend_10d_buy_threshold': [0.01, 0.015, 0.02, 0.025, 0.03, 0.035],
            'trend_10d_sell_threshold': [-0.015, -0.02, -0.025, -0.03, -0.035, -0.04],
            'rsi_overbought': [60, 65, 70, 75, 80],
            'rsi_oversold': [15, 20, 25, 30, 35],
            'volatility_threshold': [0.05, 0.06, 0.07, 0.08, 0.10],
            'volume_ratio_threshold': [1.1, 1.2, 1.4, 1.6, 1.8]
        }
        
        # Date setup for testing
        self.test_end_date = datetime.now().date()
        self.test_start_date = self.test_end_date - timedelta(days=90)  # 3 months
        
        self.best_config = None
        self.best_performance = -float('inf')
        self.optimization_results = []
        
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        data = data.copy()
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
    
    def calculate_signal_strength(self, data, config, current_idx):
        """Calculate signal strength for position sizing"""
        try:
            # Handle data properly
            if hasattr(data.columns, 'levels'):
                symbol = data.columns[0][1] if len(data.columns[0]) > 1 else 'UNKNOWN'
                close_col = ('Close', symbol)
                ma5_col = ('MA5', symbol)
                ma10_col = ('MA10', symbol)
                rsi_col = ('RSI', symbol)
                volume_col = ('Volume', symbol) if ('Volume', symbol) in data.columns else None
            else:
                close_col = 'Close'
                ma5_col = 'MA5'
                ma10_col = 'MA10'
                rsi_col = 'RSI'
                volume_col = 'Volume' if 'Volume' in data.columns else None
            
            price = float(data[close_col].iloc[current_idx])
            close_series = data[close_col]
            
            # Multi-timeframe analysis
            recent_5d = close_series.iloc[current_idx-5:current_idx]
            recent_10d = close_series.iloc[current_idx-10:current_idx]
            recent_20d = close_series.iloc[current_idx-20:current_idx]
            
            # Trend analysis
            trend_5d = (price - float(recent_5d.mean())) / float(recent_5d.mean())
            trend_10d = (price - float(recent_10d.mean())) / float(recent_10d.mean())
            
            # Moving averages and RSI
            ma5 = float(data[ma5_col].iloc[current_idx]) if not pd.isna(data[ma5_col].iloc[current_idx]) else price
            ma10 = float(data[ma10_col].iloc[current_idx]) if not pd.isna(data[ma10_col].iloc[current_idx]) else price
            rsi = float(data[rsi_col].iloc[current_idx]) if not pd.isna(data[rsi_col].iloc[current_idx]) else 50
            
            # Volatility
            volatility = float(recent_10d.std()) / float(recent_10d.mean()) if float(recent_10d.mean()) > 0 else 0
            
            # Volume analysis
            volume_ratio = 1
            if volume_col and volume_col in data.columns:
                recent_volume = float(data[volume_col].iloc[current_idx-10:current_idx].mean())
                current_volume = float(data[volume_col].iloc[current_idx])
                volume_ratio = current_volume / recent_volume if recent_volume > 0 else 1
            
            # SIGNAL LOGIC (same as before but with strength calculation)
            signal = 'HOLD'
            strength = 0
            
            # Buy conditions
            buy_score = 0
            if trend_5d > config['trend_5d_buy_threshold'] and trend_10d > config['trend_10d_buy_threshold']:
                buy_score += 0.4
            if volume_ratio > config['volume_ratio_threshold']:
                buy_score += 0.2
            if price > ma5 > ma10 and trend_5d > config['trend_5d_buy_threshold']:
                buy_score += 0.3
            if rsi < config['rsi_oversold'] and trend_5d > config['trend_5d_buy_threshold']/2:
                buy_score += 0.1
            
            # Sell conditions
            sell_score = 0
            if trend_5d < config['trend_5d_sell_threshold'] and trend_10d < config['trend_10d_sell_threshold']:
                sell_score += 0.4
            if price < ma5 < ma10:
                sell_score += 0.3
            if rsi > config['rsi_overbought'] and trend_5d < config['trend_5d_sell_threshold']/2:
                sell_score += 0.2
            if volatility > config['volatility_threshold'] and trend_10d < config['trend_10d_sell_threshold']:
                sell_score += 0.1
            
            if buy_score > 0.3 and buy_score > sell_score:
                signal = 'BUY'
                strength = min(1.0, buy_score)
            elif sell_score > 0.2 and sell_score > buy_score:
                signal = 'SELL'
                strength = min(1.0, sell_score)
            
            return {
                'signal': signal,
                'strength': strength,
                'trend_5d': trend_5d,
                'trend_10d': trend_10d,
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': volume_ratio
            }
            
        except:
            return {'signal': 'HOLD', 'strength': 0, 'trend_5d': 0, 'trend_10d': 0, 'rsi': 50, 'volatility': 0, 'volume_ratio': 1}
    
    def simulate_portfolio_with_config(self, config):
        """Run portfolio simulation with specific config"""
        try:
            # Load all data
            all_data = {}
            for symbol in self.stocks:
                data = yf.download(symbol, start=self.test_start_date, end=self.test_end_date, progress=False)
                if not data.empty:
                    data = self.calculate_technical_indicators(data)
                    all_data[symbol] = data
            
            if len(all_data) < 4:  # Need at least 4 stocks
                return None
            
            # Get common dates
            common_dates = None
            for symbol, data in all_data.items():
                if common_dates is None:
                    common_dates = set(data.index)
                else:
                    common_dates = common_dates.intersection(set(data.index))
            
            common_dates = sorted(list(common_dates))
            if len(common_dates) < 30:
                return None
            
            # Portfolio simulation
            portfolio = {
                'cash': self.starting_capital,
                'positions': {},
                'total_value': self.starting_capital
            }
            
            trades = []
            
            # Daily simulation
            for date_idx, current_date in enumerate(common_dates[20:], 20):  # Skip first 20 for indicators
                # Update portfolio value
                portfolio_value = portfolio['cash']
                for symbol in portfolio['positions']:
                    if symbol in all_data and current_date in all_data[symbol].index:
                        current_price = float(all_data[symbol].loc[current_date, 'Close'])
                        portfolio['positions'][symbol]['current_value'] = portfolio['positions'][symbol]['shares'] * current_price
                        portfolio_value += portfolio['positions'][symbol]['current_value']
                
                portfolio['total_value'] = portfolio_value
                
                # Get signals
                stock_signals = {}
                for symbol, data in all_data.items():
                    if current_date in data.index:
                        current_idx = data.index.get_loc(current_date)
                        if current_idx >= 20:
                            signal_data = self.calculate_signal_strength(data, config, current_idx)
                            signal_data['price'] = float(data.loc[current_date, 'Close'])
                            stock_signals[symbol] = signal_data
                
                # SELL decisions (risk management)
                positions_to_close = []
                for symbol, position in portfolio['positions'].items():
                    if symbol in stock_signals:
                        signal_data = stock_signals[symbol]
                        current_price = signal_data['price']
                        
                        # Sell conditions
                        if (signal_data['signal'] == 'SELL' and signal_data['strength'] > 0.3) or \
                           (current_price < position['avg_cost'] * 0.90) or \
                           (current_price > position['avg_cost'] * 1.30):
                            positions_to_close.append(symbol)
                
                # Execute sells
                for symbol in positions_to_close:
                    if symbol in portfolio['positions'] and symbol in stock_signals:
                        position = portfolio['positions'][symbol]
                        sell_price = stock_signals[symbol]['price']
                        sell_value = position['shares'] * sell_price
                        portfolio['cash'] += sell_value
                        
                        profit = sell_value - (position['shares'] * position['avg_cost'])
                        trades.append({
                            'action': 'SELL',
                            'symbol': symbol,
                            'profit': profit,
                            'date': current_date
                        })
                        
                        del portfolio['positions'][symbol]
                
                # BUY decisions
                buy_candidates = []
                for symbol, signal_data in stock_signals.items():
                    if signal_data['signal'] == 'BUY' and symbol not in portfolio['positions']:
                        # Position sizing based on signal strength
                        base_size = self.min_position_size + (signal_data['strength'] * (self.max_position_size - self.min_position_size))
                        # Adjust for volatility
                        vol_factor = max(0.5, 1 - signal_data['volatility'])
                        position_size = base_size * vol_factor
                        
                        buy_candidates.append({
                            'symbol': symbol,
                            'signal_data': signal_data,
                            'position_size': position_size,
                            'score': signal_data['strength']
                        })
                
                # Execute best buys
                buy_candidates.sort(key=lambda x: x['score'], reverse=True)
                current_positions = len(portfolio['positions'])
                
                for candidate in buy_candidates:
                    if current_positions >= self.max_positions:
                        break
                    
                    symbol = candidate['symbol']
                    signal_data = candidate['signal_data']
                    position_size = candidate['position_size']
                    
                    available_cash = portfolio['cash'] * (1 - self.cash_reserve)
                    purchase_amount = min(available_cash, portfolio_value * position_size)
                    
                    if purchase_amount >= self.starting_capital * self.min_position_size:
                        buy_price = signal_data['price']
                        shares = purchase_amount / buy_price
                        
                        portfolio['cash'] -= purchase_amount
                        portfolio['positions'][symbol] = {
                            'shares': shares,
                            'avg_cost': buy_price,
                            'current_value': purchase_amount
                        }
                        
                        trades.append({
                            'action': 'BUY',
                            'symbol': symbol,
                            'value': purchase_amount,
                            'date': current_date
                        })
                        
                        current_positions += 1
            
            # Final value calculation
            final_value = portfolio['cash']
            for symbol in portfolio['positions']:
                if symbol in all_data and len(common_dates) > 0:
                    final_price = float(all_data[symbol].loc[common_dates[-1], 'Close'])
                    final_value += portfolio['positions'][symbol]['shares'] * final_price
            
            # Buy-and-hold comparison
            buy_hold_value = 0
            equal_allocation = self.starting_capital / len(self.stocks)
            for symbol in self.stocks:
                if symbol in all_data and len(common_dates) > 20:
                    start_price = float(all_data[symbol].loc[common_dates[20], 'Close'])
                    end_price = float(all_data[symbol].loc[common_dates[-1], 'Close'])
                    buy_hold_value += equal_allocation * (end_price / start_price)
            
            strategy_return = ((final_value - self.starting_capital) / self.starting_capital) * 100
            buy_hold_return = ((buy_hold_value - self.starting_capital) / self.starting_capital) * 100
            outperformance = strategy_return - buy_hold_return
            
            return {
                'config': config,
                'final_value': final_value,
                'strategy_return': strategy_return,
                'buy_hold_return': buy_hold_return,
                'outperformance': outperformance,
                'num_trades': len(trades),
                'trades': trades
            }
            
        except Exception as e:
            return None
    
    def optimize_portfolio_parameters(self, max_iterations=50):
        """Optimize parameters specifically for portfolio trading"""
        print("🚀 OPTIMIZING PORTFOLIO PARAMETERS FOR $100K TRADING")
        print("=" * 60)
        print(f"💰 Starting Capital: ${self.starting_capital:,}")
        print(f"📊 Stocks: {', '.join(self.stocks)}")
        print(f"🎯 Goal: Maximize portfolio returns with real position sizing")
        print(f"🔄 Testing {max_iterations} parameter combinations")
        print("=" * 60)
        
        # Generate parameter combinations
        keys = list(self.parameter_space.keys())
        values = list(self.parameter_space.values())
        all_combinations = list(itertools.product(*values))
        
        # Sample combinations
        np.random.shuffle(all_combinations)
        test_combinations = all_combinations[:max_iterations]
        
        best_results = []
        
        for i, combo in enumerate(test_combinations, 1):
            config = dict(zip(keys, combo))
            
            print(f"\n🔄 Testing Configuration {i}/{max_iterations}")
            print(f"   Config: {config}")
            
            result = self.simulate_portfolio_with_config(config)
            
            if result:
                self.optimization_results.append(result)
                
                print(f"   ✅ Return: {result['strategy_return']:+.1f}% | Outperformance: {result['outperformance']:+.1f}% | Trades: {result['num_trades']}")
                
                if result['outperformance'] > self.best_performance:
                    self.best_performance = result['outperformance']
                    self.best_config = config.copy()
                    print(f"   🎯 NEW BEST! Outperformance: {result['outperformance']:+.1f}%")
                
                best_results.append(result)
                best_results.sort(key=lambda x: x['outperformance'], reverse=True)
                best_results = best_results[:10]  # Keep top 10
            else:
                print(f"   ❌ Configuration failed")
            
            if i % 10 == 0:
                print(f"\n📊 Progress Update:")
                print(f"   Current Best: {self.best_performance:+.1f}% outperformance")
                print(f"   Successful Tests: {len(self.optimization_results)}")
        
        # Final results
        print(f"\n" + "="*60)
        print(f"🏆 PORTFOLIO OPTIMIZATION COMPLETE")
        print(f"="*60)
        
        if best_results:
            best = best_results[0]
            print(f"🥇 BEST CONFIGURATION:")
            print(f"   📊 Strategy Return: {best['strategy_return']:+.1f}%")
            print(f"   🎯 Outperformance: {best['outperformance']:+.1f}%")
            print(f"   💰 Final Value: ${best['final_value']:,.2f}")
            print(f"   🔄 Trades: {best['num_trades']}")
            print(f"   ⚙️  Parameters:")
            for key, value in best['config'].items():
                print(f"      {key}: {value}")
            
            print(f"\n🏅 TOP 5 CONFIGURATIONS:")
            for i, result in enumerate(best_results[:5], 1):
                print(f"   #{i}: {result['outperformance']:+.1f}% outperformance, {result['strategy_return']:+.1f}% return")
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"portfolio_optimization_results_{timestamp}.json"
            
            save_data = {
                'best_config': self.best_config,
                'best_performance': self.best_performance,
                'top_10_results': best_results,
                'all_results': self.optimization_results,
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to {filename}")
            
            return best_results[0]
        else:
            print("❌ No successful configurations found!")
            return None

def main():
    """Run portfolio optimization"""
    print("🏛️ OPTIMIZED PORTFOLIO TRADER")
    print("💰 Finding the best parameters for $100K portfolio management")
    print("🎯 Real position sizing, risk management, and capital allocation")
    print("=" * 60)
    
    optimizer = OptimizedPortfolioTrader(starting_capital=100000)
    best_result = optimizer.optimize_portfolio_parameters(max_iterations=50)
    
    if best_result:
        print(f"\n🎯 OPTIMIZATION SUMMARY:")
        print(f"   🚀 Best configuration found!")
        print(f"   📊 Outperformed buy-and-hold by {best_result['outperformance']:+.1f}%")
        print(f"   💰 Turned $100K into ${best_result['final_value']:,.2f}")
        print(f"   🔄 Used {best_result['num_trades']} trades")
        
        return optimizer, best_result
    else:
        print("❌ Optimization failed!")
        return None, None

if __name__ == "__main__":
    optimizer, result = main()
