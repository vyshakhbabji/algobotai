#!/usr/bin/env python3
"""
REALISTIC FORWARD TEST - 3 MONTH SIMULATION
Train on 1 year of data ending 3 months ago
Test on the actual 3 months that followed
Simulates real trading with $100,000 starting capital
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class RealisticForwardTest:
    def __init__(self, starting_capital=100000):
        self.starting_capital = starting_capital
        
        # Our momentum portfolio stocks (winning foundation)
        self.stocks = ['AMD', 'GE', 'PLTR', 'MSFT', 'NVDA', 'JNJ', 'CAT', 'GOOGL']
        
        # The proven best configuration from our optimizer
        self.best_config = {
            'trend_5d_buy_threshold': 0.025,
            'trend_5d_sell_threshold': -0.02,
            'trend_10d_buy_threshold': 0.025,
            'trend_10d_sell_threshold': -0.045,
            'rsi_overbought': 65,
            'rsi_oversold': 20,
            'volatility_threshold': 0.07,
            'volume_ratio_threshold': 1.6
        }
        
        # Date setup for realistic test
        self.test_end_date = datetime.now().date()  # Today (August 8, 2025 - system date)
        self.test_start_date = self.test_end_date - timedelta(days=90)  # 3 months ago (May 10, 2025)
        self.train_end_date = self.test_start_date - timedelta(days=1)  # End training 3 months ago
        self.train_start_date = self.train_end_date - timedelta(days=365)  # 1 year of training data
        
        print(f"📅 REALISTIC FORWARD TEST DATES:")
        print(f"   🏋️  Training Period: {self.train_start_date} to {self.train_end_date}")
        print(f"   🧪 Testing Period: {self.test_start_date} to {self.test_end_date}")
        print(f"   💰 Starting Capital: ${self.starting_capital:,.2f}")
        
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators (same as optimizer)"""
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
        """Generate trading signals (same logic as optimizer)"""
        signals = []
        
        for i in range(30, len(data)):
            try:
                # Handle MultiIndex columns if present
                if hasattr(data.columns, 'levels'):
                    symbol = data.columns[0][1] if len(data.columns[0]) > 1 else 'UNKNOWN'
                    price = float(data[('Close', symbol)].iloc[i])
                    volume_col = ('Volume', symbol) if ('Volume', symbol) in data.columns else None
                    volume = float(data[volume_col].iloc[i]) if volume_col else 1000000
                else:
                    price = float(data['Close'].iloc[i])
                    volume = float(data['Volume'].iloc[i]) if 'Volume' in data.columns else 1000000
                
                # Get close prices for analysis
                if hasattr(data.columns, 'levels'):
                    close_col = ('Close', symbol) if ('Close', symbol) in data.columns else data.columns[0]
                    close_series = data[close_col]
                else:
                    close_series = data['Close']
                
                # Multi-timeframe analysis
                recent_5d = close_series.iloc[i-5:i]
                recent_10d = close_series.iloc[i-10:i]
                
                # Trend analysis
                trend_5d = (price - float(recent_5d.mean())) / float(recent_5d.mean())
                trend_10d = (price - float(recent_10d.mean())) / float(recent_10d.mean())
                
                # Moving averages
                if hasattr(data.columns, 'levels'):
                    ma5_col = ('MA5', symbol) if ('MA5', symbol) in data.columns else None
                    ma10_col = ('MA10', symbol) if ('MA10', symbol) in data.columns else None
                    rsi_col = ('RSI', symbol) if ('RSI', symbol) in data.columns else None
                    
                    ma5 = float(data[ma5_col].iloc[i]) if ma5_col and not pd.isna(data[ma5_col].iloc[i]) else price
                    ma10 = float(data[ma10_col].iloc[i]) if ma10_col and not pd.isna(data[ma10_col].iloc[i]) else price
                    rsi = float(data[rsi_col].iloc[i]) if rsi_col and not pd.isna(data[rsi_col].iloc[i]) else 50
                else:
                    ma5 = float(data['MA5'].iloc[i]) if not pd.isna(data['MA5'].iloc[i]) else price
                    ma10 = float(data['MA10'].iloc[i]) if not pd.isna(data['MA10'].iloc[i]) else price
                    rsi = float(data['RSI'].iloc[i]) if not pd.isna(data['RSI'].iloc[i]) else 50
                
                # Volatility
                volatility = float(recent_10d.std()) / float(recent_10d.mean()) if float(recent_10d.mean()) > 0 else 0
                
                # Volume ratio
                if hasattr(data.columns, 'levels') and volume_col:
                    recent_volume = float(data[volume_col].iloc[i-10:i].mean())
                elif 'Volume' in data.columns:
                    recent_volume = float(data['Volume'].iloc[i-10:i].mean())
                else:
                    recent_volume = 1000000
                
                volume_ratio = volume / recent_volume if recent_volume > 0 else 1
                
                # SIGNAL LOGIC (same as optimizer)
                signal = 'HOLD'
                
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
                    'signal': signal,
                    'trend_5d': trend_5d,
                    'trend_10d': trend_10d,
                    'rsi': rsi,
                    'volatility': volatility,
                    'volume_ratio': volume_ratio
                })
                
            except Exception as e:
                continue
        
        return signals
    
    def run_forward_test_single_stock(self, symbol):
        """Run forward test on a single stock"""
        try:
            # Download training data (1 year ending 3 months ago)
            train_data = yf.download(symbol, start=self.train_start_date, end=self.train_end_date, progress=False)
            if train_data.empty:
                return None
            
            # Download test data (the 3 months we're testing)
            test_data = yf.download(symbol, start=self.test_start_date, end=self.test_end_date, progress=False)
            if test_data.empty:
                return None
            
            # Calculate indicators on training data
            train_data = self.calculate_technical_indicators(train_data)
            
            # Calculate indicators on test data
            test_data = self.calculate_technical_indicators(test_data)
            
            # Generate signals on test data (this is what we would have traded)
            test_signals = self.generate_signals_with_config(test_data, self.best_config)
            
            if not test_signals:
                return None
            
            # SIMULATION: Trade with signals during test period
            stock_allocation = self.starting_capital / len(self.stocks)  # Equal weight
            cash = stock_allocation
            shares = 0
            position = None
            trades = []
            daily_values = []
            
            for signal_data in test_signals:
                price = signal_data['price']
                signal = signal_data['signal']
                date = signal_data['date']
                
                if signal == 'BUY' and position != 'LONG' and cash > 0:
                    # Buy shares
                    shares = cash / price
                    cash = 0
                    position = 'LONG'
                    trades.append({
                        'action': 'BUY', 
                        'price': price, 
                        'shares': shares,
                        'date': date,
                        'value': shares * price
                    })
                    
                elif signal == 'SELL' and position == 'LONG':
                    # Sell shares
                    cash = shares * price
                    value_sold = shares * price
                    shares = 0
                    position = None
                    trades.append({
                        'action': 'SELL', 
                        'price': price, 
                        'shares': 0,
                        'date': date,
                        'value': value_sold
                    })
                
                # Track daily value
                current_value = cash + (shares * price)
                daily_values.append({
                    'date': date,
                    'value': current_value,
                    'price': price,
                    'position': position
                })
            
            # Final portfolio value
            final_price = test_signals[-1]['price']
            final_value = cash + (shares * final_price)
            
            # Buy-and-hold comparison for test period
            start_price = test_signals[0]['price']
            end_price = test_signals[-1]['price']
            buy_hold_value = stock_allocation * (end_price / start_price)
            
            # Performance metrics
            strategy_return = ((final_value - stock_allocation) / stock_allocation) * 100
            buy_hold_return = ((buy_hold_value - stock_allocation) / stock_allocation) * 100
            outperformance = strategy_return - buy_hold_return
            
            return {
                'symbol': symbol,
                'start_value': stock_allocation,
                'final_value': final_value,
                'buy_hold_value': buy_hold_value,
                'strategy_return': strategy_return,
                'buy_hold_return': buy_hold_return,
                'outperformance': outperformance,
                'num_trades': len(trades),
                'trades': trades,
                'daily_values': daily_values,
                'start_price': start_price,
                'end_price': end_price
            }
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {str(e)}")
            return None
    
    def run_complete_forward_test(self):
        """Run the complete 3-month forward test"""
        print(f"\n🚀 RUNNING REALISTIC 3-MONTH FORWARD TEST")
        print(f"=" * 60)
        print(f"💰 Portfolio: ${self.starting_capital:,.2f} starting capital")
        print(f"📊 Stocks: {', '.join(self.stocks)}")
        print(f"🧪 Testing period: {self.test_start_date} to {self.test_end_date}")
        print(f"=" * 60)
        
        results = []
        total_strategy_value = 0
        total_buy_hold_value = 0
        
        for symbol in self.stocks:
            print(f"\n🔍 Testing {symbol}...")
            
            result = self.run_forward_test_single_stock(symbol)
            
            if result:
                results.append(result)
                total_strategy_value += result['final_value']
                total_buy_hold_value += result['buy_hold_value']
                
                print(f"   ✅ Strategy: ${result['final_value']:,.2f} ({result['strategy_return']:+.1f}%)")
                print(f"   📊 Buy-Hold: ${result['buy_hold_value']:,.2f} ({result['buy_hold_return']:+.1f}%)")
                print(f"   🎯 Outperformance: {result['outperformance']:+.1f}%")
                print(f"   📈 Trades: {result['num_trades']}")
            else:
                print(f"   ❌ Could not test {symbol}")
        
        # Portfolio-level results
        if results:
            portfolio_strategy_return = ((total_strategy_value - self.starting_capital) / self.starting_capital) * 100
            portfolio_buy_hold_return = ((total_buy_hold_value - self.starting_capital) / self.starting_capital) * 100
            portfolio_outperformance = portfolio_strategy_return - portfolio_buy_hold_return
            
            total_profit = total_strategy_value - self.starting_capital
            buy_hold_profit = total_buy_hold_value - self.starting_capital
            alpha_generated = total_profit - buy_hold_profit
            
            print(f"\n" + "="*60)
            print(f"🏆 REALISTIC FORWARD TEST RESULTS")
            print(f"="*60)
            print(f"💰 PORTFOLIO PERFORMANCE:")
            print(f"   🏦 Starting Capital: ${self.starting_capital:,.2f}")
            print(f"   📈 Strategy Final Value: ${total_strategy_value:,.2f}")
            print(f"   📊 Buy-Hold Final Value: ${total_buy_hold_value:,.2f}")
            print(f"   💵 Strategy Profit: ${total_profit:,.2f} ({portfolio_strategy_return:+.1f}%)")
            print(f"   💵 Buy-Hold Profit: ${buy_hold_profit:,.2f} ({portfolio_buy_hold_return:+.1f}%)")
            print(f"   🎯 Alpha Generated: ${alpha_generated:,.2f} ({portfolio_outperformance:+.1f}%)")
            
            winning_stocks = len([r for r in results if r['outperformance'] > 0])
            win_rate = winning_stocks / len(results)
            
            print(f"\n📊 DETAILED METRICS:")
            print(f"   🎯 Win Rate: {win_rate:.1%} ({winning_stocks}/{len(results)} stocks)")
            print(f"   📈 Stocks Tested: {len(results)}")
            print(f"   🔄 Total Trades: {sum(r['num_trades'] for r in results)}")
            print(f"   ⚡ Avg Trades/Stock: {sum(r['num_trades'] for r in results) / len(results):.1f}")
            
            # Best and worst performers
            best_stock = max(results, key=lambda x: x['outperformance'])
            worst_stock = min(results, key=lambda x: x['outperformance'])
            
            print(f"\n🏅 BEST PERFORMER: {best_stock['symbol']}")
            print(f"   💰 Outperformance: {best_stock['outperformance']:+.1f}%")
            print(f"   📈 Strategy Return: {best_stock['strategy_return']:+.1f}%")
            print(f"   🔄 Trades: {best_stock['num_trades']}")
            
            print(f"\n📉 WORST PERFORMER: {worst_stock['symbol']}")
            print(f"   💰 Outperformance: {worst_stock['outperformance']:+.1f}%")
            print(f"   📈 Strategy Return: {worst_stock['strategy_return']:+.1f}%")
            print(f"   🔄 Trades: {worst_stock['num_trades']}")
            
            # Save results
            self.save_forward_test_results(results, {
                'portfolio_strategy_return': portfolio_strategy_return,
                'portfolio_buy_hold_return': portfolio_buy_hold_return,
                'portfolio_outperformance': portfolio_outperformance,
                'total_strategy_value': total_strategy_value,
                'total_buy_hold_value': total_buy_hold_value,
                'alpha_generated': alpha_generated,
                'win_rate': win_rate
            })
            
            return results
        else:
            print("❌ No successful tests!")
            return []
    
    def save_forward_test_results(self, results, portfolio_metrics):
        """Save forward test results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"realistic_forward_test_{timestamp}.json"
        
        save_data = {
            'test_parameters': {
                'starting_capital': self.starting_capital,
                'train_period': f"{self.train_start_date} to {self.train_end_date}",
                'test_period': f"{self.test_start_date} to {self.test_end_date}",
                'stocks_tested': self.stocks,
                'config_used': self.best_config
            },
            'portfolio_results': portfolio_metrics,
            'individual_stock_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {filename}")

def main():
    """Run the realistic forward test"""
    print("🎯 REALISTIC FORWARD TEST - SIGNAL OPTIMIZER")
    print("=" * 50)
    print("📅 Simulating: Train 1 year ago, test last 3 months")
    print("💰 Capital: $100,000")
    print("📊 Portfolio: Momentum stocks")
    print("🧠 Strategy: Proven optimized signals")
    print("=" * 50)
    
    # Run the test
    forward_tester = RealisticForwardTest(starting_capital=100000)
    results = forward_tester.run_complete_forward_test()
    
    return forward_tester, results

if __name__ == "__main__":
    tester, results = main()
