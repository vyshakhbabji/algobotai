#!/usr/bin/env python3
"""
SINGLE STOCK FOCUSED TRADER - NVDA $10K
Simple, focused trading on one stock with optimized signals
Shows exactly how the strategy performs with real position management
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class SingleStockTrader:
    def __init__(self, symbol='NVDA', starting_capital=10000):
        self.symbol = symbol
        self.starting_capital = starting_capital
        
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
        
        # Date setup for 3-month test (with extra buffer for indicators)
        self.test_end_date = datetime.now().date()
        self.test_start_date = self.test_end_date - timedelta(days=120)  # 3 months + buffer
        
        print(f"📈 SINGLE STOCK TRADER - {symbol}")
        print(f"💰 Starting Capital: ${starting_capital:,}")
        print(f"📅 Testing Period: {self.test_start_date} to {self.test_end_date}")
        
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
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
    
    def generate_trading_signal(self, data, config, current_idx):
        """Generate trading signal for current day"""
        try:
            # Handle MultiIndex columns if present
            if hasattr(data.columns, 'levels'):
                symbol = data.columns[0][1] if len(data.columns[0]) > 1 else self.symbol
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
            
            return {
                'signal': signal,
                'price': price,
                'trend_5d': trend_5d,
                'trend_10d': trend_10d,
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'ma5': ma5,
                'ma10': ma10
            }
            
        except Exception as e:
            return {
                'signal': 'HOLD',
                'price': price if 'price' in locals() else 0,
                'trend_5d': 0,
                'trend_10d': 0,
                'rsi': 50,
                'volatility': 0,
                'volume_ratio': 1,
                'ma5': 0,
                'ma10': 0
            }
    
    def run_trading_simulation(self):
        """Run the 3-month trading simulation"""
        print(f"\n🚀 RUNNING {self.symbol} TRADING SIMULATION")
        print(f"=" * 50)
        
        # Download data
        try:
            data = yf.download(self.symbol, start=self.test_start_date, end=self.test_end_date, progress=False)
            if data.empty:
                print(f"❌ No data available for {self.symbol}")
                return None
            
            print(f"✅ Downloaded {len(data)} trading days")
            
            # Calculate indicators
            data = self.calculate_technical_indicators(data)
            
            # Portfolio tracking
            portfolio = {
                'cash': self.starting_capital,
                'shares': 0,
                'position': None,  # 'LONG', 'SHORT', or None
                'trades': [],
                'daily_values': []
            }
            
            # Daily simulation - Start earlier to catch momentum signals
            total_days = len(data)
            for day_idx in range(15, total_days):  # Start after minimal indicator warmup
                current_date = data.index[day_idx]
                
                # Get trading signal
                signal_data = self.generate_trading_signal(data, self.best_config, day_idx)
                current_price = signal_data['price']
                signal = signal_data['signal']
                
                # Calculate current portfolio value
                portfolio_value = portfolio['cash'] + (portfolio['shares'] * current_price)
                
                # TRADING LOGIC
                if signal == 'BUY' and portfolio['position'] != 'LONG':
                    # Buy signal - go long
                    if portfolio['cash'] > 100:  # Minimum trade size
                        # Use 95% of cash (keep 5% buffer for fees)
                        buy_amount = portfolio['cash'] * 0.95
                        shares_to_buy = buy_amount / current_price
                        
                        portfolio['shares'] += shares_to_buy
                        portfolio['cash'] -= buy_amount
                        portfolio['position'] = 'LONG'
                        
                        portfolio['trades'].append({
                            'date': current_date,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'amount': buy_amount,
                            'signal_data': signal_data
                        })
                        
                        print(f"📈 BUY: {current_date.strftime('%Y-%m-%d')} - ${current_price:.2f} - {shares_to_buy:.2f} shares - ${buy_amount:,.2f}")
                
                elif signal == 'SELL' and portfolio['position'] == 'LONG':
                    # Sell signal - close long position
                    if portfolio['shares'] > 0:
                        sell_amount = portfolio['shares'] * current_price
                        shares_sold = portfolio['shares']
                        
                        portfolio['cash'] += sell_amount
                        portfolio['shares'] = 0
                        portfolio['position'] = None
                        
                        # Calculate profit/loss on this trade
                        last_buy = None
                        for trade in reversed(portfolio['trades']):
                            if trade['action'] == 'BUY':
                                last_buy = trade
                                break
                        
                        if last_buy:
                            profit = sell_amount - last_buy['amount']
                            profit_pct = (profit / last_buy['amount']) * 100
                        else:
                            profit = 0
                            profit_pct = 0
                        
                        portfolio['trades'].append({
                            'date': current_date,
                            'action': 'SELL',
                            'price': current_price,
                            'shares': shares_sold,
                            'amount': sell_amount,
                            'profit': profit,
                            'profit_pct': profit_pct,
                            'signal_data': signal_data
                        })
                        
                        print(f"📉 SELL: {current_date.strftime('%Y-%m-%d')} - ${current_price:.2f} - {shares_sold:.2f} shares - ${sell_amount:,.2f} - Profit: ${profit:,.2f} ({profit_pct:+.1f}%)")
                
                # Record daily portfolio value
                current_portfolio_value = portfolio['cash'] + (portfolio['shares'] * current_price)
                portfolio['daily_values'].append({
                    'date': current_date,
                    'price': current_price,
                    'portfolio_value': current_portfolio_value,
                    'cash': portfolio['cash'],
                    'shares': portfolio['shares'],
                    'position': portfolio['position'],
                    'signal': signal
                })
                
                # Progress update
                if day_idx % 15 == 0:
                    print(f"📊 {current_date.strftime('%Y-%m-%d')}: ${current_price:.2f} - Portfolio: ${current_portfolio_value:,.2f} - Signal: {signal}")
            
            # Final calculations
            final_price = data['Close'].iloc[-1]
            if hasattr(data.columns, 'levels'):
                symbol_name = data.columns[0][1] if len(data.columns[0]) > 1 else self.symbol
                final_price = float(data[('Close', symbol_name)].iloc[-1])
            else:
                final_price = float(data['Close'].iloc[-1])
            
            final_portfolio_value = portfolio['cash'] + (portfolio['shares'] * final_price)
            
            # Buy-and-hold comparison
            start_price = data['Close'].iloc[15]  # Same starting point as trading
            if hasattr(data.columns, 'levels'):
                start_price = float(data[('Close', symbol_name)].iloc[15])
            else:
                start_price = float(data['Close'].iloc[15])
            
            buy_hold_value = self.starting_capital * (final_price / start_price)
            
            # Performance metrics
            total_return = ((final_portfolio_value - self.starting_capital) / self.starting_capital) * 100
            buy_hold_return = ((buy_hold_value - self.starting_capital) / self.starting_capital) * 100
            outperformance = total_return - buy_hold_return
            
            return {
                'symbol': self.symbol,
                'starting_capital': self.starting_capital,
                'final_value': final_portfolio_value,
                'total_return': total_return,
                'buy_hold_value': buy_hold_value,
                'buy_hold_return': buy_hold_return,
                'outperformance': outperformance,
                'num_trades': len(portfolio['trades']),
                'trades': portfolio['trades'],
                'daily_values': portfolio['daily_values'],
                'final_cash': portfolio['cash'],
                'final_shares': portfolio['shares'],
                'final_price': final_price,
                'start_price': start_price
            }
            
        except Exception as e:
            print(f"❌ Error in simulation: {str(e)}")
            return None
    
    def analyze_results(self, results):
        """Analyze and display trading results"""
        print(f"\n" + "="*50)
        print(f"🏆 {self.symbol} TRADING RESULTS - 3 MONTHS")
        print(f"="*50)
        
        print(f"💰 PERFORMANCE:")
        print(f"   🏦 Starting Capital: ${results['starting_capital']:,.2f}")
        print(f"   📈 Final Value: ${results['final_value']:,.2f}")
        print(f"   💵 Total Profit: ${results['final_value'] - results['starting_capital']:,.2f}")
        print(f"   📊 Total Return: {results['total_return']:+.1f}%")
        
        print(f"\n📊 BENCHMARK COMPARISON:")
        print(f"   🎯 Buy-Hold Value: ${results['buy_hold_value']:,.2f}")
        print(f"   📈 Buy-Hold Return: {results['buy_hold_return']:+.1f}%")
        print(f"   🏆 Outperformance: {results['outperformance']:+.1f}%")
        print(f"   💎 Alpha Generated: ${results['final_value'] - results['buy_hold_value']:,.2f}")
        
        print(f"\n🔄 TRADING ACTIVITY:")
        print(f"   📈 Total Trades: {results['num_trades']}")
        print(f"   💰 Final Cash: ${results['final_cash']:,.2f}")
        print(f"   📊 Final Shares: {results['final_shares']:.2f}")
        print(f"   💲 Price Range: ${results['start_price']:.2f} → ${results['final_price']:.2f}")
        
        # Trade analysis
        trades = results['trades']
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        if sell_trades:
            profitable_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
            win_rate = len(profitable_trades) / len(sell_trades) if sell_trades else 0
            
            total_profit = sum(t.get('profit', 0) for t in sell_trades)
            avg_profit_pct = sum(t.get('profit_pct', 0) for t in sell_trades) / len(sell_trades)
            
            print(f"\n📈 TRADE ANALYSIS:")
            print(f"   🎯 Win Rate: {win_rate:.1%} ({len(profitable_trades)}/{len(sell_trades)})")
            print(f"   💰 Total Trading Profit: ${total_profit:,.2f}")
            print(f"   📊 Average Trade Return: {avg_profit_pct:+.1f}%")
            
            if profitable_trades:
                best_trade = max(profitable_trades, key=lambda x: x.get('profit_pct', 0))
                print(f"   🏆 Best Trade: {best_trade['profit_pct']:+.1f}% on {best_trade['date'].strftime('%Y-%m-%d')}")
            
            if len(sell_trades) > len(profitable_trades):
                worst_trades = [t for t in sell_trades if t.get('profit', 0) <= 0]
                if worst_trades:
                    worst_trade = min(worst_trades, key=lambda x: x.get('profit_pct', 0))
                    print(f"   📉 Worst Trade: {worst_trade['profit_pct']:+.1f}% on {worst_trade['date'].strftime('%Y-%m-%d')}")
        
        # Recent trade details
        if trades:
            print(f"\n📋 RECENT TRADES:")
            for trade in trades[-5:]:  # Last 5 trades
                action_emoji = "📈" if trade['action'] == 'BUY' else "📉"
                profit_info = ""
                if trade['action'] == 'SELL' and 'profit_pct' in trade:
                    profit_info = f" - {trade['profit_pct']:+.1f}%"
                print(f"   {action_emoji} {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} ${trade['price']:.2f}{profit_info}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.symbol.lower()}_trading_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {filename}")
        
        return results

def main():
    """Run single stock trading simulation"""
    print("📈 SINGLE STOCK FOCUSED TRADER")
    print("🎯 Testing optimized signals on one stock with real money management")
    print("=" * 60)
    
    # Choose stock (NVDA or AAPL)
    symbol = 'NVDA'  # Change to 'AAPL' if preferred
    
    trader = SingleStockTrader(symbol=symbol, starting_capital=10000)
    results = trader.run_trading_simulation()
    
    if results:
        trader.analyze_results(results)
        
        print(f"\n🎯 KEY INSIGHTS:")
        if results['outperformance'] > 0:
            print(f"   ✅ Strategy OUTPERFORMED buy-and-hold by {results['outperformance']:+.1f}%")
        else:
            print(f"   ❌ Strategy UNDERPERFORMED buy-and-hold by {abs(results['outperformance']):.1f}%")
        
        print(f"   🔄 Made {results['num_trades']} trades over 3 months")
        print(f"   💰 Turned ${results['starting_capital']:,} into ${results['final_value']:,.2f}")
        
        return trader, results
    else:
        print("❌ Simulation failed!")
        return None, None

if __name__ == "__main__":
    trader, results = main()
