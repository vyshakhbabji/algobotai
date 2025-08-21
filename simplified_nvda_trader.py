#!/usr/bin/env python3
"""
SIMPLIFIED NVDA TRADER - AGGRESSIVE THRESHOLDS
Let's use more aggressive thresholds based on the actual NVDA data analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class SimplifiedNVDATrader:
    def __init__(self, starting_capital=10000):
        self.starting_capital = starting_capital
        
        # AGGRESSIVE MOMENTUM THRESHOLDS based on NVDA analysis
        self.config = {
            'trend_5d_buy_threshold': 0.015,    # Reduced from 0.025 to 0.015 (1.5%)
            'trend_5d_sell_threshold': -0.025,  # Keep sell threshold strict
            'trend_10d_buy_threshold': 0.015,   # Reduced from 0.025 to 0.015 (1.5%)
            'trend_10d_sell_threshold': -0.045, # Keep sell threshold strict
            'rsi_overbought': 85,               # Increased from 65 to 85 (more tolerant)
            'rsi_oversold': 30,                 # Increased from 20 to 30 (more realistic)
            'volatility_threshold': 0.10,       # Increased from 0.07 to 0.10 (10%)
            'volume_ratio_threshold': 1.2       # Reduced from 1.6 to 1.2 (easier to trigger)
        }
        
        print(f"📈 SIMPLIFIED NVDA TRADER - AGGRESSIVE MOMENTUM")
        print(f"💰 Starting Capital: ${starting_capital:,}")
        print(f"🎯 Using momentum-friendly thresholds")
        
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        data = data.copy()
        
        # Moving averages
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA10'] = data['Close'].rolling(10).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
    
    def generate_signal(self, data, idx):
        """Generate simplified signal"""
        try:
            price = float(data['Close'].iloc[idx])
            
            # Trend analysis
            recent_5d = data['Close'].iloc[idx-5:idx]
            recent_10d = data['Close'].iloc[idx-10:idx]
            
            trend_5d = (price - float(recent_5d.mean())) / float(recent_5d.mean())
            trend_10d = (price - float(recent_10d.mean())) / float(recent_10d.mean())
            
            # Technical indicators
            ma5 = float(data['MA5'].iloc[idx]) if not pd.isna(data['MA5'].iloc[idx]) else price
            ma10 = float(data['MA10'].iloc[idx]) if not pd.isna(data['MA10'].iloc[idx]) else price
            rsi = float(data['RSI'].iloc[idx]) if not pd.isna(data['RSI'].iloc[idx]) else 50
            
            # Volume
            recent_volume = float(data['Volume'].iloc[idx-10:idx].mean())
            current_volume = float(data['Volume'].iloc[idx])
            volume_ratio = current_volume / recent_volume if recent_volume > 0 else 1
            
            # SIMPLIFIED SIGNAL LOGIC
            signal = 'HOLD'
            
            # STRONG BUY CONDITIONS (easier to trigger)
            if (trend_5d > self.config['trend_5d_buy_threshold'] and 
                trend_10d > self.config['trend_10d_buy_threshold']):
                signal = 'BUY'
                
            # SELL CONDITIONS (still strict)
            elif (trend_5d < self.config['trend_5d_sell_threshold'] and 
                  trend_10d < self.config['trend_10d_sell_threshold']):
                signal = 'SELL'
            
            return {
                'signal': signal,
                'price': price,
                'trend_5d': trend_5d,
                'trend_10d': trend_10d,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'ma5': ma5,
                'ma10': ma10
            }
            
        except Exception as e:
            return {'signal': 'HOLD', 'price': 0}
    
    def run_simulation(self):
        """Run the simplified trading simulation"""
        print(f"\n🚀 RUNNING SIMPLIFIED NVDA SIMULATION")
        print(f"=" * 50)
        
        # Download 4 months of data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=120)
        
        data = yf.download('NVDA', start=start_date, end=end_date, progress=False)
        data = self.calculate_technical_indicators(data)
        
        print(f"✅ Downloaded {len(data)} trading days")
        print(f"📅 Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Portfolio tracking
        cash = self.starting_capital
        shares = 0
        position = None
        trades = []
        daily_values = []
        
        # Trade simulation
        for i in range(15, len(data)):
            date = data.index[i]
            signal_data = self.generate_signal(data, i)
            price = signal_data['price']
            signal = signal_data['signal']
            
            portfolio_value = cash + (shares * price)
            
            # TRADING LOGIC
            if signal == 'BUY' and position != 'LONG' and cash > 100:
                # Go long
                buy_amount = cash * 0.98  # Use 98% of cash
                shares_bought = buy_amount / price
                shares += shares_bought
                cash -= buy_amount
                position = 'LONG'
                
                trades.append({
                    'date': date,
                    'action': 'BUY',
                    'price': price,
                    'shares': shares_bought,
                    'amount': buy_amount,
                    'trend_5d': signal_data['trend_5d'],
                    'trend_10d': signal_data['trend_10d']
                })
                
                print(f"📈 BUY: {date.strftime('%Y-%m-%d')} - ${price:.2f} - {shares_bought:.2f} shares - ${buy_amount:,.2f}")
                print(f"     Trend 5d: {signal_data['trend_5d']:+.1%}, Trend 10d: {signal_data['trend_10d']:+.1%}")
                
            elif signal == 'SELL' and position == 'LONG' and shares > 0:
                # Sell all shares
                sell_amount = shares * price
                shares_sold = shares
                
                # Calculate profit
                last_buy = [t for t in trades if t['action'] == 'BUY'][-1] if trades else None
                profit = sell_amount - last_buy['amount'] if last_buy else 0
                profit_pct = (profit / last_buy['amount']) * 100 if last_buy else 0
                
                cash += sell_amount
                shares = 0
                position = None
                
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': price,
                    'shares': shares_sold,
                    'amount': sell_amount,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'trend_5d': signal_data['trend_5d'],
                    'trend_10d': signal_data['trend_10d']
                })
                
                print(f"📉 SELL: {date.strftime('%Y-%m-%d')} - ${price:.2f} - {shares_sold:.2f} shares - ${sell_amount:,.2f}")
                print(f"     Profit: ${profit:,.2f} ({profit_pct:+.1f}%) - Trend 5d: {signal_data['trend_5d']:+.1%}")
            
            # Record daily value
            daily_values.append({
                'date': date,
                'price': price,
                'portfolio_value': cash + (shares * price),
                'signal': signal
            })
            
            # Progress update
            if i % 20 == 0:
                current_value = cash + (shares * price)
                print(f"📊 {date.strftime('%Y-%m-%d')}: ${price:.2f} - Portfolio: ${current_value:,.2f} - Signal: {signal}")
        
        # Final results
        final_price = float(data['Close'].iloc[-1])
        final_value = cash + (shares * final_price)
        
        start_price = float(data['Close'].iloc[15])
        buy_hold_value = self.starting_capital * (final_price / start_price)
        
        total_return = ((final_value - self.starting_capital) / self.starting_capital) * 100
        buy_hold_return = ((buy_hold_value - self.starting_capital) / self.starting_capital) * 100
        outperformance = total_return - buy_hold_return
        
        results = {
            'starting_capital': self.starting_capital,
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_value': buy_hold_value,
            'buy_hold_return': buy_hold_return,
            'outperformance': outperformance,
            'num_trades': len(trades),
            'trades': trades,
            'final_cash': cash,
            'final_shares': shares,
            'final_price': final_price,
            'start_price': start_price
        }
        
        print(f"\n" + "="*50)
        print(f"🏆 SIMPLIFIED NVDA RESULTS")
        print(f"="*50)
        print(f"💰 Final Value: ${final_value:,.2f}")
        print(f"📈 Total Return: {total_return:+.1f}%")
        print(f"🎯 Buy-Hold Return: {buy_hold_return:+.1f}%")
        print(f"🏆 Outperformance: {outperformance:+.1f}%")
        print(f"🔄 Total Trades: {len(trades)}")
        print(f"💵 Final Cash: ${cash:,.2f}")
        print(f"📊 Final Shares: {shares:.2f}")
        
        if trades:
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            
            if sell_trades:
                total_profit = sum(t.get('profit', 0) for t in sell_trades)
                avg_return = sum(t.get('profit_pct', 0) for t in sell_trades) / len(sell_trades)
                profitable_trades = len([t for t in sell_trades if t.get('profit', 0) > 0])
                
                print(f"\n📈 TRADE ANALYSIS:")
                print(f"   🎯 Win Rate: {profitable_trades}/{len(sell_trades)} ({profitable_trades/len(sell_trades):.1%})")
                print(f"   💰 Total Trading Profit: ${total_profit:,.2f}")
                print(f"   📊 Average Return per Trade: {avg_return:+.1f}%")
        
        return results

def main():
    """Run simplified NVDA trader"""
    trader = SimplifiedNVDATrader(starting_capital=10000)
    results = trader.run_simulation()
    
    if results['outperformance'] > 0:
        print(f"\n✅ SUCCESS! Strategy beat buy-and-hold by {results['outperformance']:+.1f}%")
    else:
        print(f"\n❌ Strategy underperformed by {abs(results['outperformance']):.1f}%")
    
    return results

if __name__ == "__main__":
    results = main()
