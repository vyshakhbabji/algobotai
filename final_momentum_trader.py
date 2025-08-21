#!/usr/bin/env python3
"""
FINAL MOMENTUM TRADER - NVDA vs AAPL
Test our momentum strategy on both stocks with full capital utilization
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class FinalMomentumTrader:
    def __init__(self, symbol, starting_capital=10000):
        self.symbol = symbol
        self.starting_capital = starting_capital
        
        # MOMENTUM-OPTIMIZED THRESHOLDS
        self.config = {
            'trend_5d_buy_threshold': 0.015,    # 1.5% - Catches early momentum
            'trend_5d_sell_threshold': -0.025,  # -2.5% - Protects profits
            'trend_10d_buy_threshold': 0.015,   # 1.5% - Confirms momentum
            'trend_10d_sell_threshold': -0.045, # -4.5% - Strict exit
            'rsi_overbought': 85,               # 85 - Allow momentum runs
            'rsi_oversold': 30,                 # 30 - Realistic oversold
            'volatility_threshold': 0.10,       # 10% - Higher tolerance
            'volume_ratio_threshold': 1.1       # 1.1x - Easy to trigger
        }
        
        print(f"📈 MOMENTUM TRADER - {symbol}")
        print(f"💰 Starting Capital: ${starting_capital:,}")
        
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        data = data.copy()
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA10'] = data['Close'].rolling(10).mean()
        
        # RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
    
    def get_signal(self, data, idx):
        """Get trading signal for current day"""
        try:
            price = float(data['Close'].iloc[idx])
            
            # Trend calculations
            recent_5d = data['Close'].iloc[idx-5:idx]
            recent_10d = data['Close'].iloc[idx-10:idx]
            
            trend_5d = (price - float(recent_5d.mean())) / float(recent_5d.mean())
            trend_10d = (price - float(recent_10d.mean())) / float(recent_10d.mean())
            
            # Technical indicators
            rsi = float(data['RSI'].iloc[idx]) if not pd.isna(data['RSI'].iloc[idx]) else 50
            
            # Volume analysis
            recent_vol = float(data['Volume'].iloc[idx-10:idx].mean())
            current_vol = float(data['Volume'].iloc[idx])
            vol_ratio = current_vol / recent_vol if recent_vol > 0 else 1
            
            # Volatility
            volatility = float(recent_10d.std()) / float(recent_10d.mean()) if float(recent_10d.mean()) > 0 else 0
            
            # SIGNAL DETERMINATION
            signal = 'HOLD'
            
            # BUY: Strong dual-trend momentum
            if (trend_5d > self.config['trend_5d_buy_threshold'] and 
                trend_10d > self.config['trend_10d_buy_threshold'] and
                rsi < self.config['rsi_overbought'] and
                volatility < self.config['volatility_threshold']):
                signal = 'BUY'
            
            # SELL: Trend breakdown or extreme overbought
            elif (trend_5d < self.config['trend_5d_sell_threshold'] and 
                  trend_10d < self.config['trend_10d_sell_threshold']) or \
                 (rsi > self.config['rsi_overbought'] and trend_5d < 0):
                signal = 'SELL'
            
            return {
                'signal': signal,
                'price': price,
                'trend_5d': trend_5d,
                'trend_10d': trend_10d,
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': vol_ratio
            }
            
        except Exception as e:
            return {'signal': 'HOLD', 'price': price if 'price' in locals() else 0}
    
    def run_trading(self):
        """Run the momentum trading strategy"""
        print(f"\n🚀 RUNNING {self.symbol} MOMENTUM STRATEGY")
        print("=" * 50)
        
        # Download 4 months of data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=120)
        
        data = yf.download(self.symbol, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None
            
        data = self.calculate_indicators(data)
        print(f"✅ Data: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')} ({len(data)} days)")
        
        # Portfolio state
        cash = self.starting_capital
        shares = 0
        position = None
        trades = []
        
        # Trading simulation
        for i in range(15, len(data)):
            date = data.index[i]
            signal_data = self.get_signal(data, i)
            price = signal_data['price']
            signal = signal_data['signal']
            
            # Execute trades
            if signal == 'BUY' and position != 'LONG' and cash > 0:
                # Buy with full available cash
                shares_to_buy = cash / price
                amount = shares_to_buy * price
                
                shares += shares_to_buy
                cash = 0  # Use all cash
                position = 'LONG'
                
                trades.append({
                    'date': date,
                    'action': 'BUY',
                    'price': price,
                    'shares': shares_to_buy,
                    'amount': amount,
                    'signal_data': signal_data
                })
                
                print(f"📈 BUY:  {date.strftime('%Y-%m-%d')} | ${price:7.2f} | {shares_to_buy:6.2f} shares | ${amount:8,.0f}")
                print(f"         Trend 5d: {signal_data['trend_5d']:+5.1%} | Trend 10d: {signal_data['trend_10d']:+5.1%} | RSI: {signal_data['rsi']:4.0f}")
                
            elif signal == 'SELL' and position == 'LONG' and shares > 0:
                # Sell all shares
                amount = shares * price
                sold_shares = shares
                
                # Calculate trade profit
                last_buy = None
                for t in reversed(trades):
                    if t['action'] == 'BUY':
                        last_buy = t
                        break
                
                profit = amount - last_buy['amount'] if last_buy else 0
                profit_pct = (profit / last_buy['amount']) * 100 if last_buy else 0
                
                cash = amount
                shares = 0
                position = None
                
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': price,
                    'shares': sold_shares,
                    'amount': amount,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'signal_data': signal_data
                })
                
                print(f"📉 SELL: {date.strftime('%Y-%m-%d')} | ${price:7.2f} | {sold_shares:6.2f} shares | ${amount:8,.0f}")
                print(f"         Profit: ${profit:+8,.0f} ({profit_pct:+5.1f}%) | Trend 5d: {signal_data['trend_5d']:+5.1%}")
            
            # Progress updates
            if i % 25 == 0:
                portfolio_value = cash + (shares * price)
                print(f"📊 {date.strftime('%Y-%m-%d')} | ${price:7.2f} | Portfolio: ${portfolio_value:8,.0f} | {signal:4s}")
        
        # Final calculations
        final_price = float(data['Close'].iloc[-1])
        final_value = cash + (shares * final_price)
        
        start_price = float(data['Close'].iloc[15])
        buy_hold_value = self.starting_capital * (final_price / start_price)
        
        strategy_return = ((final_value - self.starting_capital) / self.starting_capital) * 100
        buy_hold_return = ((buy_hold_value - self.starting_capital) / self.starting_capital) * 100
        outperformance = strategy_return - buy_hold_return
        
        results = {
            'symbol': self.symbol,
            'final_value': final_value,
            'strategy_return': strategy_return,
            'buy_hold_value': buy_hold_value,
            'buy_hold_return': buy_hold_return,
            'outperformance': outperformance,
            'trades': trades,
            'final_cash': cash,
            'final_shares': shares,
            'price_change': ((final_price - start_price) / start_price) * 100
        }
        
        # Display results
        print(f"\n" + "="*60)
        print(f"🏆 {self.symbol} MOMENTUM TRADING RESULTS")
        print("="*60)
        print(f"💰 Strategy Value:  ${final_value:10,.2f} ({strategy_return:+6.1f}%)")
        print(f"🎯 Buy-Hold Value:  ${buy_hold_value:10,.2f} ({buy_hold_return:+6.1f}%)")
        print(f"🏆 Outperformance:  ${final_value - buy_hold_value:10,.2f} ({outperformance:+6.1f}%)")
        print(f"📊 Price Movement:  ${start_price:.2f} → ${final_price:.2f} ({results['price_change']:+.1f}%)")
        print(f"🔄 Total Trades:    {len(trades)}")
        print(f"💵 Final Cash:      ${cash:,.2f}")
        print(f"📈 Final Shares:    {shares:.2f}")
        
        if trades:
            buy_count = len([t for t in trades if t['action'] == 'BUY'])
            sell_count = len([t for t in trades if t['action'] == 'SELL'])
            
            if sell_count > 0:
                profitable = len([t for t in trades if t['action'] == 'SELL' and t.get('profit', 0) > 0])
                total_profit = sum(t.get('profit', 0) for t in trades if t['action'] == 'SELL')
                
                print(f"\n📈 TRADE PERFORMANCE:")
                print(f"   🎯 Win Rate: {profitable}/{sell_count} ({profitable/sell_count:.1%})")
                print(f"   💰 Total Trading Profit: ${total_profit:,.2f}")
        
        return results

def compare_stocks():
    """Compare NVDA vs AAPL momentum trading"""
    print("🏁 MOMENTUM STRATEGY COMPARISON - NVDA vs AAPL")
    print("=" * 70)
    
    results = {}
    
    for symbol in ['NVDA', 'AAPL']:
        trader = FinalMomentumTrader(symbol, starting_capital=10000)
        result = trader.run_trading()
        if result:
            results[symbol] = result
    
    # Comparison summary
    if len(results) == 2:
        print(f"\n" + "="*70)
        print("🏆 FINAL COMPARISON")
        print("="*70)
        
        for symbol in ['NVDA', 'AAPL']:
            r = results[symbol]
            status = "✅" if r['outperformance'] > 0 else "❌"
            print(f"{status} {symbol}: ${r['final_value']:,.0f} ({r['strategy_return']:+.1f}%) | "
                  f"vs Buy-Hold: {r['outperformance']:+.1f}% | Trades: {len(r['trades'])}")
        
        # Best performer
        best_symbol = max(results.keys(), key=lambda s: results[s]['strategy_return'])
        best = results[best_symbol]
        
        print(f"\n🥇 BEST PERFORMER: {best_symbol}")
        print(f"   💰 Turned $10,000 into ${best['final_value']:,.2f}")
        print(f"   📈 Total Return: {best['strategy_return']:+.1f}%")
        print(f"   🎯 Beat Buy-Hold by: {best['outperformance']:+.1f}%")
    
    return results

def main():
    """Run final momentum trader comparison"""
    results = compare_stocks()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"momentum_comparison_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {filename}")
    return results

if __name__ == "__main__":
    results = main()
