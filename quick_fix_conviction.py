#!/usr/bin/env python3
"""
FIXED HIGH CONVICTION TRADER
No more pandas ambiguity errors!
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json

class FixedHighConvictionTrader:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trade_history = []
        
        # Our conviction thresholds - when we go ALL IN
        self.conviction_thresholds = {
            0.50: 0.80,  # 50%+ momentum = 80% allocation
            0.30: 0.60,  # 30%+ momentum = 60% allocation  
            0.20: 0.40,  # 20%+ momentum = 40% allocation
            0.10: 0.20,  # 10%+ momentum = 20% allocation
        }
        
        # Our universe of stocks
        self.universe = [
            'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX',
            'CRM', 'UBER', 'PLTR', 'AMD', 'SNOW', 'COIN'
        ]
    
    def get_conviction_score(self, symbol, start_date, end_date):
        """Calculate conviction score with FIXED conditions"""
        try:
            data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), 
                             end=end_date.strftime('%Y-%m-%d'), progress=False)
            
            if len(data) < 60:
                return 0
                
            current_price = float(data['Close'].iloc[-1])
            
            # 30-day momentum (the main signal)
            if len(data) >= 30:
                price_30d = float(data['Close'].iloc[-30])
                momentum_30d = (current_price / price_30d - 1)
                return momentum_30d  # Simple but effective
            
            return 0
                
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return 0
    
    def get_conviction_allocation(self, momentum_score):
        """Map momentum score to allocation percentage"""
        for threshold, allocation in sorted(self.conviction_thresholds.items(), reverse=True):
            if momentum_score >= threshold:
                return allocation
        return 0
    
    def rebalance_portfolio(self, date):
        """Execute high conviction rebalancing"""
        print(f"\n🔥 HIGH CONVICTION REBALANCE - {date.strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        # Get current portfolio value
        portfolio_value = self.get_portfolio_value(date)
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        
        # Sell all current positions
        for symbol in list(self.positions.keys()):
            self.sell_position(symbol, date)
        
        print(f"\n📊 MOMENTUM ANALYSIS:")
        print("-" * 40)
        
        # Find the highest conviction plays
        conviction_plays = []
        for symbol in self.universe:
            score = self.get_conviction_score(symbol, date - timedelta(days=120), date)
            allocation = self.get_conviction_allocation(score)
            
            if allocation > 0:
                conviction_plays.append({
                    'symbol': symbol,
                    'momentum': score,
                    'allocation': allocation
                })
                print(f"🎯 {symbol}: {score:+.1%} momentum → {allocation:.0%} allocation")
        
        # Sort by conviction (highest first)
        conviction_plays.sort(key=lambda x: x['momentum'], reverse=True)
        
        print(f"\n🎯 TOP CONVICTION PLAYS:")
        print("-" * 30)
        
        total_allocation = 0
        for play in conviction_plays[:3]:  # Max 3 positions
            symbol = play['symbol']
            allocation = play['allocation']
            momentum = play['momentum']
            
            position_value = portfolio_value * allocation
            current_price = self.get_current_price(symbol, date)
            shares = int(position_value / current_price)
            
            if shares > 0:
                self.buy_position(symbol, shares, current_price, date)
                total_allocation += allocation
                print(f"💰 {symbol}: {allocation:.0%} (${position_value:,.0f}) - Momentum: {momentum:+.1%}")
        
        print(f"\nTotal Target Allocation: {total_allocation:.0%}")
        print(f"Cash Reserve: {100-total_allocation*100:.0%}%")
        
        print(f"\n💰 EXECUTING HIGH CONVICTION TRADES:")
        print("-" * 40)
        portfolio_value = self.get_portfolio_value(date)
        print(f"📈 Portfolio Value: ${portfolio_value:,.2f} (Cash: ${self.cash:,.2f})")
    
    def get_current_price(self, symbol, date):
        """Get stock price on specific date"""
        try:
            # Try the exact date first
            data = yf.download(symbol, start=date.strftime('%Y-%m-%d'), 
                             end=(date + timedelta(days=1)).strftime('%Y-%m-%d'), progress=False)
            if len(data) > 0:
                return float(data['Close'].iloc[-1])
            
            # If weekend, try a few days later
            for i in range(1, 5):
                try_date = date + timedelta(days=i)
                data = yf.download(symbol, start=try_date.strftime('%Y-%m-%d'), 
                                 end=(try_date + timedelta(days=1)).strftime('%Y-%m-%d'), progress=False)
                if len(data) > 0:
                    return float(data['Close'].iloc[-1])
            
            return 0
        except:
            return 0
    
    def buy_position(self, symbol, shares, price, date):
        """Buy a position"""
        cost = shares * price
        if cost <= self.cash:
            self.positions[symbol] = {'shares': shares, 'price': price}
            self.cash -= cost
            self.trade_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'action': 'BUY',
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'value': cost
            })
    
    def sell_position(self, symbol, date):
        """Sell a position"""
        if symbol in self.positions:
            shares = self.positions[symbol]['shares']
            current_price = self.get_current_price(symbol, date)
            
            if current_price > 0:
                proceeds = shares * current_price
                self.cash += proceeds
                
                self.trade_history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'action': 'SELL',
                    'symbol': symbol,
                    'shares': shares,
                    'price': current_price,
                    'value': proceeds
                })
                
                del self.positions[symbol]
    
    def get_portfolio_value(self, date):
        """Calculate total portfolio value"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol, date)
            if current_price > 0:
                total_value += position['shares'] * current_price
        
        return total_value
    
    def run_backtest(self):
        """Run the 3-month HIGH CONVICTION backtest"""
        print("🚀 STARTING HIGH CONVICTION MOMENTUM STRATEGY")
        print("=" * 60)
        print("📈 We FOLLOW our conviction signals!")
        print("🎯 50%+ momentum = 80% allocation!")
        print("💰 Starting Capital: $100,000")
        print()
        
        # Our 3-month test period
        start_date = datetime(2024, 5, 13)
        end_date = datetime(2024, 8, 13)
        
        # Monthly rebalancing dates
        rebalance_dates = [
            datetime(2024, 5, 13),  # Month 1
            datetime(2024, 6, 13),  # Month 2  
            datetime(2024, 7, 15),  # Month 3 (avoid weekend)
        ]
        
        for i, date in enumerate(rebalance_dates, 1):
            print(f"🗓️  MONTH {i} CONVICTION CHECK - {date.strftime('%Y-%m-%d')}")
            self.rebalance_portfolio(date)
        
        # Final results
        final_value = self.get_portfolio_value(end_date)
        
        print(f"\n🏆 FINAL HIGH CONVICTION RESULTS - {end_date.strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        print(f"\n🎯 FINAL POSITIONS:")
        print("-" * 25)
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol, end_date)
            position_value = position['shares'] * current_price
            profit = position_value - (position['shares'] * position['price'])
            print(f"📊 {symbol}: {position['shares']} shares @ ${current_price:.2f} = ${position_value:,.2f} (P/L: ${profit:+,.2f})")
        
        total_return = (final_value / self.initial_capital - 1)
        profit = final_value - self.initial_capital
        
        print(f"\n💰 HIGH CONVICTION RESULTS:")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:+.2%}")
        print(f"Profit/Loss: ${profit:+,.2f}")
        print(f"Cash Remaining: ${self.cash:,.2f}")
        
        # Compare to NVDA buy and hold
        nvda_data = yf.download('NVDA', start='2024-05-13', end='2024-08-13', progress=False)
        nvda_start = float(nvda_data['Close'].iloc[0])
        nvda_end = float(nvda_data['Close'].iloc[-1])
        nvda_return = (nvda_end / nvda_start - 1)
        nvda_profit = 100000 * nvda_return
        
        print(f"\n📊 COMPARISON:")
        print(f"Our High Conviction: {total_return:+.2%} (${profit:+,.2f})")
        print(f"NVDA Buy & Hold: {nvda_return:+.2%} (${nvda_profit:+,.2f})")
        print(f"😤 We missed NVDA by {nvda_return - total_return:.2%}")
        
        # Save results
        results = {
            'strategy': 'HIGH CONVICTION MOMENTUM',
            'period': '2024-05-13 to 2024-08-13',
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'profit_loss': profit,
            'nvda_return': nvda_return,
            'underperformance': nvda_return - total_return,
            'final_positions': self.positions,
            'cash_remaining': self.cash,
            'trade_history': self.trade_history
        }
        
        with open('fixed_conviction_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to 'fixed_conviction_results.json'")
        print(f"\n🔥 THIS IS HOW YOU FOLLOW CONVICTION SIGNALS!")

if __name__ == "__main__":
    trader = FixedHighConvictionTrader()
    trader.run_backtest()
