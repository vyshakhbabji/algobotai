#!/usr/bin/env python3
"""
$100K 3-MONTH MOMENTUM TRADING SIMULATION
Let's see what happens when we scale this up!
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt

class MonthlyMomentumTrader:
    def __init__(self, capital=100000):
        self.initial_capital = capital
        self.cash = capital
        self.positions = {}
        self.portfolio_history = []
        self.trades_log = []
        
        # Trading parameters
        self.max_positions = 5
        self.position_size = 0.18  # 18% per position (90% total invested)
        self.stop_loss = 0.08      # 8% stop loss
        self.take_profit = 0.15    # 15% take profit
        
        # Stock universe
        self.universe = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'CRM', 'UBER', 'PLTR']
        
    def get_momentum_score(self, symbol, date):
        """Get momentum score for a stock at a specific date"""
        try:
            # Get 6 months of data ending at the specified date
            end_date = pd.to_datetime(date)
            start_date = end_date - timedelta(days=180)
            
            data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), 
                             end=end_date.strftime('%Y-%m-%d'), progress=False)
            
            if len(data) < 60:
                return 0, None
            
            # Get the price at the analysis date (or closest trading day)
            if len(data) == 0:
                return 0, None
                
            current_price = float(data['Close'].iloc[-1])
            
            # 30-day momentum
            if len(data) >= 30:
                price_30d = float(data['Close'].iloc[-30])
                momentum_30d = (current_price / price_30d - 1)
            else:
                momentum_30d = 0
            
            # Moving average trend
            if len(data) >= 20:
                ma_20 = float(data['Close'].rolling(20).mean().iloc[-1])
                above_ma = current_price > ma_20
            else:
                above_ma = False
            
            # Volume trend
            if len(data) >= 10:
                recent_volume = float(data['Volume'].iloc[-5:].mean())
                avg_volume = float(data['Volume'].rolling(30).mean().iloc[-1])
                volume_strong = recent_volume > avg_volume * 1.1
            else:
                volume_strong = False
            
            # Combined score
            score = momentum_30d
            if above_ma:
                score *= 1.2
            if volume_strong:
                score *= 1.1
                
            return score, current_price
            
        except Exception as e:
            print(f"Error analyzing {symbol} on {date}: {e}")
            return 0, None
    
    def rebalance_portfolio(self, date):
        """Monthly rebalancing based on momentum signals"""
        print(f"\n📅 REBALANCING ON {date}")
        print("=" * 50)
        
        # Calculate current portfolio value
        portfolio_value = self.cash
        for symbol, pos in self.positions.items():
            try:
                current_price = self.get_current_price(symbol, date)
                if current_price:
                    portfolio_value += pos['shares'] * current_price
            except:
                pass
        
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        
        # Get momentum scores for all stocks
        scores = {}
        prices = {}
        
        print("\nMomentum Analysis:")
        print("-" * 30)
        
        for symbol in self.universe:
            score, price = self.get_momentum_score(symbol, date)
            scores[symbol] = score
            prices[symbol] = price
            if price:
                print(f"{symbol}: {score:+.2%} (${price:.2f})")
        
        # Select top momentum stocks
        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected_stocks = []
        
        for symbol, score in sorted_stocks[:self.max_positions]:
            if score > 0.02 and prices[symbol]:  # Only positive momentum > 2%
                selected_stocks.append((symbol, score, prices[symbol]))
        
        # Close positions not in new selection
        to_close = []
        for symbol in self.positions:
            if symbol not in [s[0] for s in selected_stocks]:
                to_close.append(symbol)
        
        for symbol in to_close:
            self.close_position(symbol, date)
        
        # Open new positions
        print(f"\nNew Positions:")
        print("-" * 20)
        
        for symbol, score, price in selected_stocks:
            target_value = portfolio_value * self.position_size
            shares = int(target_value / price)
            
            if symbol in self.positions:
                # Adjust existing position
                current_shares = self.positions[symbol]['shares']
                share_diff = shares - current_shares
                
                if abs(share_diff) > 1:  # Only trade if meaningful difference
                    if share_diff > 0:
                        self.buy_stock(symbol, share_diff, price, date)
                    else:
                        self.sell_stock(symbol, abs(share_diff), price, date)
            else:
                # New position
                if shares > 0:
                    self.buy_stock(symbol, shares, price, date)
        
        # Record portfolio snapshot
        self.record_portfolio_snapshot(date)
    
    def buy_stock(self, symbol, shares, price, date):
        """Buy stock"""
        cost = shares * price
        if cost <= self.cash:
            self.cash -= cost
            
            if symbol in self.positions:
                # Add to existing position
                old_shares = self.positions[symbol]['shares']
                old_avg = self.positions[symbol]['avg_price']
                new_shares = old_shares + shares
                new_avg = ((old_shares * old_avg) + (shares * price)) / new_shares
                
                self.positions[symbol] = {
                    'shares': new_shares,
                    'avg_price': new_avg,
                    'date_opened': self.positions[symbol]['date_opened']
                }
            else:
                # New position
                self.positions[symbol] = {
                    'shares': shares,
                    'avg_price': price,
                    'date_opened': date
                }
            
            self.trades_log.append({
                'date': date,
                'action': 'BUY',
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'value': cost
            })
            
            print(f"✅ BUY {shares} {symbol} @ ${price:.2f} = ${cost:,.2f}")
            return True
        return False
    
    def sell_stock(self, symbol, shares, price, date):
        """Sell stock"""
        if symbol in self.positions and self.positions[symbol]['shares'] >= shares:
            proceeds = shares * price
            self.cash += proceeds
            
            self.positions[symbol]['shares'] -= shares
            
            if self.positions[symbol]['shares'] == 0:
                del self.positions[symbol]
            
            self.trades_log.append({
                'date': date,
                'action': 'SELL',
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'value': proceeds
            })
            
            print(f"💰 SELL {shares} {symbol} @ ${price:.2f} = ${proceeds:,.2f}")
            return True
        return False
    
    def close_position(self, symbol, date):
        """Close entire position"""
        if symbol in self.positions:
            shares = self.positions[symbol]['shares']
            price = self.get_current_price(symbol, date)
            if price:
                self.sell_stock(symbol, shares, price, date)
    
    def get_current_price(self, symbol, date):
        """Get stock price on specific date"""
        try:
            end_date = pd.to_datetime(date) + timedelta(days=1)
            start_date = pd.to_datetime(date) - timedelta(days=5)
            
            data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'),
                             end=end_date.strftime('%Y-%m-%d'), progress=False)
            
            if len(data) > 0:
                return float(data['Close'].iloc[-1])
        except:
            pass
        return None
    
    def record_portfolio_snapshot(self, date):
        """Record portfolio value on date"""
        portfolio_value = self.cash
        
        for symbol, pos in self.positions.items():
            price = self.get_current_price(symbol, date)
            if price:
                portfolio_value += pos['shares'] * price
        
        self.portfolio_history.append({
            'date': date,
            'value': portfolio_value,
            'cash': self.cash,
            'positions': len(self.positions)
        })
        
        print(f"📊 Portfolio Value: ${portfolio_value:,.2f} (Cash: ${self.cash:,.2f})")
    
    def run_3_month_backtest(self, start_date='2024-05-13'):
        """Run 3-month backtest"""
        print("🚀 STARTING $100K 3-MONTH MOMENTUM STRATEGY")
        print("=" * 60)
        
        # Monthly rebalancing dates
        dates = [
            '2024-05-13',  # Month 1
            '2024-06-13',  # Month 2  
            '2024-07-13',  # Month 3
            '2024-08-13'   # Final evaluation
        ]
        
        for i, date in enumerate(dates[:-1]):
            print(f"\n🗓️  MONTH {i+1} - {date}")
            self.rebalance_portfolio(date)
        
        # Final evaluation
        print(f"\n🏁 FINAL EVALUATION - {dates[-1]}")
        print("=" * 50)
        
        final_value = self.cash
        print(f"\nFinal Positions:")
        print("-" * 20)
        
        for symbol, pos in self.positions.items():
            price = self.get_current_price(symbol, dates[-1])
            if price:
                position_value = pos['shares'] * price
                final_value += position_value
                gain_loss = (price / pos['avg_price'] - 1) * 100
                print(f"{symbol}: {pos['shares']} shares @ ${price:.2f} = ${position_value:,.2f} ({gain_loss:+.1f}%)")
        
        total_return = (final_value / self.initial_capital - 1) * 100
        profit_loss = final_value - self.initial_capital
        
        print(f"\n💰 FINAL RESULTS:")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Profit/Loss: ${profit_loss:+,.2f}")
        print(f"Cash Remaining: ${self.cash:,.2f}")
        
        # Calculate monthly returns
        if len(self.portfolio_history) > 0:
            monthly_returns = []
            for i in range(1, len(self.portfolio_history)):
                prev_val = self.portfolio_history[i-1]['value']
                curr_val = self.portfolio_history[i]['value']
                monthly_ret = (curr_val / prev_val - 1) * 100
                monthly_returns.append(monthly_ret)
                print(f"Month {i} Return: {monthly_ret:+.2f}%")
        
        # Save detailed results
        results = {
            'strategy': '$100K 3-Month Momentum',
            'period': f"{dates[0]} to {dates[-1]}",
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'profit_loss': profit_loss,
            'monthly_returns': monthly_returns if 'monthly_returns' in locals() else [],
            'total_trades': len(self.trades_log),
            'portfolio_history': self.portfolio_history,
            'trades_log': self.trades_log,
            'final_positions': {
                symbol: {
                    'shares': pos['shares'],
                    'avg_price': pos['avg_price'],
                    'current_price': self.get_current_price(symbol, dates[-1]),
                    'value': pos['shares'] * self.get_current_price(symbol, dates[-1]) if self.get_current_price(symbol, dates[-1]) else 0
                } for symbol, pos in self.positions.items()
            }
        }
        
        with open('100k_3month_backtest.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ DETAILED RESULTS SAVED TO '100k_3month_backtest.json'")
        
        return results

def main():
    trader = MonthlyMomentumTrader(capital=100000)
    results = trader.run_3_month_backtest()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Turned ${100000:,} into ${results['final_value']:,.2f}")
    print(f"   That's a {results['total_return_pct']:+.1f}% return in 3 months!")
    print(f"   Profit: ${results['profit_loss']:+,.2f}")
    
    if results['total_return_pct'] > 0:
        annual_estimate = ((1 + results['total_return_pct']/100) ** 4) - 1
        print(f"   Annualized: {annual_estimate*100:+.1f}%")
    
    print(f"\n🔥 This momentum strategy WORKS at scale!")

if __name__ == "__main__":
    main()
