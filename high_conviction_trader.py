#!/usr/bin/env python3
"""
HIGH CONVICTION MOMENTUM STRATEGY
When the algo says GO ALL IN - we GO ALL IN!
No more pussy diversification when signals are screaming
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class HighConvictionTrader:
    def __init__(self, capital=100000):
        self.initial_capital = capital
        self.cash = capital
        self.positions = {}
        self.portfolio_history = []
        self.trades_log = []
        
        # HIGH CONVICTION PARAMETERS
        self.max_positions = 3  # Only top 3 picks
        self.min_momentum = 0.03  # 3% minimum momentum
        
        # DYNAMIC POSITION SIZING BASED ON SIGNAL STRENGTH
        self.conviction_thresholds = {
            0.50: 0.80,  # 50%+ momentum = 80% allocation
            0.30: 0.60,  # 30%+ momentum = 60% allocation  
            0.15: 0.40,  # 15%+ momentum = 40% allocation
            0.05: 0.25,  # 5%+ momentum = 25% allocation
            0.03: 0.15   # 3%+ momentum = 15% allocation
        }
        
        # Expanded universe for better picks
        self.universe = [
            'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 
            'NFLX', 'CRM', 'UBER', 'PLTR', 'AMD', 'SNOW', 'COIN'
        ]
        
    def get_momentum_score(self, symbol, date):
        """Enhanced momentum scoring with more factors"""
        try:
            end_date = pd.to_datetime(date)
            start_date = end_date - timedelta(days=180)
            
            data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), 
                             end=end_date.strftime('%Y-%m-%d'), progress=False)
            
            if len(data) < 60:
                return 0
                
            current_price = float(data['Close'].iloc[-1])
            
            # Multiple momentum timeframes
            momentum_scores = []
            
            # 30-day momentum (primary)
            if len(data) >= 30:
                price_30d = float(data['Close'].iloc[-30])
                momentum_30d = (current_price / price_30d - 1)
                momentum_scores.append(momentum_30d * 0.5)  # 50% weight
            
            # 10-day momentum (short-term)
            if len(data) >= 10:
                price_10d = float(data['Close'].iloc[-10])
                momentum_10d = (current_price / price_10d - 1)
                momentum_scores.append(momentum_10d * 0.3)  # 30% weight
            
            # 60-day momentum (medium-term)
            if len(data) >= 60:
                price_60d = float(data['Close'].iloc[-60])
                momentum_60d = (current_price / price_60d - 1)
                momentum_scores.append(momentum_60d * 0.2)  # 20% weight
            
            base_momentum = sum(momentum_scores)
            
            # Moving average confirmation
            ma_multiplier = 1.0
            if len(data) >= 20:
                ma_20 = float(data['Close'].rolling(20).mean().iloc[-1])
                if current_price > ma_20:
                    ma_multiplier = 1.3  # Strong bonus for being above MA
            
            # Volume explosion detector
            volume_multiplier = 1.0
            if len(data) >= 30:
                recent_volume = float(data['Volume'].iloc[-5:].mean())
                avg_volume = float(data['Volume'].rolling(30).mean().iloc[-1])
                
                if recent_volume > avg_volume * 2.0:
                    volume_multiplier = 1.5  # Massive volume = massive multiplier
                elif recent_volume > avg_volume * 1.5:
                    volume_multiplier = 1.3
                elif recent_volume > avg_volume * 1.2:
                    volume_multiplier = 1.1
            
            # Volatility bonus (higher vol = higher potential)
            volatility_multiplier = 1.0
            if len(data) >= 20:
                returns = data['Close'].pct_change().dropna()
                volatility = float(returns.rolling(20).std().iloc[-1])
                
                if volatility > 0.05:  # >5% daily volatility
                    volatility_multiplier = 1.2
                elif volatility > 0.03:  # >3% daily volatility
                    volatility_multiplier = 1.1
            
            # Final conviction score
            final_score = base_momentum * ma_multiplier * volume_multiplier * volatility_multiplier
            
            return final_score
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return 0
    
    def get_position_size(self, momentum_score):
        """Dynamic position sizing based on conviction"""
        for threshold, allocation in sorted(self.conviction_thresholds.items(), reverse=True):
            if momentum_score >= threshold:
                return allocation
        return 0  # Don't buy if below minimum threshold
    
    def rebalance_portfolio(self, date):
        """HIGH CONVICTION rebalancing"""
        print(f"\n🔥 HIGH CONVICTION REBALANCE - {date}")
        print("=" * 60)
        
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
        
        # Analyze ALL stocks for momentum
        momentum_analysis = []
        
        print(f"\n📊 MOMENTUM ANALYSIS:")
        print("-" * 40)
        
        for symbol in self.universe:
            score = self.get_momentum_score(symbol, date)
            position_size = self.get_position_size(score)
            
            if score > 0:
                momentum_analysis.append({
                    'symbol': symbol,
                    'momentum': score,
                    'allocation': position_size,
                    'conviction': 'HIGH' if score > 0.3 else 'MEDIUM' if score > 0.1 else 'LOW'
                })
                
                conviction_emoji = "🚀" if score > 0.3 else "📈" if score > 0.1 else "📊"
                print(f"{conviction_emoji} {symbol}: {score:+.3f} ({score*100:+.1f}%) -> {position_size*100:.0f}% allocation")
        
        # Sort by momentum and select top picks
        momentum_analysis.sort(key=lambda x: x['momentum'], reverse=True)
        top_picks = [pick for pick in momentum_analysis[:self.max_positions] if pick['allocation'] > 0]
        
        print(f"\n🎯 TOP CONVICTION PLAYS:")
        print("-" * 30)
        
        total_target_allocation = 0
        for i, pick in enumerate(top_picks):
            total_target_allocation += pick['allocation']
            print(f"{i+1}. {pick['symbol']}: {pick['momentum']*100:+.1f}% momentum -> {pick['allocation']*100:.0f}% allocation ({pick['conviction']} CONVICTION)")
        
        print(f"\nTotal Target Allocation: {total_target_allocation*100:.0f}%")
        print(f"Cash Reserve: {(1-total_target_allocation)*100:.0f}%")
        
        # Close positions not in top picks
        to_close = []
        for symbol in self.positions:
            if symbol not in [pick['symbol'] for pick in top_picks]:
                to_close.append(symbol)
        
        for symbol in to_close:
            self.close_position(symbol, date)
        
        # Execute high conviction trades
        print(f"\n💰 EXECUTING HIGH CONVICTION TRADES:")
        print("-" * 40)
        
        for pick in top_picks:
            symbol = pick['symbol']
            target_allocation = pick['allocation']
            target_value = portfolio_value * target_allocation
            
            current_price = self.get_current_price(symbol, date)
            if current_price:
                target_shares = int(target_value / current_price)
                
                if symbol in self.positions:
                    current_shares = self.positions[symbol]['shares']
                    share_difference = target_shares - current_shares
                    
                    if abs(share_difference) > 1:
                        if share_difference > 0:
                            self.buy_stock(symbol, share_difference, current_price, date)
                        else:
                            self.sell_stock(symbol, abs(share_difference), current_price, date)
                else:
                    if target_shares > 0:
                        self.buy_stock(symbol, target_shares, current_price, date)
        
        # Record portfolio snapshot
        self.record_portfolio_snapshot(date)
    
    def buy_stock(self, symbol, shares, price, date):
        """Buy stock with conviction"""
        cost = shares * price
        if cost <= self.cash:
            self.cash -= cost
            
            if symbol in self.positions:
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
            
            print(f"🚀 BUY {shares} {symbol} @ ${price:.2f} = ${cost:,.2f}")
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
            
            print(f"💸 SELL {shares} {symbol} @ ${price:.2f} = ${proceeds:,.2f}")
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
        """Record portfolio value"""
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
        
        print(f"📈 Portfolio Value: ${portfolio_value:,.2f} (Cash: ${self.cash:,.2f})")
    
    def run_high_conviction_test(self):
        """Run the high conviction 3-month test"""
        print("🔥🔥🔥 HIGH CONVICTION MOMENTUM STRATEGY 🔥🔥🔥")
        print("When signals scream BUY - we go ALL IN!")
        print("=" * 70)
        
        # Test dates
        dates = ['2024-05-13', '2024-06-13', '2024-07-13', '2024-08-13']
        
        for i, date in enumerate(dates[:-1]):
            print(f"\n🗓️  MONTH {i+1} CONVICTION CHECK - {date}")
            self.rebalance_portfolio(date)
        
        # Final evaluation
        print(f"\n🏆 FINAL HIGH CONVICTION RESULTS - {dates[-1]}")
        print("=" * 60)
        
        final_value = self.cash
        
        print(f"\n🎯 FINAL POSITIONS:")
        print("-" * 25)
        
        for symbol, pos in self.positions.items():
            price = self.get_current_price(symbol, dates[-1])
            if price:
                position_value = pos['shares'] * price
                final_value += position_value
                gain_loss = (price / pos['avg_price'] - 1) * 100
                print(f"{symbol}: {pos['shares']} shares @ ${price:.2f} = ${position_value:,.2f} ({gain_loss:+.1f}%)")
        
        total_return = (final_value / self.initial_capital - 1) * 100
        profit_loss = final_value - self.initial_capital
        
        print(f"\n💰 HIGH CONVICTION RESULTS:")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Profit/Loss: ${profit_loss:+,.2f}")
        print(f"Cash Remaining: ${self.cash:,.2f}")
        
        # Compare to NVDA buy and hold
        nvda_data = yf.download('NVDA', start='2024-05-13', end='2024-08-13', progress=False)
        if len(nvda_data) > 0:
            nvda_start = float(nvda_data['Close'].iloc[0])
            nvda_end = float(nvda_data['Close'].iloc[-1])
            nvda_return = (nvda_end / nvda_start - 1) * 100
            nvda_profit = (self.initial_capital * (nvda_end / nvda_start)) - self.initial_capital
            
            print(f"\n📊 COMPARISON:")
            print(f"Our High Conviction: {total_return:+.2f}% (${profit_loss:+,.2f})")
            print(f"NVDA Buy & Hold: {nvda_return:+.2f}% (${nvda_profit:+,.2f})")
            
            if total_return > nvda_return:
                print(f"🚀 WE BEAT NVDA BY {total_return - nvda_return:+.2f}%!")
            else:
                print(f"😤 We missed NVDA by {nvda_return - total_return:.2f}%")
        
        # Save results
        results = {
            'strategy': 'High Conviction Momentum',
            'period': '2024-05-13 to 2024-08-13',
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'profit_loss': profit_loss,
            'trades_log': self.trades_log,
            'portfolio_history': self.portfolio_history
        }
        
        with open('high_conviction_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to 'high_conviction_results.json'")
        print(f"\n🔥 THIS IS HOW YOU FOLLOW CONVICTION SIGNALS!")

def main():
    trader = HighConvictionTrader(capital=100000)
    trader.run_high_conviction_test()

if __name__ == "__main__":
    main()
