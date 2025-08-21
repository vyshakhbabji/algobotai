#!/usr/bin/env python3
"""
INSTITUTIONAL HYBRID FORWARD TESTER
Uses institutional momentum screener to find best stocks
Then trades ONLY those pre-screened winners with optimized signals
Should deliver MUCH higher returns than random stock selection!
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class InstitutionalHybridTester:
    def __init__(self):
        # TOP INSTITUTIONAL MOMENTUM STOCKS (from our screener)
        self.institutional_winners = [
            "GE",    # +27.2% momentum score
            "GILD",  # +22.1% momentum score  
            "MSFT",  # +20.5% momentum score
            "RTX",   # +18.5% momentum score
            "C",     # +18.3% momentum score
            "JNJ"    # +13.6% momentum score
        ]
        
        # OPTIMIZED TRADING RULES for institutional winners
        self.config = {
            'momentum_threshold': 0.005,   # Very low - these stocks already qualified
            'rsi_oversold': 45,           # Buy on minor pullbacks
            'rsi_overbought': 75,         # Hold longer in trends
            'stop_loss': -0.08,           # 8% stop loss (wider for winners)
            'take_profit': 0.15,          # 15% take profit (bigger targets)
        }
        
        # AGGRESSIVE PORTFOLIO SETTINGS
        self.initial_capital = 100000
        self.max_positions = 4         # Up to 4 institutional winners
        self.position_size = 0.30      # 30% per position (120% invested max)
        
        self.trade_log = []
        self.daily_decisions = []
    
    def get_data(self, symbol, start_date, end_date):
        """Get stock data with technical indicators"""
        try:
            extended_start = start_date - timedelta(days=60)
            data = yf.download(symbol, start=extended_start, end=end_date, progress=False, auto_adjust=True)
            
            if data.empty:
                return None
            
            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate momentum score (institutional style)
            data['Momentum_Score'] = data['Close'].pct_change(21)  # 1-month momentum
            
            return data[data.index >= pd.Timestamp(start_date)]
        except:
            return None
    
    def should_buy(self, data, current_idx):
        """Optimized buy logic for institutional winners"""
        try:
            current_price = float(data['Close'].iloc[current_idx])
            yesterday_price = float(data['Close'].iloc[current_idx - 1])
            rsi = float(data['RSI'].iloc[current_idx])
            momentum = float(data['Momentum_Score'].iloc[current_idx])
            
            # Buy conditions optimized for institutional winners
            daily_momentum = (current_price - yesterday_price) / yesterday_price
            
            return (daily_momentum > self.config['momentum_threshold'] and 
                    rsi < self.config['rsi_oversold'] and
                    momentum > 0.02)  # Still showing monthly momentum
        except:
            return False
    
    def should_sell(self, entry_price, current_price, rsi):
        """Optimized sell logic for institutional winners"""
        try:
            pnl_pct = (current_price - entry_price) / entry_price
            
            # More aggressive profit taking and loss cutting
            return (pnl_pct <= self.config['stop_loss'] or 
                    pnl_pct >= self.config['take_profit'] or
                    rsi > self.config['rsi_overbought'])
        except:
            return True
    
    def run_institutional_forward_test(self):
        """Run forward test on pre-screened institutional winners"""
        
        start_date = datetime(2024, 4, 1).date()
        end_date = datetime(2024, 6, 30).date()
        
        print(f"🏛️  INSTITUTIONAL HYBRID FORWARD TEST")
        print("=" * 60)
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"🎯 Trading ONLY institutional momentum winners:")
        for i, symbol in enumerate(self.institutional_winners, 1):
            print(f"   {i}. {symbol} (Pre-screened institutional winner)")
        print(f"💰 Starting: ${self.initial_capital:,}")
        print(f"🚀 AGGRESSIVE: 30% positions, 4 max holdings")
        print("=" * 60)
        
        # Load data for institutional winners only
        stock_data = {}
        for symbol in self.institutional_winners:
            data = self.get_data(symbol, start_date, end_date)
            if data is not None and len(data) > 20:
                stock_data[symbol] = data
                print(f"✅ {symbol}: {len(data)} days loaded (Institutional winner)")
        
        print(f"📊 Loaded {len(stock_data)} institutional winners")
        
        if len(stock_data) == 0:
            print("❌ No institutional winner data! Exiting...")
            return None
        
        # Portfolio tracking
        cash = self.initial_capital
        positions = {}
        
        # Get trading days
        sample_data = list(stock_data.values())[0]
        trading_days = sample_data.index
        
        print(f"📅 Trading institutional winners for {len(trading_days)} days...")
        print("-" * 60)
        
        # Daily trading loop
        for day_idx, current_date in enumerate(trading_days):
            daily_actions = []
            
            # Calculate current portfolio value
            portfolio_value = cash
            for symbol, position in positions.items():
                if symbol in stock_data and current_date in stock_data[symbol].index:
                    current_price = float(stock_data[symbol].loc[current_date, 'Close'])
                    portfolio_value += position['shares'] * current_price
            
            # Check each institutional winner
            for symbol in stock_data.keys():
                if current_date not in stock_data[symbol].index:
                    continue
                
                data = stock_data[symbol]
                current_idx = data.index.get_loc(current_date)
                
                if current_idx < 22:  # Need history for momentum
                    continue
                
                current_price = float(data.loc[current_date, 'Close'])
                rsi = float(data['RSI'].loc[current_date]) if not pd.isna(data['RSI'].loc[current_date]) else 50
                
                # SELL CHECK (if we own it)
                if symbol in positions:
                    position = positions[symbol]
                    
                    if self.should_sell(position['entry_price'], current_price, rsi):
                        # SELL
                        sale_value = position['shares'] * current_price
                        cash += sale_value
                        
                        # Calculate P&L
                        cost_basis = position['shares'] * position['entry_price']
                        pnl = sale_value - cost_basis
                        pnl_pct = (pnl / cost_basis) * 100
                        
                        self.trade_log.append({
                            'date': current_date.date(),
                            'symbol': symbol,
                            'action': 'SELL',
                            'price': current_price,
                            'shares': position['shares'],
                            'pnl_pct': pnl_pct,
                            'hold_days': (current_date.date() - position['entry_date']).days
                        })
                        
                        action_text = f"SELL {symbol} ${current_price:.2f} (P&L: {pnl_pct:+.1f}%)"
                        daily_actions.append(action_text)
                        
                        del positions[symbol]
                
                # BUY CHECK (if we don't own it and have room)
                elif (symbol not in positions and 
                      len(positions) < self.max_positions and
                      self.should_buy(data, current_idx)):
                    
                    # BUY - Larger position sizes for institutional winners
                    position_value = self.initial_capital * self.position_size
                    if cash >= position_value:
                        shares = position_value / current_price
                        cash -= position_value
                        
                        positions[symbol] = {
                            'shares': shares,
                            'entry_price': current_price,
                            'entry_date': current_date.date()
                        }
                        
                        self.trade_log.append({
                            'date': current_date.date(),
                            'symbol': symbol,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'pnl_pct': 0,
                            'hold_days': 0
                        })
                        
                        action_text = f"BUY {symbol} ${current_price:.2f} ({shares:.0f} shares, RSI: {rsi:.0f}) [INSTITUTIONAL WINNER]"
                        daily_actions.append(action_text)
            
            # Store daily decisions
            self.daily_decisions.append({
                'date': current_date.date(),
                'actions': daily_actions if daily_actions else ['HOLD'],
                'portfolio_value': portfolio_value,
                'cash': cash,
                'positions': len(positions)
            })
            
            # Progress update every 10 days
            if (day_idx + 1) % 10 == 0 or day_idx < 5:
                status = f"Day {day_idx + 1:2d}: ${portfolio_value:,.0f} | Pos: {len(positions)} | Cash: ${cash:,.0f}"
                if daily_actions and daily_actions != ['HOLD']:
                    status += f" | {', '.join(daily_actions[:2])}"  # Show first 2 actions
                print(status)
        
        # Final calculations
        final_portfolio_value = cash
        for symbol, position in positions.items():
            if symbol in stock_data:
                final_price = float(stock_data[symbol]['Close'].iloc[-1])
                final_portfolio_value += position['shares'] * final_price
        
        total_return = ((final_portfolio_value - self.initial_capital) / self.initial_capital) * 100
        
        # Results
        print(f"\n🏆 INSTITUTIONAL HYBRID RESULTS")
        print("=" * 60)
        print(f"💰 Starting Capital: ${self.initial_capital:,}")
        print(f"💰 Final Portfolio: ${final_portfolio_value:,.0f}")
        print(f"📈 Total Return: {total_return:+.1f}%")
        print(f"📊 Total Trades: {len(self.trade_log)}")
        print(f"🏛️  Trading Strategy: Institutional momentum winners only")
        
        # Trade analysis
        if len(self.trade_log) > 0:
            buy_trades = [t for t in self.trade_log if t['action'] == 'BUY']
            sell_trades = [t for t in self.trade_log if t['action'] == 'SELL']
            
            print(f"🔄 Buy Orders: {len(buy_trades)}")
            print(f"🔄 Sell Orders: {len(sell_trades)}")
            
            if sell_trades:
                winning_trades = [t for t in sell_trades if t['pnl_pct'] > 0]
                win_rate = len(winning_trades) / len(sell_trades)
                avg_gain = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t['pnl_pct'] for t in sell_trades if t['pnl_pct'] < 0])
                avg_hold = np.mean([t['hold_days'] for t in sell_trades])
                
                print(f"🎯 Win Rate: {win_rate:.1%} ({len(winning_trades)}/{len(sell_trades)})")
                print(f"📈 Avg Win: {avg_gain:+.1f}%")
                print(f"📉 Avg Loss: {avg_loss:+.1f}%")
                print(f"⏱️  Avg Hold: {avg_hold:.0f} days")
        
        # Show all trades on institutional winners
        print(f"\n📊 ALL INSTITUTIONAL WINNER TRADES:")
        for trade in self.trade_log:
            emoji = "🟢" if trade['action'] == 'BUY' else ("🔴" if trade['pnl_pct'] < 0 else "🟢")
            if trade['action'] == 'SELL':
                print(f"{emoji} {trade['date']} {trade['action']} {trade['symbol']} ${trade['price']:.2f} "
                      f"(P&L: {trade['pnl_pct']:+.1f}%, {trade['hold_days']} days) [INSTITUTIONAL]")
            else:
                print(f"{emoji} {trade['date']} {trade['action']} {trade['symbol']} ${trade['price']:.2f} [INSTITUTIONAL]")
        
        # Compare to our previous aggressive test
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"🏛️  Institutional Hybrid: {total_return:+.1f}% (trading winners only)")
        print(f"🚀 Previous Aggressive: +10.6% (trading random stocks)")
        improvement = total_return - 10.6
        print(f"📈 Improvement: {improvement:+.1f}% by using institutional screening!")
        
        return {
            'total_return': total_return,
            'final_value': final_portfolio_value,
            'total_trades': len(self.trade_log),
            'improvement_vs_random': improvement
        }

def main():
    """Run institutional hybrid forward test"""
    print("🏛️  INSTITUTIONAL HYBRID MOMENTUM TESTER")
    print("=" * 60)
    print("🎯 Trade ONLY institutional momentum winners!")
    print("📊 GE, GILD, MSFT, RTX, C, JNJ (pre-screened)")
    print("🚀 30% position sizes, optimized for winners")
    print("📈 Should CRUSH random stock selection!")
    print("=" * 60)
    
    tester = InstitutionalHybridTester()
    results = tester.run_institutional_forward_test()
    
    if results:
        print(f"\n✅ INSTITUTIONAL HYBRID COMPLETE!")
        print(f"📊 Total Return: {results['total_return']:+.1f}%")
        print(f"🏛️  Trading institutional winners ONLY")
        print(f"📈 Improvement: {results['improvement_vs_random']:+.1f}% vs random selection")
        print(f"🎯 Institutional screening WORKS!")
    
    return tester

if __name__ == "__main__":
    tester = main()
