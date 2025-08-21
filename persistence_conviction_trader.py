#!/usr/bin/env python3
"""
PERSISTENCE CONVICTION TRADER
The solution to your trading problem:
- High conviction allocations when momentum is strong
- PERSISTENCE: Hold winners longer, don't chase new momentum
- Smart rebalancing: Only sell when momentum turns negative
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json

class PersistenceConvictionTrader:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trade_history = []
        self.position_entry_momentum = {}  # Track when we entered each position
        
        # Conviction thresholds - when we go big
        self.conviction_thresholds = {
            0.50: 0.70,  # 50%+ momentum = 70% allocation (was 80%, reducing leverage)
            0.30: 0.50,  # 30%+ momentum = 50% allocation  
            0.20: 0.30,  # 20%+ momentum = 30% allocation
            0.10: 0.15,  # 10%+ momentum = 15% allocation
        }
        
        # PERSISTENCE RULES: When do we sell?
        self.persistence_rules = {
            'momentum_decay_threshold': -0.10,  # Sell if momentum drops below -10%
            'min_hold_periods': 2,  # Hold for at least 2 rebalance periods
            'stop_loss': -0.25,  # Hard stop at -25% loss
            'profit_taking_threshold': 0.80,  # Take some profits at +80%
        }
        
        self.universe = [
            'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX',
            'CRM', 'UBER', 'PLTR', 'AMD', 'SNOW', 'COIN',
            'BAC', 'DIS', 'JPM', 'KO', 'PG', 'WMT', 'JNJ', 'JPM', 'GOOG',
            'PEP', 'V', 'MA', 'CSCO', 'MCD', 'ABT', 'TMO', 'COST', 'AVGO', 'LLY'
        ][:30]  # Use first 30 for now
        
        self.rebalance_count = 0
    
    def get_momentum_score(self, symbol, start_date, end_date):
        """Calculate 30-day momentum score"""
        try:
            data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), 
                             end=end_date.strftime('%Y-%m-%d'), progress=False)
            
            if len(data) < 30:
                return 0
                
            current_price = float(data['Close'].iloc[-1])
            price_30d = float(data['Close'].iloc[-30])
            momentum = (current_price / price_30d - 1)
            return momentum
                
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return 0
    
    def get_conviction_allocation(self, momentum_score):
        """Map momentum score to allocation percentage"""
        for threshold, allocation in sorted(self.conviction_thresholds.items(), reverse=True):
            if momentum_score >= threshold:
                return allocation
        return 0
    
    def should_sell_position(self, symbol, current_momentum, date):
        """PERSISTENCE LOGIC: When should we sell?"""
        if symbol not in self.positions:
            return False
            
        position = self.positions[symbol]
        entry_momentum = self.position_entry_momentum.get(symbol, 0)
        periods_held = self.rebalance_count - position.get('entry_period', 0)
        
        # Calculate current P&L
        current_price = self.get_current_price(symbol, date)
        if current_price == 0:
            return False
            
        entry_price = position['price']
        pnl_pct = (current_price / entry_price - 1)
        
        # Selling triggers (in order of priority):
        
        # 1. Hard stop loss
        if pnl_pct <= self.persistence_rules['stop_loss']:
            print(f"🛑 STOP LOSS: {symbol} at {pnl_pct:+.1%}")
            return True
        
        # 2. Don't sell during minimum hold period (unless stop loss)
        if periods_held < self.persistence_rules['min_hold_periods']:
            print(f"🔒 HOLDING: {symbol} (period {periods_held}/{self.persistence_rules['min_hold_periods']})")
            return False
        
        # 3. Momentum has turned significantly negative
        if current_momentum <= self.persistence_rules['momentum_decay_threshold']:
            print(f"📉 MOMENTUM DECAY: {symbol} momentum {current_momentum:+.1%}")
            return True
        
        # 4. Take some profits on big winners (sell half)
        if pnl_pct >= self.persistence_rules['profit_taking_threshold']:
            print(f"💰 PROFIT TAKING: {symbol} at {pnl_pct:+.1%}")
            # TODO: Implement partial selling
            return False
        
        # 5. Hold everything else - PERSISTENCE!
        return False
    
    def rebalance_portfolio(self, date):
        """Execute PERSISTENCE CONVICTION rebalancing"""
        self.rebalance_count += 1
        
        print(f"\n🔥 PERSISTENCE CONVICTION REBALANCE #{self.rebalance_count} - {date.strftime('%Y-%m-%d')}")
        print("=" * 70)
        
        portfolio_value = self.get_portfolio_value(date)
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        
        print(f"\n📊 MOMENTUM ANALYSIS:")
        print("-" * 40)
        
        # Analyze all stocks for momentum
        momentum_data = []
        for symbol in self.universe:
            momentum = self.get_momentum_score(symbol, date - timedelta(days=60), date)
            allocation = self.get_conviction_allocation(momentum)
            
            momentum_data.append({
                'symbol': symbol,
                'momentum': momentum,
                'allocation': allocation,
                'in_portfolio': symbol in self.positions
            })
            
            if allocation > 0:
                print(f"🎯 {symbol}: {momentum:+.1%} momentum → {allocation:.0%} allocation")
        
        # PERSISTENCE DECISION: What to sell?
        print(f"\n🤔 PERSISTENCE DECISIONS:")
        print("-" * 40)
        
        positions_to_sell = []
        for symbol in list(self.positions.keys()):
            current_momentum = next((x['momentum'] for x in momentum_data if x['symbol'] == symbol), 0)
            
            if self.should_sell_position(symbol, current_momentum, date):
                positions_to_sell.append(symbol)
            else:
                print(f"✅ HOLDING: {symbol} (momentum: {current_momentum:+.1%})")
        
        # Execute sells
        for symbol in positions_to_sell:
            self.sell_position(symbol, date)
        
        # Calculate available cash after sells
        available_cash = self.cash
        current_allocation = 0
        
        # Account for existing positions we're keeping
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol, date)
            position_value = position['shares'] * current_price
            current_allocation += position_value / portfolio_value
        
        print(f"\n🎯 NEW POSITION TARGETS:")
        print("-" * 40)
        
        # Sort by momentum strength
        momentum_data.sort(key=lambda x: x['momentum'], reverse=True)
        
        total_target_allocation = 0
        new_positions = []
        
        for data in momentum_data:
            if data['allocation'] > 0 and not data['in_portfolio']:
                # Only add new positions if we have room
                if total_target_allocation + data['allocation'] <= 1.0:
                    new_positions.append(data)
                    total_target_allocation += data['allocation']
                    print(f"🆕 {data['symbol']}: {data['momentum']:+.1%} → {data['allocation']:.0%}")
        
        # Execute new buys
        for data in new_positions:
            symbol = data['symbol']
            allocation = data['allocation']
            position_value = portfolio_value * allocation
            current_price = self.get_current_price(symbol, date)
            
            if current_price > 0 and position_value <= available_cash:
                shares = int(position_value / current_price)
                if shares > 0:
                    self.buy_position(symbol, shares, current_price, date, data['momentum'])
                    available_cash -= shares * current_price
        
        final_value = self.get_portfolio_value(date)
        print(f"\n💰 PORTFOLIO UPDATE:")
        print(f"📈 Portfolio Value: ${final_value:,.2f} (Cash: ${self.cash:,.2f})")
        
        # Show current positions
        if self.positions:
            print(f"\n📊 CURRENT POSITIONS:")
            for symbol, position in self.positions.items():
                current_price = self.get_current_price(symbol, date)
                position_value = position['shares'] * current_price
                pnl = position_value - (position['shares'] * position['price'])
                pnl_pct = (current_price / position['price'] - 1)
                periods_held = self.rebalance_count - position.get('entry_period', 0)
                print(f"   {symbol}: {position['shares']} shares, ${position_value:,.0f}, {pnl_pct:+.1%} (held {periods_held} periods)")
    
    def buy_position(self, symbol, shares, price, date, entry_momentum):
        """Buy a position with persistence tracking"""
        cost = shares * price
        if cost <= self.cash:
            self.positions[symbol] = {
                'shares': shares, 
                'price': price,
                'entry_period': self.rebalance_count,
                'entry_date': date.strftime('%Y-%m-%d')
            }
            self.position_entry_momentum[symbol] = entry_momentum
            self.cash -= cost
            
            self.trade_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'action': 'BUY',
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'value': cost,
                'entry_momentum': entry_momentum
            })
    
    def sell_position(self, symbol, date):
        """Sell a position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            shares = position['shares']
            current_price = self.get_current_price(symbol, date)
            
            if current_price > 0:
                proceeds = shares * current_price
                self.cash += proceeds
                
                pnl = proceeds - (shares * position['price'])
                pnl_pct = (current_price / position['price'] - 1)
                
                self.trade_history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'action': 'SELL',
                    'symbol': symbol,
                    'shares': shares,
                    'price': current_price,
                    'value': proceeds,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                
                del self.positions[symbol]
                if symbol in self.position_entry_momentum:
                    del self.position_entry_momentum[symbol]
    
    def get_current_price(self, symbol, date):
        """Get stock price on specific date with weekend handling"""
        try:
            for i in range(5):  # Try up to 5 days ahead for weekends
                try_date = date + timedelta(days=i)
                data = yf.download(symbol, start=try_date.strftime('%Y-%m-%d'), 
                                 end=(try_date + timedelta(days=1)).strftime('%Y-%m-%d'), progress=False)
                if len(data) > 0:
                    return float(data['Close'].iloc[-1])
            return 0
        except:
            return 0
    
    def get_portfolio_value(self, date):
        """Calculate total portfolio value"""
        total_value = self.cash
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol, date)
            if current_price > 0:
                total_value += position['shares'] * current_price
        return total_value
    
    def run_backtest(self):
        """Run the PERSISTENCE CONVICTION backtest"""
        print("🚀 STARTING PERSISTENCE CONVICTION STRATEGY")
        print("=" * 60)
        print("🎯 High conviction allocations when momentum is strong")
        print("⏳ PERSISTENCE: Hold winners, don't chase every signal")
        print("🛡️  Smart exits: Stop losses, momentum decay, profit taking")
        print("💰 Starting Capital: $100,000")
        print()
        
        # Test period
        start_date = datetime(2024, 5, 13)
        end_date = datetime(2024, 8, 13)
        
        # Monthly rebalancing
        rebalance_dates = [
            datetime(2024, 5, 13),  # Month 1
            datetime(2024, 6, 13),  # Month 2  
            datetime(2024, 7, 15),  # Month 3
        ]
        
        for date in rebalance_dates:
            self.rebalance_portfolio(date)
        
        # Final results
        final_value = self.get_portfolio_value(end_date)
        total_return = (final_value / self.initial_capital - 1)
        profit = final_value - self.initial_capital
        
        print(f"\n🏆 FINAL PERSISTENCE CONVICTION RESULTS - {end_date.strftime('%Y-%m-%d')}")
        print("=" * 70)
        
        print(f"\n🎯 FINAL POSITIONS:")
        print("-" * 25)
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol, end_date)
            position_value = position['shares'] * current_price
            pnl = position_value - (position['shares'] * position['price'])
            pnl_pct = (current_price / position['price'] - 1)
            periods_held = self.rebalance_count - position.get('entry_period', 0)
            entry_momentum = self.position_entry_momentum.get(symbol, 0)
            
            print(f"📊 {symbol}: {position['shares']} shares @ ${current_price:.2f}")
            print(f"   Value: ${position_value:,.2f} | P/L: {pnl_pct:+.1%} (${pnl:+,.2f})")
            print(f"   Held: {periods_held} periods | Entry momentum: {entry_momentum:+.1%}")
        
        print(f"\n💰 PERSISTENCE CONVICTION RESULTS:")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:+.2%}")
        print(f"Profit/Loss: ${profit:+,.2f}")
        print(f"Cash Remaining: ${self.cash:,.2f}")
        
        # Compare benchmarks
        nvda_data = yf.download('NVDA', start='2024-05-13', end='2024-08-13', progress=False)
        nvda_start = float(nvda_data['Close'].iloc[0])
        nvda_end = float(nvda_data['Close'].iloc[-1])
        nvda_return = (nvda_end / nvda_start - 1)
        nvda_profit = 100000 * nvda_return
        
        print(f"\n📊 BENCHMARK COMPARISON:")
        print(f"Our Persistence Strategy: {total_return:+.2%} (${profit:+,.2f})")
        print(f"NVDA Buy & Hold: {nvda_return:+.2%} (${nvda_profit:+,.2f})")
        
        if total_return > nvda_return:
            print(f"🎉 WE BEAT NVDA by {total_return - nvda_return:+.2%}! 🚀")
        else:
            print(f"😤 We missed NVDA by {nvda_return - total_return:.2%}")
        
        # Trade summary
        buys = [t for t in self.trade_history if t['action'] == 'BUY']
        sells = [t for t in self.trade_history if t['action'] == 'SELL']
        
        print(f"\n📈 TRADING SUMMARY:")
        print(f"Total Trades: {len(self.trade_history)} ({len(buys)} buys, {len(sells)} sells)")
        
        if sells:
            winning_trades = [t for t in sells if t['pnl'] > 0]
            print(f"Winning Trades: {len(winning_trades)}/{len(sells)} ({len(winning_trades)/len(sells)*100:.1f}%)")
            
            avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t['pnl'] for t in sells if t['pnl'] < 0) / len([t for t in sells if t['pnl'] < 0]) if [t for t in sells if t['pnl'] < 0] else 0
            
            print(f"Average Win: ${avg_win:+,.2f}")
            print(f"Average Loss: ${avg_loss:+,.2f}")
        
        # Save results
        results = {
            'strategy': 'PERSISTENCE CONVICTION',
            'period': '2024-05-13 to 2024-08-13',
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'profit_loss': profit,
            'nvda_return': nvda_return,
            'performance_vs_nvda': total_return - nvda_return,
            'final_positions': self.positions,
            'cash_remaining': self.cash,
            'trade_history': self.trade_history,
            'persistence_rules': self.persistence_rules
        }
        
        with open('persistence_conviction_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to 'persistence_conviction_results.json'")
        print(f"\n🔥 THIS IS PERSISTENCE CONVICTION TRADING!")

if __name__ == "__main__":
    trader = PersistenceConvictionTrader()
    trader.run_backtest()
