#!/usr/bin/env python3
"""
3-MONTH FORWARD TESTING SYSTEM
Train on 1 year historical data, then trade live for 3 months
Track daily buy/sell/hold decisions on 25 stocks
See exactly how much we make with institutional momentum strategy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class ThreeMonthForwardTester:
    def __init__(self):
        # OUR 25 TRACKED STOCKS
        self.stocks = [
            # TECH GIANTS
            "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN",
            # VOLATILE GROWTH  
            "TSLA", "META", "NFLX", "CRM", "UBER",
            # TRADITIONAL VALUE
            "JPM", "WMT", "JNJ", "PG", "KO",
            # EMERGING/SPECULATIVE
            "PLTR", "COIN", "SNOW", "AMD", "INTC",
            # ENERGY/MATERIALS
            "XOM", "CVX", "CAT", "BA", "GE"
        ]
        
        # PROVEN INSTITUTIONAL MOMENTUM PARAMETERS
        self.config = {
            'momentum_6m_threshold': 0.15,     # 15% 6-month momentum
            'momentum_3m_threshold': 0.08,     # 8% 3-month momentum  
            'momentum_1m_threshold': 0.03,     # 3% 1-month momentum
            'rsi_max': 75,                     # Not overbought
            'rsi_min': 30,                     # Not oversold
            'volume_multiplier': 1.2,          # Above average volume
            'volatility_max': 0.40,            # Max volatility
        }
        
        # TRADING RESULTS
        self.daily_decisions = []
        self.portfolio_value_history = []
        self.trade_log = []
        
        # PORTFOLIO SETTINGS
        self.initial_capital = 100000  # $100K starting capital
        self.max_positions = 5         # Max 5 stocks at once
        self.position_size = 0.18      # 18% per position (90% invested max)
    
    def get_training_data(self, symbol, training_end_date):
        """Get 1 year of training data ending at specific date"""
        try:
            training_start = training_end_date - timedelta(days=400)  # ~1 year + buffer
            
            data = yf.download(symbol, start=training_start, end=training_end_date, progress=False)
            if data.empty or len(data) < 200:
                return None
            
            return self.calculate_indicators(data)
        except:
            return None
    
    def get_forward_test_data(self, symbol, start_date, end_date):
        """Get 3 months of forward testing data"""
        try:
            # Get extra data for indicators
            extended_start = start_date - timedelta(days=200)
            
            # Convert dates to datetime objects for yfinance
            if isinstance(start_date, datetime):
                start_date = start_date.date()
            if isinstance(end_date, datetime):
                end_date = end_date.date()
            if isinstance(extended_start, datetime):
                extended_start = extended_start.date()
            
            print(f"   Downloading {symbol} from {extended_start} to {end_date}")
            data = yf.download(symbol, start=extended_start, end=end_date, progress=False, auto_adjust=True)
            
            if data.empty:
                print(f"   ❌ {symbol}: Empty dataset")
                return None
            
            data = self.calculate_indicators(data)
            
            # Return only the forward test period
            forward_data = data[data.index >= pd.Timestamp(start_date)]
            print(f"   ✅ {symbol}: {len(forward_data)} forward test days")
            return forward_data
            
        except Exception as e:
            print(f"   ❌ {symbol}: Exception - {str(e)}")
            return None
    
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        data = data.copy()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume MA
        data['Volume_MA'] = data['Volume'].rolling(20).mean()
        
        return data
    
    def calculate_momentum_score(self, data, current_idx):
        """Calculate institutional momentum score for a specific date"""
        try:
            current_price = float(data['Close'].iloc[current_idx])
            
            # Need enough historical data for momentum
            if current_idx < 126:  # Need ~6 months of data
                return None
            
            # MOMENTUM CALCULATIONS
            price_6m_ago = float(data['Close'].iloc[current_idx - 126])  # ~6 months
            price_3m_ago = float(data['Close'].iloc[current_idx - 63])   # ~3 months  
            price_1m_ago = float(data['Close'].iloc[current_idx - 21])   # ~1 month
            
            momentum_6m = (current_price - price_6m_ago) / price_6m_ago
            momentum_3m = (current_price - price_3m_ago) / price_3m_ago
            momentum_1m = (current_price - price_1m_ago) / price_1m_ago
            
            # RSI
            rsi = float(data['RSI'].iloc[current_idx]) if not pd.isna(data['RSI'].iloc[current_idx]) else 50
            
            # Volume ratio
            current_volume = float(data['Volume'].iloc[current_idx])
            avg_volume = float(data['Volume_MA'].iloc[current_idx])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volatility
            recent_returns = data['Close'].iloc[current_idx-10:current_idx].pct_change().dropna()
            volatility = float(recent_returns.std()) * np.sqrt(252) if len(recent_returns) > 0 else 0
            
            return {
                'momentum_6m': momentum_6m,
                'momentum_3m': momentum_3m,
                'momentum_1m': momentum_1m,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'price': current_price
            }
        except:
            return None
    
    def should_buy_stock(self, metrics):
        """Determine if we should buy based on institutional criteria"""
        if not metrics:
            return False
        
        # ALL momentum criteria must be met
        momentum_good = (
            metrics['momentum_6m'] >= self.config['momentum_6m_threshold'] and
            metrics['momentum_3m'] >= self.config['momentum_3m_threshold'] and
            metrics['momentum_1m'] >= self.config['momentum_1m_threshold']
        )
        
        # Risk criteria
        rsi_good = self.config['rsi_min'] <= metrics['rsi'] <= self.config['rsi_max']
        volume_good = metrics['volume_ratio'] >= self.config['volume_multiplier']
        volatility_good = metrics['volatility'] <= self.config['volatility_max']
        
        return momentum_good and rsi_good and volume_good and volatility_good
    
    def should_sell_stock(self, metrics):
        """Determine if we should sell based on momentum breakdown"""
        if not metrics:
            return True  # Sell if we can't calculate metrics
        
        # Sell if momentum breaks down significantly
        momentum_broken = (
            metrics['momentum_6m'] < 0.05 or    # 6M momentum below 5%
            metrics['momentum_1m'] < -0.02 or   # 1M momentum below -2%
            metrics['rsi'] > 85 or              # Extremely overbought
            metrics['volatility'] > 0.50       # Too volatile
        )
        
        return momentum_broken
    
    def run_forward_test(self, start_date=None, end_date=None):
        """Run 3-month forward test with daily decisions"""
        
        # Default to recent 3-month period with available data  
        if not end_date:
            end_date = datetime(2024, 6, 30).date()  # June 2024
        if not start_date:
            start_date = datetime(2024, 4, 1).date()  # April-June 2024 (3 months)
        
        print(f"🚀 3-MONTH FORWARD TEST")
        print("=" * 60)
        print(f"📅 Test Period: {start_date} to {end_date}")
        print(f"📊 Tracking {len(self.stocks)} stocks daily")
        print(f"💰 Starting Capital: ${self.initial_capital:,}")
        print(f"📈 Max Positions: {self.max_positions}")
        print(f"🎯 Position Size: {self.position_size:.0%}")
        print("=" * 60)
        
        # Portfolio state
        cash = self.initial_capital
        positions = {}  # {symbol: {'shares': X, 'avg_price': Y, 'entry_date': Z}}
        daily_portfolio_values = []
        
        # Get all stock data for the period
        print("📥 Loading stock data...")
        stock_data = {}
        for symbol in self.stocks:
            try:
                data = self.get_forward_test_data(symbol, start_date, end_date)
                if data is not None and len(data) > 0:
                    stock_data[symbol] = data
                    print(f"✅ {symbol}: {len(data)} days")
                else:
                    print(f"❌ {symbol}: No data")
            except Exception as e:
                print(f"❌ {symbol}: Error - {str(e)}")
        
        print(f"✅ Loaded data for {len(stock_data)} stocks")
        
        if len(stock_data) == 0:
            print("❌ No stock data loaded! Trying different date range...")
            # Try a different date range
            end_date = datetime(2024, 8, 30).date()
            start_date = datetime(2024, 6, 1).date()
            print(f"📅 Trying new period: {start_date} to {end_date}")
            
            for symbol in self.stocks[:5]:  # Try first 5 stocks
                try:
                    data = self.get_forward_test_data(symbol, start_date, end_date)
                    if data is not None and len(data) > 0:
                        stock_data[symbol] = data
                        print(f"✅ {symbol}: {len(data)} days")
                        break  # Just need one to proceed
                except Exception as e:
                    print(f"❌ {symbol}: Error - {str(e)}")
        
        if len(stock_data) == 0:
            print("❌ Still no data! Check internet connection and try again.")
            return None
        
        # Get trading days in the period
        sample_stock = list(stock_data.values())[0]
        trading_days = sample_stock[sample_stock.index >= pd.Timestamp(start_date)].index
        
        print(f"📅 Trading {len(trading_days)} days...")
        print("-" * 60)
        
        # Daily trading loop
        for day_idx, current_date in enumerate(trading_days):
            daily_decisions = {'date': current_date.date(), 'actions': []}
            
            # Calculate current portfolio value
            portfolio_value = cash
            for symbol, position in positions.items():
                if symbol in stock_data and current_date in stock_data[symbol].index:
                    current_price = float(stock_data[symbol].loc[current_date, 'Close'])
                    portfolio_value += position['shares'] * current_price
            
            daily_portfolio_values.append({
                'date': current_date.date(),
                'portfolio_value': portfolio_value,
                'cash': cash,
                'positions': len(positions)
            })
            
            # Check each stock for buy/sell signals
            for symbol in self.stocks:
                if symbol not in stock_data or current_date not in stock_data[symbol].index:
                    continue
                
                data = stock_data[symbol]
                current_idx = data.index.get_loc(current_date)
                
                # Skip if not enough historical data
                if current_idx < 126:
                    continue
                
                metrics = self.calculate_momentum_score(data, current_idx)
                current_price = float(data.loc[current_date, 'Close'])
                
                # SELL DECISION
                if symbol in positions:
                    if self.should_sell_stock(metrics):
                        # SELL
                        position = positions[symbol]
                        sale_value = position['shares'] * current_price
                        cash += sale_value
                        
                        # Calculate P&L
                        cost_basis = position['shares'] * position['avg_price']
                        pnl = sale_value - cost_basis
                        pnl_pct = (pnl / cost_basis) * 100
                        
                        self.trade_log.append({
                            'date': current_date.date(),
                            'symbol': symbol,
                            'action': 'SELL',
                            'shares': position['shares'],
                            'price': current_price,
                            'value': sale_value,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'hold_days': (current_date.date() - position['entry_date']).days
                        })
                        
                        daily_decisions['actions'].append(f"SELL {symbol} ${current_price:.2f} (P&L: {pnl_pct:+.1f}%)")
                        del positions[symbol]
                
                # BUY DECISION
                elif (symbol not in positions and 
                      len(positions) < self.max_positions and 
                      self.should_buy_stock(metrics)):
                    
                    # BUY
                    position_value = self.initial_capital * self.position_size
                    if cash >= position_value:
                        shares = position_value / current_price
                        cash -= position_value
                        
                        positions[symbol] = {
                            'shares': shares,
                            'avg_price': current_price,
                            'entry_date': current_date.date()
                        }
                        
                        self.trade_log.append({
                            'date': current_date.date(),
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': current_price,
                            'value': position_value,
                            'pnl': 0,
                            'pnl_pct': 0,
                            'hold_days': 0
                        })
                        
                        daily_decisions['actions'].append(f"BUY {symbol} ${current_price:.2f} ({shares:.0f} shares)")
            
            # Store daily decisions
            self.daily_decisions.append(daily_decisions)
            
            # Progress update every 10 days
            if (day_idx + 1) % 10 == 0:
                print(f"📅 Day {day_idx + 1:2d}/{len(trading_days)}: "
                      f"Portfolio ${portfolio_value:,.0f} | "
                      f"Positions: {len(positions)} | "
                      f"Cash: ${cash:,.0f}")
        
        # Final portfolio value
        final_portfolio_value = cash
        for symbol, position in positions.items():
            if symbol in stock_data:
                final_price = float(stock_data[symbol]['Close'].iloc[-1])
                final_portfolio_value += position['shares'] * final_price
        
        # PERFORMANCE ANALYSIS
        total_return = ((final_portfolio_value - self.initial_capital) / self.initial_capital) * 100
        
        print(f"\n🏆 3-MONTH FORWARD TEST RESULTS")
        print("=" * 60)
        print(f"💰 Starting Capital: ${self.initial_capital:,}")
        print(f"💰 Final Portfolio Value: ${final_portfolio_value:,.0f}")
        print(f"📈 Total Return: {total_return:+.1f}%")
        print(f"📊 Total Trades: {len(self.trade_log)}")
        
        # Analyze trades
        buy_trades = [t for t in self.trade_log if t['action'] == 'BUY']
        sell_trades = [t for t in self.trade_log if t['action'] == 'SELL']
        
        if sell_trades:
            winning_trades = [t for t in sell_trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(sell_trades)
            avg_gain = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl_pct'] for t in sell_trades if t['pnl'] < 0])
            avg_hold_days = np.mean([t['hold_days'] for t in sell_trades])
            
            print(f"🎯 Win Rate: {win_rate:.1%} ({len(winning_trades)}/{len(sell_trades)})")
            print(f"📈 Average Gain: {avg_gain:+.1f}%")
            print(f"📉 Average Loss: {avg_loss:+.1f}%")
            print(f"⏱️  Average Hold Time: {avg_hold_days:.0f} days")
        
        # Calculate buy-and-hold benchmark
        self.calculate_benchmark_performance(start_date, end_date)
        
        return {
            'total_return': total_return,
            'final_value': final_portfolio_value,
            'trades': len(self.trade_log),
            'daily_decisions': self.daily_decisions,
            'portfolio_history': daily_portfolio_values
        }
    
    def calculate_benchmark_performance(self, start_date, end_date):
        """Calculate buy-and-hold performance for comparison"""
        print(f"\n📊 BENCHMARK COMPARISON (Buy & Hold)")
        print("-" * 40)
        
        # Calculate equal-weight buy-and-hold for our 25 stocks
        benchmark_returns = []
        
        for symbol in self.stocks[:5]:  # Sample 5 stocks for benchmark
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    start_price = float(data['Close'].iloc[0])
                    end_price = float(data['Close'].iloc[-1])
                    stock_return = ((end_price - start_price) / start_price) * 100
                    benchmark_returns.append(stock_return)
                    print(f"   {symbol}: {stock_return:+.1f}%")
            except:
                continue
        
        if benchmark_returns:
            avg_benchmark = np.mean(benchmark_returns)
            print(f"📈 Average Buy & Hold: {avg_benchmark:+.1f}%")
        
    def show_daily_activity(self, days=10):
        """Show recent daily trading activity"""
        print(f"\n📅 LAST {days} DAYS TRADING ACTIVITY")
        print("-" * 50)
        
        recent_days = self.daily_decisions[-days:] if len(self.daily_decisions) >= days else self.daily_decisions
        
        for day in recent_days:
            date_str = day['date'].strftime('%Y-%m-%d')
            if day['actions']:
                print(f"{date_str}: {', '.join(day['actions'])}")
            else:
                print(f"{date_str}: HOLD (No actions)")
    
    def save_results(self):
        """Save detailed results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"3month_forward_test_{timestamp}.json"
        
        # Convert dates to strings for JSON serialization
        save_data = {
            'test_config': {
                'stocks': self.stocks,
                'parameters': self.config,
                'initial_capital': self.initial_capital,
                'max_positions': self.max_positions,
                'position_size': self.position_size
            },
            'trade_log': self.trade_log,
            'daily_decisions': self.daily_decisions,
            'portfolio_history': self.portfolio_value_history
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filename}")

def main():
    """Run 3-month forward test"""
    print("🚀 3-MONTH INSTITUTIONAL MOMENTUM FORWARD TEST")
    print("=" * 60)
    print("📚 Strategy: Jegadeesh & Titman (1993) momentum")
    print("📊 Training: 1 year historical data")  
    print("🎯 Testing: 3 months forward with daily decisions")
    print("💰 Portfolio: $100K with 5 max positions")
    print("=" * 60)
    
    tester = ThreeMonthForwardTester()
    
    # Run the test
    results = tester.run_forward_test()
    
    # Show daily activity
    tester.show_daily_activity(days=10)
    
    # Save results
    tester.save_results()
    
    print(f"\n✅ FORWARD TEST COMPLETE!")
    print(f"📊 Total Return: {results['total_return']:+.1f}%")
    print(f"💰 Final Value: ${results['final_value']:,.0f}")
    
    return tester

if __name__ == "__main__":
    tester = main()
