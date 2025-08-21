#!/usr/bin/env python3
"""
🚀 LIVE TRADING ALGORITHM BACKTEST
Backtest our actual live trading system (ImprovedAIPortfolioManager)
Train on 1 year of data, then trade for 3 months
Shows exactly what our Alpaca integration would do
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Import our live trading system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from improved_ai_portfolio_manager import ImprovedAIPortfolioManager

class LiveTradingBacktester:
    def __init__(self, starting_capital=10000):
        self.starting_capital = starting_capital
        self.ai_manager = ImprovedAIPortfolioManager(capital=starting_capital)
        self.results = {}
        
    def run_comprehensive_backtest(self):
        """Run complete backtest: 1 year training + 3 months trading"""
        print("🚀 LIVE TRADING ALGORITHM BACKTEST")
        print("=" * 60)
        print("📊 Strategy: ImprovedAIPortfolioManager (actual live trading system)")
        print("🧠 Training Period: 1 year of historical data")
        print("📈 Trading Period: 3 months")
        print(f"💰 Starting Capital: ${self.starting_capital:,}")
        print("=" * 60)
        
        # Define time periods
        end_date = datetime.now()
        trading_start = end_date - timedelta(days=90)  # 3 months ago
        training_start = trading_start - timedelta(days=365)  # 1 year before trading
        
        print(f"📅 Training Period: {training_start.strftime('%Y-%m-%d')} to {trading_start.strftime('%Y-%m-%d')}")
        print(f"📅 Trading Period: {trading_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print()
        
        # Step 1: Train models on 1 year of data
        print("🧠 PHASE 1: TRAINING AI MODELS")
        print("-" * 40)
        self.train_models_on_historical_data(training_start, trading_start)
        
        # Step 2: Trade for 3 months
        print("\n📈 PHASE 2: LIVE TRADING SIMULATION")
        print("-" * 40)
        trading_results = self.simulate_trading_period(trading_start, end_date)
        
        # Step 3: Generate comprehensive report
        print("\n📊 PHASE 3: PERFORMANCE ANALYSIS")
        print("-" * 40)
        self.generate_performance_report(trading_results, trading_start, end_date)
        
        return trading_results
    
    def train_models_on_historical_data(self, start_date, end_date):
        """Train AI models on 1 year of historical data"""
        print("Training AI models on historical data...")
        
        # Get stock universe
        stock_universe = self.ai_manager.stock_universe
        print(f"📊 Training on {len(stock_universe)} stocks: {', '.join(stock_universe[:10])}{'...' if len(stock_universe) > 10 else ''}")
        
        successful_models = 0
        
        for i, symbol in enumerate(stock_universe, 1):
            try:
                print(f"   🧠 Training {symbol} ({i}/{len(stock_universe)})...")
                
                # Download training data
                stock = yf.Ticker(symbol)
                training_data = stock.history(start=start_date, end=end_date)
                
                if len(training_data) < 100:
                    print(f"      ❌ Insufficient data for {symbol}")
                    continue
                
                # Train model using our live trading system's method
                model, scaler, r2_score = self.ai_manager.train_improved_model(symbol, data=training_data)
                
                if model is not None:
                    self.ai_manager.models[symbol] = (model, scaler, r2_score)
                    successful_models += 1
                    print(f"      ✅ Trained successfully (R² = {r2_score:.3f})")
                else:
                    print(f"      ❌ Training failed")
                    
            except Exception as e:
                print(f"      ❌ Error training {symbol}: {e}")
                continue
        
        print(f"\n✅ Training Complete: {successful_models}/{len(stock_universe)} models trained successfully")
        return successful_models
    
    def simulate_trading_period(self, start_date, end_date):
        """Simulate 3 months of trading with weekly rebalancing"""
        print("Simulating live trading with weekly rebalancing...")
        
        # Initialize portfolio
        cash = self.starting_capital
        positions = {}
        portfolio_values = []
        trades = []
        
        # Generate weekly trading dates
        current_date = start_date
        trading_dates = []
        
        while current_date <= end_date:
            trading_dates.append(current_date)
            current_date += timedelta(days=7)  # Weekly rebalancing
        
        print(f"📅 Trading on {len(trading_dates)} dates (weekly rebalancing)")
        
        # Simulate trading for each date
        for i, trade_date in enumerate(trading_dates, 1):
            print(f"\n📊 Week {i}/{len(trading_dates)}: {trade_date.strftime('%Y-%m-%d')}")
            
            # Get current market data for all stocks
            stock_data = {}
            stock_predictions = {}
            
            for symbol in self.ai_manager.stock_universe:
                try:
                    # Get data up to this trading date
                    stock = yf.Ticker(symbol)
                    data = stock.history(start=trade_date - timedelta(days=60), end=trade_date + timedelta(days=1))
                    
                    if len(data) < 30:
                        continue
                    
                    current_price = float(data['Close'].iloc[-1])
                    stock_data[symbol] = {'price': current_price, 'data': data}
                    
                    # Get AI prediction strength
                    strength = self.ai_manager.get_prediction_strength(symbol, data)
                    stock_predictions[symbol] = strength
                    
                except Exception as e:
                    continue
            
            # Select top 5 stocks to hold (like our live system)
            sorted_predictions = sorted(stock_predictions.items(), key=lambda x: x[1], reverse=True)
            top_stocks = sorted_predictions[:5]
            
            print(f"   🎯 Top stocks selected:")
            for symbol, strength in top_stocks:
                if symbol in stock_data:
                    price = stock_data[symbol]['price']
                    print(f"      {symbol}: Strength {strength:.1f}, Price ${price:.2f}")
            
            # Rebalance portfolio
            total_portfolio_value = cash
            for symbol, shares in positions.items():
                if symbol in stock_data:
                    total_portfolio_value += shares * stock_data[symbol]['price']
            
            # Sell positions not in top 5
            for symbol in list(positions.keys()):
                if symbol not in [s[0] for s in top_stocks] or symbol not in stock_data:
                    if positions[symbol] > 0:
                        sell_value = positions[symbol] * stock_data[symbol]['price']
                        cash += sell_value
                        trades.append({
                            'date': trade_date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'shares': positions[symbol],
                            'price': stock_data[symbol]['price'],
                            'value': sell_value
                        })
                        print(f"      🔴 SELL {symbol}: {positions[symbol]:.2f} shares at ${stock_data[symbol]['price']:.2f}")
                        del positions[symbol]
            
            # Buy new positions (equal weight)
            target_position_value = total_portfolio_value / len(top_stocks)
            
            for symbol, strength in top_stocks:
                if symbol in stock_data and strength > 30:  # Minimum strength threshold
                    current_shares = positions.get(symbol, 0)
                    current_value = current_shares * stock_data[symbol]['price']
                    target_shares = target_position_value / stock_data[symbol]['price']
                    
                    shares_to_buy = target_shares - current_shares
                    
                    if shares_to_buy > 0:
                        buy_value = shares_to_buy * stock_data[symbol]['price']
                        if buy_value <= cash:
                            cash -= buy_value
                            positions[symbol] = positions.get(symbol, 0) + shares_to_buy
                            trades.append({
                                'date': trade_date,
                                'symbol': symbol,
                                'action': 'BUY',
                                'shares': shares_to_buy,
                                'price': stock_data[symbol]['price'],
                                'value': buy_value
                            })
                            print(f"      🟢 BUY {symbol}: {shares_to_buy:.2f} shares at ${stock_data[symbol]['price']:.2f}")
            
            # Calculate portfolio value
            portfolio_value = cash
            for symbol, shares in positions.items():
                if symbol in stock_data:
                    portfolio_value += shares * stock_data[symbol]['price']
            
            portfolio_values.append({
                'date': trade_date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'positions': dict(positions),
                'return_pct': ((portfolio_value - self.starting_capital) / self.starting_capital) * 100
            })
            
            print(f"   💰 Portfolio Value: ${portfolio_value:,.2f} ({((portfolio_value - self.starting_capital) / self.starting_capital) * 100:+.1f}%)")
        
        return {
            'portfolio_values': portfolio_values,
            'trades': trades,
            'final_value': portfolio_values[-1]['portfolio_value'] if portfolio_values else self.starting_capital,
            'total_return': ((portfolio_values[-1]['portfolio_value'] - self.starting_capital) / self.starting_capital) * 100 if portfolio_values else 0
        }
    
    def generate_performance_report(self, results, start_date, end_date):
        """Generate comprehensive performance report"""
        portfolio_values = results['portfolio_values']
        trades = results['trades']
        final_value = results['final_value']
        total_return = results['total_return']
        
        print("📊 LIVE TRADING BACKTEST RESULTS")
        print("=" * 50)
        
        # Basic performance metrics
        print(f"💰 Starting Capital: ${self.starting_capital:,}")
        print(f"💰 Final Portfolio Value: ${final_value:,.2f}")
        print(f"📈 Total Return: {total_return:+.2f}%")
        print(f"📅 Trading Period: {(end_date - start_date).days} days")
        print(f"🔄 Total Trades: {len(trades)}")
        
        # Annualized return
        days_traded = (end_date - start_date).days
        annualized_return = ((final_value / self.starting_capital) ** (365 / days_traded) - 1) * 100
        print(f"📊 Annualized Return: {annualized_return:+.2f}%")
        
        # Compare to buy-and-hold SPY
        try:
            spy = yf.Ticker("SPY")
            spy_data = spy.history(start=start_date, end=end_date)
            spy_return = ((spy_data['Close'].iloc[-1] - spy_data['Close'].iloc[0]) / spy_data['Close'].iloc[0]) * 100
            outperformance = total_return - spy_return
            print(f"📈 SPY Return (3 months): {spy_return:+.2f}%")
            print(f"🎯 Outperformance vs SPY: {outperformance:+.2f}%")
        except:
            print("📈 SPY comparison unavailable")
        
        # Weekly performance
        if len(portfolio_values) > 1:
            weekly_returns = []
            for i in range(1, len(portfolio_values)):
                prev_value = portfolio_values[i-1]['portfolio_value']
                curr_value = portfolio_values[i]['portfolio_value']
                weekly_return = ((curr_value - prev_value) / prev_value) * 100
                weekly_returns.append(weekly_return)
            
            if weekly_returns:
                avg_weekly_return = np.mean(weekly_returns)
                weekly_volatility = np.std(weekly_returns)
                sharpe_ratio = avg_weekly_return / weekly_volatility if weekly_volatility > 0 else 0
                
                print(f"📊 Average Weekly Return: {avg_weekly_return:+.2f}%")
                print(f"📊 Weekly Volatility: {weekly_volatility:.2f}%")
                print(f"📊 Sharpe Ratio (weekly): {sharpe_ratio:.2f}")
        
        # Best and worst weeks
        if len(portfolio_values) > 1:
            best_week = max(weekly_returns)
            worst_week = min(weekly_returns)
            print(f"🟢 Best Week: {best_week:+.2f}%")
            print(f"🔴 Worst Week: {worst_week:+.2f}%")
        
        # Trading activity
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        print(f"\n🔄 TRADING ACTIVITY:")
        print(f"   🟢 Buy Trades: {len(buy_trades)}")
        print(f"   🔴 Sell Trades: {len(sell_trades)}")
        
        if buy_trades:
            total_bought = sum(t['value'] for t in buy_trades)
            print(f"   💰 Total Purchased: ${total_bought:,.2f}")
        
        # Most traded stocks
        symbol_counts = {}
        for trade in trades:
            symbol = trade['symbol']
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        if symbol_counts:
            most_traded = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n📊 MOST TRADED STOCKS:")
            for symbol, count in most_traded:
                print(f"   {symbol}: {count} trades")
        
        # Save detailed results
        filename = f"live_trading_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        save_data = {
            'backtest_config': {
                'starting_capital': self.starting_capital,
                'training_period': f"{start_date.strftime('%Y-%m-%d')} to {(start_date + timedelta(days=365)).strftime('%Y-%m-%d')}",
                'trading_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'strategy': 'ImprovedAIPortfolioManager (Live Trading System)'
            },
            'performance_summary': {
                'total_return_pct': total_return,
                'annualized_return_pct': annualized_return,
                'final_portfolio_value': final_value,
                'total_trades': len(trades),
                'trading_days': days_traded
            },
            'detailed_results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        
        # Final verdict
        print(f"\n🎯 LIVE TRADING SYSTEM VERDICT:")
        if total_return > 0:
            print(f"   ✅ PROFITABLE: {total_return:+.2f}% return in 3 months")
            if total_return > 5:
                print(f"   🚀 STRONG PERFORMANCE: Above 5% in 3 months")
            elif total_return > 2:
                print(f"   📈 GOOD PERFORMANCE: Solid positive returns")
            else:
                print(f"   📊 MODEST GAINS: Small but positive returns")
        else:
            print(f"   ❌ LOSS: {total_return:+.2f}% return in 3 months")
            print(f"   🔧 NEEDS OPTIMIZATION: Consider adjusting parameters")
        
        if annualized_return > 15:
            print(f"   🏆 EXCELLENT: {annualized_return:+.1f}% annualized return")
        elif annualized_return > 8:
            print(f"   ✅ GOOD: {annualized_return:+.1f}% annualized return")
        else:
            print(f"   📊 AVERAGE: {annualized_return:+.1f}% annualized return")
        
        return results

def main():
    """Run the live trading backtest"""
    print("🎯 BACKTESTING OUR LIVE TRADING ALGORITHM")
    print("Using the exact same system that runs in Alpaca integration")
    print()
    
    # Run backtest
    backtester = LiveTradingBacktester(starting_capital=10000)
    results = backtester.run_comprehensive_backtest()
    
    return backtester, results

if __name__ == "__main__":
    backtester, results = main()
