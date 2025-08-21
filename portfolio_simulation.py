#!/usr/bin/env python3
"""
Portfolio Performance Simulation - Elite AI v2.0
$10,000 investment simulation based on AI recommendations
Shows actual profit/loss over 3 months following BUY/SELL/HOLD signals
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
import warnings
warnings.filterwarnings('ignore')

class PortfolioSimulation:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.stocks = [
            # MEGA CAP TECH
            "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA",
            # FINANCE & TRADITIONAL
            "JPM", "BAC", "WMT", "JNJ", "PG", "KO", "DIS",
            # GROWTH & EMERGING
            "NFLX", "CRM", "UBER", "PLTR", "SNOW", "COIN"
        ]
        self.portfolio_results = {}
        
    def simulate_portfolio_performance(self):
        """Simulate portfolio performance following AI recommendations"""
        
        print("💰 PORTFOLIO SIMULATION - $10,000 INVESTMENT")
        print("=" * 55)
        print("Following Elite AI v2.0 BUY/SELL/HOLD recommendations")
        print("Training Period: July 2023 - June 2024")
        print("Investment Period: July 2024 - September 2024")
        print("=" * 55)
        
        # Time periods for simulation
        train_start = datetime(2023, 7, 1)
        train_end = datetime(2024, 6, 30)
        invest_start = datetime(2024, 7, 1)  # When we make investment decisions
        invest_end = datetime(2024, 9, 30)   # End of 3-month period
        
        print(f"📅 AI Training: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"📅 Investment: {invest_start.strftime('%Y-%m-%d')} to {invest_end.strftime('%Y-%m-%d')}")
        
        # Get AI recommendations and actual performance for each stock
        stock_results = []
        total_portfolio_value = 0
        cash_position = 0
        
        for symbol in self.stocks:
            result = self.analyze_stock_investment(symbol, train_start, train_end, invest_start, invest_end)
            if result:
                stock_results.append(result)
        
        # Portfolio allocation strategy
        print(f"\n💼 PORTFOLIO ALLOCATION STRATEGY")
        print("=" * 40)
        
        buy_stocks = [r for r in stock_results if r['signal'] == 'BUY']
        sell_stocks = [r for r in stock_results if r['signal'] == 'SELL']
        hold_stocks = [r for r in stock_results if r['signal'] == 'HOLD']
        
        print(f"🟢 BUY signals: {len(buy_stocks)} stocks")
        print(f"🔴 SELL signals: {len(sell_stocks)} stocks")
        print(f"🟡 HOLD signals: {len(hold_stocks)} stocks")
        
        # Investment strategy:
        # - If BUY signals: Invest equally across all BUY stocks
        # - If no BUY signals but HOLD: Invest equally across top HOLD stocks
        # - If SELL signals: Short or avoid (we'll avoid for simplicity)
        
        if buy_stocks:
            # Invest in BUY signals
            investment_per_stock = self.initial_capital / len(buy_stocks)
            investment_stocks = buy_stocks
            strategy = f"BUYING {len(buy_stocks)} stocks with BUY signals"
        elif hold_stocks:
            # If no BUY signals, invest in best HOLD stocks (top 5 by confidence)
            top_holds = sorted(hold_stocks, key=lambda x: abs(x['predicted_return']), reverse=True)[:5]
            investment_per_stock = self.initial_capital / len(top_holds)
            investment_stocks = top_holds
            strategy = f"HOLDING top {len(top_holds)} stocks with best HOLD signals"
        else:
            # If only SELL signals, keep cash
            investment_stocks = []
            cash_position = self.initial_capital
            strategy = "KEEPING CASH due to only SELL signals"
        
        print(f"📊 Strategy: {strategy}")
        if investment_stocks:
            print(f"💰 Investment per stock: ${investment_per_stock:,.2f}")
        
        # Calculate portfolio performance
        portfolio_investments = []
        total_final_value = cash_position
        
        print(f"\n📈 INDIVIDUAL STOCK PERFORMANCE")
        print("=" * 60)
        print(f"{'Stock':<6} {'Signal':<6} {'Invested':<10} {'Final Value':<12} {'Profit/Loss':<12} {'Return':<8}")
        print("-" * 60)
        
        for stock in investment_stocks:
            symbol = stock['symbol']
            signal = stock['signal']
            investment = investment_per_stock
            
            # Calculate shares bought
            shares = investment / stock['price_start']
            
            # Calculate final value
            final_value = shares * stock['price_end']
            profit_loss = final_value - investment
            return_pct = (profit_loss / investment) * 100
            
            total_final_value += final_value
            
            print(f"{symbol:<6} {signal:<6} ${investment:>8.0f} ${final_value:>10.0f} ${profit_loss:>+10.0f} {return_pct:>+6.1f}%")
            
            portfolio_investments.append({
                'symbol': symbol,
                'signal': signal,
                'investment': investment,
                'shares': shares,
                'price_start': stock['price_start'],
                'price_end': stock['price_end'],
                'final_value': final_value,
                'profit_loss': profit_loss,
                'return_pct': return_pct
            })
        
        print("-" * 60)
        
        # Portfolio summary
        total_profit_loss = total_final_value - self.initial_capital
        total_return_pct = (total_profit_loss / self.initial_capital) * 100
        
        print(f"\n💰 PORTFOLIO SUMMARY")
        print("=" * 30)
        print(f"🏦 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"💵 Final Value: ${total_final_value:,.2f}")
        print(f"📈 Total Profit/Loss: ${total_profit_loss:+,.2f}")
        print(f"📊 Total Return: {total_return_pct:+.2f}%")
        print(f"💰 Cash Position: ${cash_position:,.2f}")
        
        # Compare to benchmarks
        print(f"\n📊 BENCHMARK COMPARISON")
        print("=" * 30)
        
        # S&P 500 equivalent (using SPY as proxy)
        spy_return = self.get_benchmark_return("SPY", invest_start, invest_end)
        spy_final_value = self.initial_capital * (1 + spy_return/100)
        spy_profit = spy_final_value - self.initial_capital
        
        print(f"📈 S&P 500 (SPY): {spy_return:+.2f}% (${spy_profit:+,.0f})")
        print(f"🤖 AI Portfolio: {total_return_pct:+.2f}% (${total_profit_loss:+,.0f})")
        
        outperformance = total_return_pct - spy_return
        print(f"🎯 Outperformance: {outperformance:+.2f}%")
        
        # Risk analysis
        if portfolio_investments:
            returns = [inv['return_pct'] for inv in portfolio_investments]
            volatility = np.std(returns)
            best_stock = max(portfolio_investments, key=lambda x: x['return_pct'])
            worst_stock = min(portfolio_investments, key=lambda x: x['return_pct'])
            
            print(f"\n📊 RISK ANALYSIS")
            print("=" * 25)
            print(f"📈 Portfolio Volatility: {volatility:.2f}%")
            print(f"🌟 Best Performer: {best_stock['symbol']} ({best_stock['return_pct']:+.1f}%)")
            print(f"📉 Worst Performer: {worst_stock['symbol']} ({worst_stock['return_pct']:+.1f}%)")
        
        # AI Signal Analysis
        print(f"\n🤖 AI SIGNAL ANALYSIS")
        print("=" * 30)
        
        if stock_results:
            correct_predictions = sum(1 for r in stock_results if r['direction_correct'])
            total_predictions = len(stock_results)
            accuracy = correct_predictions / total_predictions
            
            print(f"📊 AI Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
            print(f"📈 Average Prediction Error: {np.mean([r['prediction_error'] for r in stock_results]):.2f}%")
            
            # Performance by signal type
            if buy_stocks:
                buy_returns = [r['actual_return'] for r in buy_stocks]
                print(f"🟢 BUY signals avg return: {np.mean(buy_returns):+.2f}%")
            
            if sell_stocks:
                sell_returns = [r['actual_return'] for r in sell_stocks]
                print(f"🔴 SELL signals avg return: {np.mean(sell_returns):+.2f}%")
            
            if hold_stocks:
                hold_returns = [r['actual_return'] for r in hold_stocks]
                print(f"🟡 HOLD signals avg return: {np.mean(hold_returns):+.2f}%")
        
        # Final verdict
        print(f"\n🏆 INVESTMENT VERDICT")
        print("=" * 25)
        
        if total_return_pct > 10:
            verdict = "🌟 EXCELLENT"
        elif total_return_pct > 5:
            verdict = "✅ GOOD"
        elif total_return_pct > 0:
            verdict = "🟡 POSITIVE"
        elif total_return_pct > -5:
            verdict = "🟠 SMALL LOSS"
        else:
            verdict = "❌ POOR"
        
        print(f"📊 Performance: {verdict}")
        print(f"💰 3-Month Return: {total_return_pct:+.2f}%")
        print(f"📈 vs S&P 500: {outperformance:+.2f}%")
        
        if total_profit_loss > 0:
            print(f"💵 Profit: ${total_profit_loss:,.2f} in 3 months")
        else:
            print(f"📉 Loss: ${abs(total_profit_loss):,.2f} in 3 months")
        
        print(f"\n🔍 KEY INSIGHTS:")
        print(f"   • AI provided {len(stock_results)} stock recommendations")
        print(f"   • {accuracy:.1%} directional accuracy")
        print(f"   • {'Outperformed' if outperformance > 0 else 'Underperformed'} S&P 500 by {abs(outperformance):.2f}%")
        print(f"   • Portfolio strategy: {strategy}")
        
        return {
            'portfolio_investments': portfolio_investments,
            'initial_capital': self.initial_capital,
            'final_value': total_final_value,
            'profit_loss': total_profit_loss,
            'return_pct': total_return_pct,
            'benchmark_return': spy_return,
            'outperformance': outperformance,
            'ai_accuracy': accuracy if stock_results else 0
        }
    
    def analyze_stock_investment(self, symbol, train_start, train_end, invest_start, invest_end):
        """Analyze single stock for investment simulation"""
        try:
            # Download data
            data = yf.download(symbol, start=train_start - timedelta(days=60), end=invest_end + timedelta(days=5), progress=False)
            
            if len(data) < 100:
                return None
            
            # Create features
            data['returns'] = data['Close'].pct_change()
            data['sma_5'] = data['Close'].rolling(5).mean()
            data['sma_20'] = data['Close'].rolling(20).mean()
            data['target'] = data['returns'].shift(-1)
            
            # Split data for training
            train_data = data[data.index <= train_end].copy()
            
            if len(train_data) < 50:
                return None
            
            # Train AI model
            train_clean = train_data.dropna()
            features = ['returns', 'sma_5', 'sma_20']
            X_train = train_clean[features].values
            y_train = train_clean['target'].values
            
            if len(X_train) < 30:
                return None
            
            # Simple ensemble model
            models = {
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'rf': RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
            }
            
            predictions = {}
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    last_features = X_train[-1:]
                    pred = model.predict(last_features)[0]
                    predictions[name] = pred
                except:
                    continue
            
            if not predictions:
                return None
            
            # Ensemble prediction
            ensemble_pred = np.mean(list(predictions.values()))
            predicted_return = ensemble_pred * 100
            
            # Generate signal
            if predicted_return > 3:
                signal = "BUY"
            elif predicted_return < -3:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            # Get actual prices for investment period
            price_start = float(data[data.index <= invest_start]['Close'].iloc[-1])
            price_end = float(data[data.index <= invest_end]['Close'].iloc[-1])
            actual_return = ((price_end - price_start) / price_start) * 100
            
            # Calculate prediction accuracy
            prediction_error = abs(predicted_return - actual_return)
            direction_correct = (predicted_return > 0) == (actual_return > 0)
            
            return {
                'symbol': symbol,
                'signal': signal,
                'predicted_return': predicted_return,
                'actual_return': actual_return,
                'prediction_error': prediction_error,
                'direction_correct': direction_correct,
                'price_start': price_start,
                'price_end': price_end
            }
            
        except Exception as e:
            return None
    
    def get_benchmark_return(self, symbol, start_date, end_date):
        """Get benchmark return for comparison"""
        try:
            data = yf.download(symbol, start=start_date - timedelta(days=10), end=end_date + timedelta(days=5), progress=False)
            price_start = float(data[data.index <= start_date]['Close'].iloc[-1])
            price_end = float(data[data.index <= end_date]['Close'].iloc[-1])
            return ((price_end - price_start) / price_start) * 100
        except:
            return 0

def main():
    """Run portfolio simulation"""
    
    # Test different investment amounts
    amounts = [10000, 25000, 50000]
    
    for amount in amounts:
        print(f"\n{'='*60}")
        print(f"💰 SIMULATION: ${amount:,} INVESTMENT")
        print(f"{'='*60}")
        
        simulator = PortfolioSimulation(initial_capital=amount)
        results = simulator.simulate_portfolio_performance()
        
        if results['profit_loss'] > 0:
            print(f"\n🎉 SUCCESS: Made ${results['profit_loss']:,.2f} profit in 3 months!")
        else:
            print(f"\n📉 Loss: ${abs(results['profit_loss']):,.2f} loss in 3 months")
        
        print(f"📊 ROI: {results['return_pct']:+.2f}% in 3 months")
        print(f"📈 Annualized: {results['return_pct']*4:+.2f}% (if sustained)")
    
    print(f"\n" + "="*65)
    print(f"🎯 PORTFOLIO SIMULATION COMPLETE!")
    print(f"   • Real investment simulation based on AI signals")
    print(f"   • 3-month performance tracked")
    print(f"   • Compared to S&P 500 benchmark")
    print(f"   • Shows actual profit/loss from following AI")
    print(f"="*65)

if __name__ == "__main__":
    main()
