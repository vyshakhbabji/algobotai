#!/usr/bin/env python3
"""
Improved Money Making Simulator
Using the proven trading strategy logic that worked before
"""

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ImprovedMoneySimulator:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.current_shares = 0
        self.models = {}
        self.trade_log = []
        self.portfolio_history = []
        
    def load_models(self):
        """Load trained models"""
        print("🤖 Loading trained models...")
        
        try:
            model_files = {
                'rf': 'fixed_data/models/random_forest_model.pkl',
                'gb': 'fixed_data/models/gradient_boosting_model.pkl',
                'linear': 'fixed_data/models/linear_regression_model.pkl',
                'ridge': 'fixed_data/models/ridge_model.pkl'
            }
            
            for name, path in model_files.items():
                try:
                    self.models[name] = joblib.load(path)
                    print(f"  ✅ {name.upper()} loaded")
                except Exception as e:
                    print(f"  ❌ {name.upper()}: {e}")
            
            print(f"📊 Successfully loaded {len(self.models)} models")
            return len(self.models) > 0
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def fetch_historical_data(self, symbol='NVDA', start_date='2024-06-01'):
        """Fetch historical data from June 2024 to now"""
        print(f"📡 Fetching historical data for {symbol} from {start_date}...")
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=datetime.now().strftime('%Y-%m-%d'))
            
            if len(hist) == 0:
                print("❌ No data retrieved")
                return None
            
            print(f"📈 Retrieved {len(hist)} days of data")
            print(f"📅 Date range: {hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}")
            
            return hist
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators exactly like the working strategy"""
        print("🔧 Calculating technical indicators...")
        
        df = data.copy()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic Oscillator
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # VWAP
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # Price features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Returns
        for period in [1, 2, 3, 5, 10]:
            df[f'Return_{period}d'] = df['Close'].pct_change(period) * 100
        
        # Volatility
        for window in [5, 10, 20]:
            df[f'Volatility_{window}d'] = df['Close'].pct_change().rolling(window=window).std() * np.sqrt(252) * 100
        
        print(f"✅ Calculated technical indicators, dataset now has {df.shape[1]} columns")
        return df
    
    def generate_proven_signals(self, data):
        """Generate signals using the proven strategy logic"""
        print("🧠 Generating signals using proven strategy...")
        
        if not self.models:
            print("❌ No models loaded")
            return None
        
        # Use same feature columns as proven strategy
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
            'Stoch_K', 'Stoch_D', 'ATR', 'OBV', 'Volume_SMA', 'Volume_Ratio',
            'VWAP', 'High_Low_Pct', 'Open_Close_Pct',
            'Return_1d', 'Return_2d', 'Return_3d', 'Return_5d', 'Return_10d',
            'Volatility_5d', 'Volatility_10d', 'Volatility_20d'
        ]
        
        # Create sequences like the working strategy
        X = data[feature_columns].dropna()
        signals = []
        sequence_length = 10
        
        print(f"📊 Creating sequences for {len(X)} days")
        
        for i in range(sequence_length, len(X)):
            try:
                # Create sequence for current day
                sequence = X.iloc[i-sequence_length:i].values
                current_price = data.iloc[X.index[i]]['Close']
                
                # Get ensemble predictions
                predictions = []
                for model_name, model in self.models.items():
                    try:
                        # Flatten sequence for prediction
                        seq_flat = sequence.reshape(1, -1)
                        pred = model.predict(seq_flat)[0]
                        predictions.append(pred)
                    except:
                        predictions.append(current_price)
                
                if len(predictions) == 0:
                    signals.append('HOLD')
                    continue
                
                # Ensemble prediction
                ensemble_pred = np.mean(predictions)
                predicted_return = (ensemble_pred - current_price) / current_price * 100
                
                # Get technical indicators for current day
                current_idx = X.index[i]
                rsi = data.loc[current_idx, 'RSI'] if not pd.isna(data.loc[current_idx, 'RSI']) else 50
                bb_position = data.loc[current_idx, 'BB_Position'] if not pd.isna(data.loc[current_idx, 'BB_Position']) else 0.5
                volume_ratio = data.loc[current_idx, 'Volume_Ratio'] if not pd.isna(data.loc[current_idx, 'Volume_Ratio']) else 1
                
                # Proven signal generation logic (less conservative)
                signal_strength = 0
                
                # 1. Model prediction signal
                if predicted_return > 0.3:  # Lower threshold
                    signal_strength += 2
                elif predicted_return < -0.3:
                    signal_strength -= 2
                elif predicted_return > 0:
                    signal_strength += 1
                else:
                    signal_strength -= 1
                
                # 2. RSI signals
                if rsi < 40:  # Oversold
                    signal_strength += 2
                elif rsi > 60:  # Overbought
                    signal_strength -= 1
                elif rsi < 50:
                    signal_strength += 1
                
                # 3. Bollinger Bands
                if bb_position < 0.2:  # Near lower band
                    signal_strength += 2
                elif bb_position > 0.8:  # Near upper band
                    signal_strength -= 1
                
                # 4. Volume confirmation
                if volume_ratio > 1.2:  # High volume
                    if signal_strength > 0:
                        signal_strength += 1
                    elif signal_strength < 0:
                        signal_strength -= 1
                
                # Generate final signal based on strength
                if signal_strength >= 3:
                    signal = 'BUY'
                elif signal_strength <= -2:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
                
                signals.append(signal)
                
            except Exception as e:
                signals.append('HOLD')
        
        # Create signals dataframe
        valid_indices = X.index[sequence_length:]
        signals_df = pd.DataFrame({
            'Date': valid_indices,
            'Close': data.loc[valid_indices, 'Close'],
            'Signal': signals
        })
        
        # Print signal distribution
        signal_counts = signals_df['Signal'].value_counts()
        print(f"📊 Signal distribution:")
        for signal, count in signal_counts.items():
            print(f"   {signal}: {count} days ({count/len(signals_df)*100:.1f}%)")
        
        return signals_df
    
    def simulate_realistic_trading(self, signals_df):
        """Simulate realistic trading with transaction costs"""
        print(f"\n💰 SIMULATING REALISTIC TRADING WITH ${self.initial_capital:,}")
        print(f"{'='*80}")
        
        self.current_capital = self.initial_capital
        self.current_shares = 0
        self.trade_log = []
        self.portfolio_history = []
        
        transaction_cost = 0.001  # 0.1% transaction cost
        
        for idx, row in signals_df.iterrows():
            date = row['Date']
            price = row['Close']
            signal = row['Signal']
            
            portfolio_value = self.current_capital + (self.current_shares * price)
            action_taken = ""
            
            if signal == 'BUY' and self.current_capital > 500:  # Minimum trade size
                # Calculate shares to buy (reserve some cash)
                available_cash = self.current_capital * 0.95  # Use 95% of cash
                shares_to_buy = int(available_cash / price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    transaction_fee = cost * transaction_cost
                    total_cost = cost + transaction_fee
                    
                    if total_cost <= self.current_capital:
                        self.current_capital -= total_cost
                        self.current_shares += shares_to_buy
                        action_taken = f"BOUGHT {shares_to_buy} shares at ${price:.2f} (fee: ${transaction_fee:.2f})"
                        
                        self.trade_log.append({
                            'Date': date,
                            'Action': 'BUY',
                            'Shares': shares_to_buy,
                            'Price': price,
                            'Cost': cost,
                            'Fee': transaction_fee,
                            'Cash_Remaining': self.current_capital,
                            'Total_Shares': self.current_shares
                        })
            
            elif signal == 'SELL' and self.current_shares > 0:
                # Sell all shares
                revenue = self.current_shares * price
                transaction_fee = revenue * transaction_cost
                net_revenue = revenue - transaction_fee
                
                action_taken = f"SOLD {self.current_shares} shares at ${price:.2f} (fee: ${transaction_fee:.2f})"
                
                self.trade_log.append({
                    'Date': date,
                    'Action': 'SELL',
                    'Shares': self.current_shares,
                    'Price': price,
                    'Revenue': revenue,
                    'Fee': transaction_fee,
                    'Net_Revenue': net_revenue,
                    'Cash_After': self.current_capital + net_revenue
                })
                
                self.current_capital += net_revenue
                self.current_shares = 0
            
            else:
                action_taken = "HELD"
            
            # Record portfolio history
            self.portfolio_history.append({
                'Date': date,
                'Price': price,
                'Signal': signal,
                'Cash': self.current_capital,
                'Shares': self.current_shares,
                'Portfolio_Value': portfolio_value,
                'Action': action_taken
            })
        
        # Final portfolio value
        final_price = signals_df.iloc[-1]['Close']
        final_portfolio_value = self.current_capital + (self.current_shares * final_price)
        total_return = ((final_portfolio_value / self.initial_capital) - 1) * 100
        
        print(f"\n📊 FINAL TRADING RESULTS:")
        print(f"   💰 Initial Investment: ${self.initial_capital:,}")
        print(f"   💰 Final Portfolio Value: ${final_portfolio_value:,.2f}")
        print(f"   💰 Total Profit/Loss: ${final_portfolio_value - self.initial_capital:+,.2f}")
        print(f"   📊 Return Percentage: {total_return:+.2f}%")
        print(f"   💵 Final Cash: ${self.current_capital:,.2f}")
        print(f"   📈 Final Shares: {self.current_shares} (worth ${self.current_shares * final_price:,.2f})")
        print(f"   🔄 Total Trades: {len(self.trade_log)}")
        
        return final_portfolio_value
    
    def comprehensive_analysis(self, signals_df):
        """Comprehensive performance analysis"""
        print(f"\n📊 COMPREHENSIVE PERFORMANCE ANALYSIS")
        print(f"{'='*80}")
        
        # Calculate buy & hold benchmark
        start_price = signals_df.iloc[0]['Close']
        end_price = signals_df.iloc[-1]['Close']
        buy_hold_return = ((end_price / start_price) - 1) * 100
        buy_hold_value = self.initial_capital * (end_price / start_price)
        
        # Our strategy performance
        portfolio_df = pd.DataFrame(self.portfolio_history)
        final_value = portfolio_df.iloc[-1]['Portfolio_Value']
        strategy_return = ((final_value / self.initial_capital) - 1) * 100
        
        # Performance comparison
        alpha = strategy_return - buy_hold_return
        outperformance = final_value - buy_hold_value
        
        print(f"🎯 STRATEGY vs BENCHMARK:")
        print(f"   📈 Buy & Hold Return: {buy_hold_return:+.2f}% (${buy_hold_value:,.2f})")
        print(f"   🤖 AI Strategy Return: {strategy_return:+.2f}% (${final_value:,.2f})")
        print(f"   🏆 Alpha (Outperformance): {alpha:+.2f}%")
        print(f"   💰 Extra Money Made: ${outperformance:+,.2f}")
        
        # Trading statistics
        buy_trades = [t for t in self.trade_log if t['Action'] == 'BUY']
        sell_trades = [t for t in self.trade_log if t['Action'] == 'SELL']
        
        print(f"\n📈 TRADING STATISTICS:")
        print(f"   🔄 Total Trades: {len(self.trade_log)}")
        print(f"   📊 Buy Orders: {len(buy_trades)}")
        print(f"   📊 Sell Orders: {len(sell_trades)}")
        
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            avg_buy_price = np.mean([t['Price'] for t in buy_trades])
            avg_sell_price = np.mean([t['Price'] for t in sell_trades])
            total_fees = sum([t.get('Fee', 0) for t in self.trade_log])
            
            print(f"   💵 Average Buy Price: ${avg_buy_price:.2f}")
            print(f"   💵 Average Sell Price: ${avg_sell_price:.2f}")
            print(f"   💵 Total Transaction Fees: ${total_fees:.2f}")
            print(f"   📊 Average Trade Return: {((avg_sell_price / avg_buy_price) - 1) * 100:+.2f}%")
        
        # Risk metrics
        daily_returns = portfolio_df['Portfolio_Value'].pct_change().dropna() * 100
        if len(daily_returns) > 1:
            volatility = daily_returns.std()
            max_drawdown = self.calculate_max_drawdown(portfolio_df['Portfolio_Value'])
            
            print(f"\n⚠️ RISK ANALYSIS:")
            print(f"   📊 Daily Volatility: {volatility:.2f}%")
            print(f"   📉 Maximum Drawdown: {max_drawdown:.2f}%")
            
            if volatility > 0:
                sharpe_ratio = (strategy_return / 252 * len(daily_returns)) / (volatility * np.sqrt(252))
                print(f"   📊 Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Monthly breakdown
        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
        portfolio_df['Month'] = portfolio_df['Date'].dt.to_period('M')
        monthly_returns = portfolio_df.groupby('Month')['Portfolio_Value'].last().pct_change() * 100
        
        print(f"\n📅 MONTHLY PERFORMANCE:")
        for month, ret in monthly_returns.dropna().items():
            print(f"   {month}: {ret:+.2f}%")
        
        return {
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'alpha': alpha,
            'final_value': final_value,
            'outperformance': outperformance,
            'total_trades': len(self.trade_log)
        }
    
    def calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values.cummax()
        drawdown = (portfolio_values - peak) / peak * 100
        return drawdown.min()
    
    def create_detailed_visualization(self, signals_df):
        """Create detailed visualization"""
        print(f"\n📊 Creating detailed visualization...")
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 16))
        
        # Plot 1: Price with trading signals
        ax1 = axes[0, 0]
        ax1.plot(portfolio_df['Date'], portfolio_df['Price'], label='NVDA Price', color='black', linewidth=2)
        
        # Mark trades
        buy_dates = [t['Date'] for t in self.trade_log if t['Action'] == 'BUY']
        sell_dates = [t['Date'] for t in self.trade_log if t['Action'] == 'SELL']
        
        if buy_dates:
            buy_prices = [portfolio_df[portfolio_df['Date'] == d]['Price'].iloc[0] for d in buy_dates]
            ax1.scatter(buy_dates, buy_prices, color='green', marker='^', s=150, label='BUY', alpha=0.8, edgecolors='black')
        
        if sell_dates:
            sell_prices = [portfolio_df[portfolio_df['Date'] == d]['Price'].iloc[0] for d in sell_dates]
            ax1.scatter(sell_dates, sell_prices, color='red', marker='v', s=150, label='SELL', alpha=0.8, edgecolors='black')
        
        ax1.set_title('NVDA Price with AI Trading Signals', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio value comparison
        ax2 = axes[0, 1]
        ax2.plot(portfolio_df['Date'], portfolio_df['Portfolio_Value'], label='AI Trading Strategy', color='blue', linewidth=4)
        
        # Buy & hold comparison
        start_price = portfolio_df.iloc[0]['Price']
        buy_hold_values = [(price / start_price) * self.initial_capital for price in portfolio_df['Price']]
        ax2.plot(portfolio_df['Date'], buy_hold_values, label='Buy & Hold', color='orange', linewidth=3, linestyle='--')
        
        ax2.set_title('Portfolio Performance Comparison', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add performance annotation
        final_ai = portfolio_df.iloc[-1]['Portfolio_Value']
        final_bh = buy_hold_values[-1]
        performance_text = f'AI Strategy: ${final_ai:,.0f}\nBuy & Hold: ${final_bh:,.0f}\nOutperformance: ${final_ai - final_bh:+,.0f}'
        ax2.text(0.02, 0.98, performance_text, transform=ax2.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
                verticalalignment='top', fontweight='bold')
        
        # Plot 3: Cash and shares over time
        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()
        
        ax3.plot(portfolio_df['Date'], portfolio_df['Cash'], color='green', linewidth=3, label='Cash')
        ax3_twin.plot(portfolio_df['Date'], portfolio_df['Shares'], color='purple', linewidth=3, label='Shares Owned')
        
        ax3.set_title('Cash and Share Holdings Over Time', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Cash ($)', color='green', fontsize=12, fontweight='bold')
        ax3_twin.set_ylabel('Shares Owned', color='purple', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Signal distribution pie chart
        ax4 = axes[1, 1]
        signal_counts = signals_df['Signal'].value_counts()
        colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
        wedges, texts, autotexts = ax4.pie(signal_counts.values, labels=signal_counts.index, autopct='%1.1f%%',
                                          colors=[colors.get(sig, 'blue') for sig in signal_counts.index],
                                          startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        
        ax4.set_title('Trading Signal Distribution', fontsize=16, fontweight='bold')
        
        # Plot 5: Cumulative returns
        ax5 = axes[2, 0]
        ai_returns = (portfolio_df['Portfolio_Value'] / self.initial_capital - 1) * 100
        bh_returns = (np.array(buy_hold_values) / self.initial_capital - 1) * 100
        
        ax5.plot(portfolio_df['Date'], ai_returns, label='AI Strategy', color='blue', linewidth=4)
        ax5.plot(portfolio_df['Date'], bh_returns, label='Buy & Hold', color='orange', linewidth=3)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax5.set_title('Cumulative Returns Comparison', fontsize=16, fontweight='bold')
        ax5.set_ylabel('Return (%)')
        ax5.legend(fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Monthly returns bar chart
        ax6 = axes[2, 1]
        portfolio_df['Month'] = pd.to_datetime(portfolio_df['Date']).dt.to_period('M')
        monthly_portfolio = portfolio_df.groupby('Month')['Portfolio_Value'].last()
        monthly_returns = monthly_portfolio.pct_change() * 100
        
        colors = ['green' if x > 0 else 'red' for x in monthly_returns.dropna()]
        bars = ax6.bar(range(len(monthly_returns.dropna())), monthly_returns.dropna(), color=colors, alpha=0.7)
        
        ax6.set_title('Monthly Returns', fontsize=16, fontweight='bold')
        ax6.set_ylabel('Monthly Return (%)')
        ax6.set_xlabel('Month')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, monthly_returns.dropna()):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                    f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('fixed_data/results/improved_money_simulation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Detailed visualization saved to 'fixed_data/results/improved_money_simulation.png'")

def main():
    """Run the improved money making simulation"""
    print(f"💰 IMPROVED MONEY MAKING SIMULATION")
    print(f"{'='*90}")
    print(f"🎯 Investment: $10,000 from June 2024 → August 2025")
    print(f"🧠 Strategy: Proven AI trading signals with realistic execution")
    print(f"💵 Includes: Transaction costs (0.1% per trade)")
    print(f"📊 Benchmark: Buy & Hold comparison")
    
    simulator = ImprovedMoneySimulator(initial_capital=10000)
    
    # Load models
    if not simulator.load_models():
        print("❌ Cannot proceed without models")
        return
    
    # Fetch data
    data = simulator.fetch_historical_data('NVDA', '2024-06-01')
    if data is None:
        return
    
    # Calculate indicators
    data_with_indicators = simulator.calculate_technical_indicators(data)
    
    # Generate proven signals
    signals = simulator.generate_proven_signals(data_with_indicators)
    if signals is None:
        return
    
    # Simulate realistic trading
    final_value = simulator.simulate_realistic_trading(signals)
    
    # Comprehensive analysis
    performance = simulator.comprehensive_analysis(signals)
    
    # Create detailed visualization
    simulator.create_detailed_visualization(signals)
    
    # Final summary
    print(f"\n🎊 FINAL MONEY MAKING SUMMARY:")
    print(f"{'='*90}")
    print(f"💰 You invested: ${simulator.initial_capital:,} in June 2024")
    print(f"💰 You would have: ${final_value:,.2f} today (August 2025)")
    print(f"💰 Total profit: ${final_value - simulator.initial_capital:+,.2f}")
    print(f"📊 Your return: {performance['strategy_return']:+.2f}%")
    print(f"📊 Buy & hold return: {performance['buy_hold_return']:+.2f}%")
    print(f"🏆 You {'BEAT' if performance['alpha'] > 0 else 'LOST TO'} buy & hold by {abs(performance['alpha']):.2f}%")
    
    if performance['outperformance'] > 0:
        print(f"🎉 Congratulations! You made an extra ${performance['outperformance']:,.2f} vs buy & hold!")
    else:
        print(f"😔 You lost ${abs(performance['outperformance']):,.2f} vs buy & hold")
    
    print(f"\n📊 Check 'fixed_data/results/improved_money_simulation.png' for detailed charts")
    
    return simulator, signals, performance

if __name__ == "__main__":
    simulator, signals, performance = main()
