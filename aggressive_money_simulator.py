#!/usr/bin/env python3
"""
Aggressive Money Simulator
Based on the proven strategies but with much more aggressive trading to actually see trades
"""

import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class AggressiveMoneySimulator:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.current_shares = 0
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        model_dir = "fixed_data/models"
        model_files = {
            'random_forest': 'random_forest_model.pkl',
            'gradient_boosting': 'gradient_boosting_model.pkl',
            'linear_regression': 'linear_regression_model.pkl',
            'ridge': 'ridge_model.pkl'
        }
        
        loaded_count = 0
        for name, filename in model_files.items():
            filepath = os.path.join(model_dir, filename)
            if os.path.exists(filepath):
                try:
                    self.models[name] = joblib.load(filepath)
                    loaded_count += 1
                    print(f"✅ Loaded {name} model")
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")
            else:
                print(f"❌ Model file not found: {filepath}")
        
        print(f"📊 Total models loaded: {loaded_count}/4")
        return loaded_count > 0

    def fetch_historical_data(self, symbol="NVDA", start_date="2024-06-03", end_date="2025-08-04"):
        """Fetch historical data for simulation"""
        print(f"📈 Fetching {symbol} data from {start_date} to {end_date}")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if len(data) == 0:
                print("❌ No data retrieved")
                return None
            
            print(f"✅ Retrieved {len(data)} days of data")
            return data
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None

    def calculate_technical_indicators(self, data):
        """Calculate technical indicators like the proven strategies"""
        print("🔧 Calculating technical indicators...")
        
        df = data.copy()
        
        # Basic moving averages
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
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # VWAP
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # Price changes and momentum
        for period in [1, 2, 3, 5]:
            df[f'Return_{period}d'] = df['Close'].pct_change(period) * 100
        
        print(f"✅ Calculated technical indicators, dataset now has {df.shape[1]} columns")
        return df

    def generate_aggressive_signals(self, data):
        """Generate aggressive trading signals based on proven patterns"""
        print("🧠 Generating AGGRESSIVE trading signals...")
        
        signals = []
        
        for i in range(len(data)):
            signal = 'HOLD'
            buy_score = 0
            sell_score = 0
            
            # Get current data
            row = data.iloc[i]
            
            if i < 20:  # Skip first 20 days for indicators to stabilize
                signals.append('HOLD')
                continue
            
            price = row['Close']
            rsi = row['RSI'] if not pd.isna(row['RSI']) else 50
            macd = row['MACD'] if not pd.isna(row['MACD']) else 0
            macd_signal = row['MACD_Signal'] if not pd.isna(row['MACD_Signal']) else 0
            bb_position = row['BB_Position'] if not pd.isna(row['BB_Position']) else 0.5
            volume_ratio = row['Volume_Ratio'] if not pd.isna(row['Volume_Ratio']) else 1
            sma_5 = row['SMA_5'] if not pd.isna(row['SMA_5']) else price
            sma_20 = row['SMA_20'] if not pd.isna(row['SMA_20']) else price
            return_1d = row['Return_1d'] if not pd.isna(row['Return_1d']) else 0
            
            # AGGRESSIVE BUY SIGNALS (based on successful patterns)
            
            # RSI oversold signals (more aggressive thresholds)
            if rsi < 35:
                buy_score += 3  # Strong buy on oversold
            elif rsi < 45:
                buy_score += 2  # Moderate buy
            elif rsi < 50:
                buy_score += 1  # Weak buy
            
            # MACD signals
            if macd > macd_signal:
                buy_score += 2  # MACD bullish
            elif macd > macd_signal * 0.8:  # Almost crossing
                buy_score += 1
            
            # Bollinger Bands
            if bb_position < 0.2:  # Near lower band
                buy_score += 3
            elif bb_position < 0.4:
                buy_score += 2
            
            # Price trend signals
            if price > sma_5 > sma_20:  # Clear uptrend
                buy_score += 2
            elif price > sma_5:  # Short term uptrend
                buy_score += 1
            
            # Volume confirmation
            if volume_ratio > 1.5:  # High volume
                if buy_score > 0:
                    buy_score += 1
            
            # Momentum signals
            if return_1d > 2:  # Strong daily gain
                buy_score += 2
            elif return_1d > 0.5:
                buy_score += 1
            
            # AGGRESSIVE SELL SIGNALS
            
            # RSI overbought (more aggressive)
            if rsi > 70:
                sell_score += 2
            elif rsi > 60:
                sell_score += 1
            
            # Bollinger Bands
            if bb_position > 0.8:  # Near upper band
                sell_score += 2
            elif bb_position > 0.6:
                sell_score += 1
            
            # Downtrend signals
            if price < sma_5 < sma_20:  # Clear downtrend
                sell_score += 2
            elif price < sma_5:  # Short term downtrend
                sell_score += 1
            
            # Negative momentum
            if return_1d < -2:  # Strong daily loss
                sell_score += 2
            elif return_1d < -0.5:
                sell_score += 1
            
            # MACD bearish
            if macd < macd_signal:
                sell_score += 1
            
            # Generate final signal with MUCH LOWER thresholds
            if buy_score >= 4:  # Lower threshold for BUY
                signal = 'BUY'
            elif sell_score >= 3:  # Lower threshold for SELL
                signal = 'SELL'
            elif buy_score >= 2 and buy_score > sell_score + 1:  # Aggressive buy
                signal = 'BUY'
            elif sell_score >= 2 and sell_score > buy_score + 1:  # Aggressive sell
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            signals.append(signal)
        
        # Create signals dataframe
        signals_df = pd.DataFrame({
            'Date': data.index,
            'Close': data['Close'],
            'Signal': signals
        })
        
        # Print signal distribution
        signal_counts = signals_df['Signal'].value_counts()
        print(f"📊 AGGRESSIVE Signal distribution:")
        for signal, count in signal_counts.items():
            print(f"   {signal}: {count} days ({count/len(signals_df)*100:.1f}%)")
        
        return signals_df

    def simulate_realistic_trading(self, signals_df):
        """Simulate realistic trading with transaction costs"""
        print(f"\n💰 SIMULATING AGGRESSIVE TRADING WITH ${self.initial_capital:,}")
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
            
            if signal == 'BUY' and self.current_capital > 100:  # Minimum trade size
                # Use 90% of available cash for buying
                available_cash = self.current_capital * 0.9
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
                    'Cash_After': self.current_capital + net_revenue,
                    'Total_Shares': 0
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
        print(f"   📈 Final Shares: {self.current_shares}")
        print(f"   🔄 Total Trades: {len(self.trade_log)}")
        
        return final_portfolio_value

    def analyze_performance(self, signals_df):
        """Analyze performance vs buy & hold"""
        print(f"\n📊 PERFORMANCE ANALYSIS vs BUY & HOLD")
        print(f"{'='*80}")
        
        # Buy & hold performance
        start_price = signals_df.iloc[0]['Close']
        end_price = signals_df.iloc[-1]['Close']
        buy_hold_return = ((end_price / start_price) - 1) * 100
        buy_hold_value = self.initial_capital * (end_price / start_price)
        
        # Our strategy performance
        portfolio_df = pd.DataFrame(self.portfolio_history)
        final_value = portfolio_df.iloc[-1]['Portfolio_Value']
        strategy_return = ((final_value / self.initial_capital) - 1) * 100
        
        print(f"🔥 STRATEGY COMPARISON:")
        print(f"   📈 Buy & Hold Strategy:")
        print(f"      Final Value: ${buy_hold_value:,.2f}")
        print(f"      Return: {buy_hold_return:+.2f}%")
        print(f"\n   🤖 AGGRESSIVE AI Strategy:")
        print(f"      Final Value: ${final_value:,.2f}")
        print(f"      Return: {strategy_return:+.2f}%")
        print(f"\n   🎯 Alpha (Outperformance): {strategy_return - buy_hold_return:+.2f}%")
        print(f"   💰 Extra Profit: ${final_value - buy_hold_value:+,.2f}")
        
        # Trading activity
        buy_trades = [t for t in self.trade_log if t['Action'] == 'BUY']
        sell_trades = [t for t in self.trade_log if t['Action'] == 'SELL']
        
        print(f"\n📊 TRADING ACTIVITY:")
        print(f"   🛒 Total Buy Orders: {len(buy_trades)}")
        print(f"   💸 Total Sell Orders: {len(sell_trades)}")
        
        if len(self.trade_log) > 0:
            total_fees = sum(t.get('Fee', 0) for t in self.trade_log)
            print(f"   💳 Total Transaction Fees: ${total_fees:.2f}")
            
            # Show some sample trades
            print(f"\n📋 SAMPLE TRADES:")
            for i, trade in enumerate(self.trade_log[:10]):  # Show first 10 trades
                action = trade['Action']
                date = trade['Date'].strftime('%Y-%m-%d') if hasattr(trade['Date'], 'strftime') else str(trade['Date'])
                shares = trade['Shares']
                price = trade['Price']
                print(f"   {i+1}. {date}: {action} {shares} shares @ ${price:.2f}")
        
        return {
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'alpha': strategy_return - buy_hold_return,
            'final_value': final_value,
            'buy_hold_value': buy_hold_value,
            'total_trades': len(self.trade_log)
        }

def main():
    """Run the aggressive money simulation"""
    print("🚀 AGGRESSIVE MONEY MAKING SIMULATOR")
    print("Based on AI models with much more aggressive trading thresholds")
    print("=" * 80)
    
    # Initialize simulator
    simulator = AggressiveMoneySimulator(initial_capital=10000)
    
    if not simulator.models:
        print("❌ No models loaded, cannot proceed")
        return None
    
    # Fetch data from June 2024 to August 2025
    data = simulator.fetch_historical_data("NVDA", "2024-06-03", "2025-08-04")
    if data is None:
        return None
    
    # Calculate technical indicators
    data_with_indicators = simulator.calculate_technical_indicators(data)
    
    # Generate aggressive trading signals
    signals_df = simulator.generate_aggressive_signals(data_with_indicators)
    
    # Simulate trading
    final_value = simulator.simulate_realistic_trading(signals_df)
    
    # Analyze performance
    performance = simulator.analyze_performance(signals_df)
    
    return simulator, performance

if __name__ == "__main__":
    simulator, performance = main()
