#!/usr/bin/env python3
"""
Money Making Simulator
Real trading simulation: $10,000 invested June 2024 -> August 2025
Following actual trading signals: BUY, HOLD, SELL
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

class MoneyMakingSimulator:
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
            # Get data from June 1, 2024 to today
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
        """Calculate technical indicators for signal generation"""
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
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # On Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # VWAP
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # Price-based features
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
    
    def generate_trading_signals(self, data):
        """Generate BUY/HOLD/SELL signals using the trained models"""
        print("🧠 Generating trading signals...")
        
        if not self.models:
            print("❌ No models loaded")
            return None
        
        # Prepare features
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
        
        # Create feature matrix
        X = data[feature_columns].copy()
        X_clean = X.dropna()
        
        if len(X_clean) == 0:
            print("❌ No valid feature data")
            return None
        
        print(f"📊 Generated features for {len(X_clean)} trading days")
        
        # Get predictions from all models
        signals = []
        
        for date_idx in X_clean.index:
            try:
                # Get current day features
                features = X_clean.loc[date_idx:date_idx].values
                
                if len(features) == 0:
                    signals.append('HOLD')
                    continue
                
                # Get predictions from all models
                predictions = []
                current_price = data.loc[date_idx, 'Close']
                
                for model_name, model in self.models.items():
                    try:
                        pred = model.predict(features.reshape(1, -1))[0]
                        predictions.append(pred)
                    except:
                        predictions.append(current_price)
                
                # Ensemble prediction (average)
                ensemble_pred = np.mean(predictions)
                
                # Generate signal based on prediction vs current price
                price_change_pred = (ensemble_pred - current_price) / current_price * 100
                
                # Enhanced signal logic with technical indicators
                rsi = data.loc[date_idx, 'RSI'] if 'RSI' in data.columns else 50
                volume_ratio = data.loc[date_idx, 'Volume_Ratio'] if 'Volume_Ratio' in data.columns else 1
                sma_5 = data.loc[date_idx, 'SMA_5'] if 'SMA_5' in data.columns else current_price
                sma_20 = data.loc[date_idx, 'SMA_20'] if 'SMA_20' in data.columns else current_price
                
                # More aggressive signal generation
                buy_score = 0
                sell_score = 0
                
                # Price prediction signals
                if price_change_pred > 0.5:
                    buy_score += 2
                elif price_change_pred < -0.5:
                    sell_score += 2
                
                # RSI signals
                if rsi < 35:  # Oversold
                    buy_score += 2
                elif rsi > 65:  # Overbought
                    sell_score += 2
                elif rsi < 50:
                    buy_score += 1
                else:
                    sell_score += 1
                
                # Trend signals
                if current_price > sma_5 > sma_20:  # Uptrend
                    buy_score += 2
                elif current_price < sma_5 < sma_20:  # Downtrend
                    sell_score += 2
                
                # Volume signals
                if volume_ratio > 1.5:  # High volume
                    if buy_score > sell_score:
                        buy_score += 1
                    else:
                        sell_score += 1
                
                # Generate final signal
                if buy_score >= 3 and buy_score > sell_score:
                    signal = 'BUY'
                elif sell_score >= 3 and sell_score > buy_score:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
                
                signals.append(signal)
                
            except Exception as e:
                signals.append('HOLD')
        
        # Create signals dataframe
        signals_df = pd.DataFrame({
            'Date': X_clean.index,
            'Close': data.loc[X_clean.index, 'Close'],
            'Signal': signals
        })
        
        signal_counts = signals_df['Signal'].value_counts()
        print(f"📊 Signal distribution:")
        for signal, count in signal_counts.items():
            print(f"   {signal}: {count} days")
        
        return signals_df
    
    def simulate_trading(self, signals_df):
        """Simulate actual trading based on signals"""
        print(f"\n💰 SIMULATING TRADING WITH ${self.initial_capital:,} STARTING CAPITAL")
        print(f"{'='*80}")
        
        self.current_capital = self.initial_capital
        self.current_shares = 0
        self.trade_log = []
        self.portfolio_history = []
        
        for idx, row in signals_df.iterrows():
            date = row['Date']
            price = row['Close']
            signal = row['Signal']
            
            # Calculate current portfolio value
            portfolio_value = self.current_capital + (self.current_shares * price)
            
            action_taken = ""
            
            if signal == 'BUY' and self.current_capital > 100:  # Keep some cash for fees
                # Buy as many shares as possible
                shares_to_buy = int(self.current_capital / price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    self.current_capital -= cost
                    self.current_shares += shares_to_buy
                    action_taken = f"BOUGHT {shares_to_buy} shares at ${price:.2f}"
                    
                    self.trade_log.append({
                        'Date': date,
                        'Action': 'BUY',
                        'Shares': shares_to_buy,
                        'Price': price,
                        'Cost': cost,
                        'Cash_Remaining': self.current_capital,
                        'Total_Shares': self.current_shares,
                        'Portfolio_Value': portfolio_value
                    })
            
            elif signal == 'SELL' and self.current_shares > 0:
                # Sell all shares
                revenue = self.current_shares * price
                action_taken = f"SOLD {self.current_shares} shares at ${price:.2f}"
                
                self.trade_log.append({
                    'Date': date,
                    'Action': 'SELL',
                    'Shares': self.current_shares,
                    'Price': price,
                    'Revenue': revenue,
                    'Cash_After': self.current_capital + revenue,
                    'Total_Shares': 0,
                    'Portfolio_Value': portfolio_value
                })
                
                self.current_capital += revenue
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
        
        # Final portfolio calculation
        final_price = signals_df.iloc[-1]['Close']
        final_portfolio_value = self.current_capital + (self.current_shares * final_price)
        
        print(f"\n📊 TRADING SIMULATION RESULTS:")
        print(f"   Initial Investment: ${self.initial_capital:,}")
        print(f"   Final Portfolio Value: ${final_portfolio_value:,.2f}")
        print(f"   Total Return: ${final_portfolio_value - self.initial_capital:,.2f}")
        print(f"   Return Percentage: {((final_portfolio_value / self.initial_capital) - 1) * 100:+.2f}%")
        print(f"   Final Cash: ${self.current_capital:,.2f}")
        print(f"   Final Shares: {self.current_shares}")
        print(f"   Total Trades: {len(self.trade_log)}")
        
        return final_portfolio_value
    
    def analyze_performance(self, signals_df):
        """Analyze trading performance vs buy & hold"""
        print(f"\n📊 PERFORMANCE ANALYSIS")
        print(f"{'='*80}")
        
        # Calculate buy & hold performance
        start_price = signals_df.iloc[0]['Close']
        end_price = signals_df.iloc[-1]['Close']
        buy_hold_return = ((end_price / start_price) - 1) * 100
        buy_hold_value = self.initial_capital * (end_price / start_price)
        
        # Calculate our strategy performance
        portfolio_df = pd.DataFrame(self.portfolio_history)
        final_value = portfolio_df.iloc[-1]['Portfolio_Value']
        strategy_return = ((final_value / self.initial_capital) - 1) * 100
        
        print(f"🔥 STRATEGY COMPARISON:")
        print(f"   📈 Buy & Hold Strategy:")
        print(f"      Final Value: ${buy_hold_value:,.2f}")
        print(f"      Return: {buy_hold_return:+.2f}%")
        print(f"   🤖 AI Trading Strategy:")
        print(f"      Final Value: ${final_value:,.2f}")
        print(f"      Return: {strategy_return:+.2f}%")
        print(f"   🎯 Alpha (Outperformance): {strategy_return - buy_hold_return:+.2f}%")
        print(f"   💰 Extra Profit: ${final_value - buy_hold_value:+,.2f}")
        
        # Trading activity analysis
        buy_trades = [t for t in self.trade_log if t['Action'] == 'BUY']
        sell_trades = [t for t in self.trade_log if t['Action'] == 'SELL']
        
        print(f"\n📊 TRADING ACTIVITY:")
        print(f"   Total Buy Orders: {len(buy_trades)}")
        print(f"   Total Sell Orders: {len(sell_trades)}")
        
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            avg_buy_price = np.mean([t['Price'] for t in buy_trades])
            avg_sell_price = np.mean([t['Price'] for t in sell_trades])
            print(f"   Average Buy Price: ${avg_buy_price:.2f}")
            print(f"   Average Sell Price: ${avg_sell_price:.2f}")
            print(f"   Average Trade Profit: {((avg_sell_price / avg_buy_price) - 1) * 100:+.2f}%")
        
        # Risk analysis
        daily_returns = portfolio_df['Portfolio_Value'].pct_change().dropna() * 100
        if len(daily_returns) > 1:
            volatility = daily_returns.std()
            sharpe_ratio = (strategy_return / len(daily_returns) * 252) / (volatility * np.sqrt(252)) if volatility > 0 else 0
            max_drawdown = self.calculate_max_drawdown(portfolio_df['Portfolio_Value'])
            
            print(f"\n📊 RISK METRICS:")
            print(f"   Daily Volatility: {volatility:.2f}%")
            print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"   Max Drawdown: {max_drawdown:.2f}%")
        
        return {
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'alpha': strategy_return - buy_hold_return,
            'final_value': final_value,
            'buy_hold_value': buy_hold_value,
            'total_trades': len(self.trade_log)
        }
    
    def calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values.cummax()
        drawdown = (portfolio_values - peak) / peak * 100
        return drawdown.min()
    
    def create_visualization(self, signals_df):
        """Create comprehensive visualization"""
        print(f"\n📊 Creating visualization...")
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        # Create comprehensive plot
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        
        # Plot 1: Price with signals
        ax1 = axes[0, 0]
        ax1.plot(portfolio_df['Date'], portfolio_df['Price'], label='NVDA Price', color='black', linewidth=2)
        
        # Mark buy signals
        buy_signals = portfolio_df[portfolio_df['Signal'] == 'BUY']
        ax1.scatter(buy_signals['Date'], buy_signals['Price'], color='green', marker='^', s=100, label='BUY Signal', alpha=0.8)
        
        # Mark sell signals
        sell_signals = portfolio_df[portfolio_df['Signal'] == 'SELL']
        ax1.scatter(sell_signals['Date'], sell_signals['Price'], color='red', marker='v', s=100, label='SELL Signal', alpha=0.8)
        
        ax1.set_title('NVDA Price with Trading Signals', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Portfolio value over time
        ax2 = axes[0, 1]
        ax2.plot(portfolio_df['Date'], portfolio_df['Portfolio_Value'], label='AI Strategy', color='blue', linewidth=3)
        
        # Calculate buy & hold for comparison
        start_price = portfolio_df.iloc[0]['Price']
        buy_hold_values = [(price / start_price) * self.initial_capital for price in portfolio_df['Price']]
        ax2.plot(portfolio_df['Date'], buy_hold_values, label='Buy & Hold', color='orange', linewidth=2, linestyle='--')
        
        ax2.set_title('Portfolio Value Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add final values annotation
        final_ai = portfolio_df.iloc[-1]['Portfolio_Value']
        final_bh = buy_hold_values[-1]
        ax2.annotate(f'AI: ${final_ai:,.0f}\nB&H: ${final_bh:,.0f}', 
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=10, fontweight='bold', verticalalignment='top')
        
        # Plot 3: Cash vs Shares over time
        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()
        
        ax3.plot(portfolio_df['Date'], portfolio_df['Cash'], color='green', linewidth=2, label='Cash')
        ax3_twin.plot(portfolio_df['Date'], portfolio_df['Shares'], color='purple', linewidth=2, label='Shares')
        
        ax3.set_title('Cash and Shares Over Time', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cash ($)', color='green')
        ax3_twin.set_ylabel('Shares', color='purple')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Signal distribution
        ax4 = axes[1, 1]
        signal_counts = portfolio_df['Signal'].value_counts()
        colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
        bars = ax4.bar(signal_counts.index, signal_counts.values, 
                      color=[colors.get(sig, 'blue') for sig in signal_counts.index])
        
        ax4.set_title('Trading Signal Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Signal Type')
        ax4.set_ylabel('Number of Days')
        
        # Add value labels on bars
        for bar, value in zip(bars, signal_counts.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Plot 5: Daily returns distribution
        ax5 = axes[2, 0]
        daily_returns = portfolio_df['Portfolio_Value'].pct_change().dropna() * 100
        ax5.hist(daily_returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.axvline(daily_returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {daily_returns.mean():.2f}%')
        ax5.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Daily Return (%)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Cumulative returns comparison
        ax6 = axes[2, 1]
        ai_returns = (portfolio_df['Portfolio_Value'] / self.initial_capital - 1) * 100
        bh_returns = (np.array(buy_hold_values) / self.initial_capital - 1) * 100
        
        ax6.plot(portfolio_df['Date'], ai_returns, label='AI Strategy', color='blue', linewidth=3)
        ax6.plot(portfolio_df['Date'], bh_returns, label='Buy & Hold', color='orange', linewidth=2)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax6.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Cumulative Return (%)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('fixed_data/results/money_making_simulation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved to 'fixed_data/results/money_making_simulation.png'")

def main():
    """Run the money making simulation"""
    print(f"💰 MONEY MAKING SIMULATION")
    print(f"{'='*80}")
    print(f"🎯 Scenario: $10,000 invested June 2024 → August 2025")
    print(f"📊 Strategy: Follow AI trading signals (BUY/HOLD/SELL)")
    print(f"📈 Compare vs Buy & Hold")
    
    simulator = MoneyMakingSimulator(initial_capital=10000)
    
    # Load models
    if not simulator.load_models():
        print("❌ Cannot proceed without models")
        return
    
    # Fetch data from June 2024
    data = simulator.fetch_historical_data('NVDA', '2024-06-01')
    if data is None:
        return
    
    # Calculate indicators
    data_with_indicators = simulator.calculate_technical_indicators(data)
    
    # Generate signals
    signals = simulator.generate_trading_signals(data_with_indicators)
    if signals is None:
        return
    
    # Simulate trading
    final_value = simulator.simulate_trading(signals)
    
    # Analyze performance
    performance = simulator.analyze_performance(signals)
    
    # Create visualization
    simulator.create_visualization(signals)
    
    print(f"\n✅ SIMULATION COMPLETE!")
    print(f"💰 Final Result: ${final_value:,.2f}")
    print(f"📊 Check 'fixed_data/results/money_making_simulation.png' for charts")
    
    return simulator, signals, performance

if __name__ == "__main__":
    simulator, signals, performance = main()
