#!/usr/bin/env python3
"""
Tesla Trading Signal Backtest
Test the actual performance of our AI signals vs buy-and-hold
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def backtest_tesla_signals():
    """Backtest the Tesla signals to see actual performance"""
    print("🧪 BACKTESTING Tesla AI Signals...")
    print("💰 Testing actual trading performance vs buy-and-hold")
    
    # Download 4 months of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)
    
    stock_data = yf.download("TSLA", start=start_date, end=end_date)
    
    if stock_data.empty:
        print("❌ Failed to download data")
        return None
    
    print(f"✅ Downloaded {len(stock_data)} days of TSLA data")
    
    # Calculate technical indicators
    stock_data['MA5'] = stock_data['Close'].rolling(5).mean()
    stock_data['MA10'] = stock_data['Close'].rolling(10).mean()
    stock_data['MA20'] = stock_data['Close'].rolling(20).mean()
    
    # RSI calculation
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    stock_data['RSI'] = calculate_rsi(stock_data['Close'])
    
    # Generate signals (same logic as improved version)
    signals = []
    
    for i in range(30, len(stock_data)):
        date = stock_data.index[i]
        price = float(stock_data['Close'].iloc[i])
        
        # Multi-timeframe analysis
        recent_5d = stock_data['Close'].iloc[i-5:i]
        recent_10d = stock_data['Close'].iloc[i-10:i]
        recent_20d = stock_data['Close'].iloc[i-20:i]
        
        # Trend analysis
        trend_5d = (price - float(recent_5d.mean())) / float(recent_5d.mean())
        trend_10d = (price - float(recent_10d.mean())) / float(recent_10d.mean())
        trend_20d = (price - float(recent_20d.mean())) / float(recent_20d.mean())
        
        # Moving average signals
        ma5 = float(stock_data['MA5'].iloc[i]) if not pd.isna(stock_data['MA5'].iloc[i]) else price
        ma10 = float(stock_data['MA10'].iloc[i]) if not pd.isna(stock_data['MA10'].iloc[i]) else price
        ma20 = float(stock_data['MA20'].iloc[i]) if not pd.isna(stock_data['MA20'].iloc[i]) else price
        
        # RSI signal
        rsi = float(stock_data['RSI'].iloc[i]) if not pd.isna(stock_data['RSI'].iloc[i]) else 50
        
        # Volatility
        volatility = float(recent_10d.std()) / float(recent_10d.mean()) if float(recent_10d.mean()) > 0 else 0
        
        # Volume analysis
        try:
            recent_volume = float(stock_data['Volume'].iloc[i-10:i].mean())
            current_volume = float(stock_data['Volume'].iloc[i])
            volume_ratio = current_volume / recent_volume if recent_volume > 0 else 1
        except:
            volume_ratio = 1
        
        # SIGNAL LOGIC (same as improved version)
        # Strong SELL conditions
        if (trend_5d < -0.03 and trend_10d < -0.02) or \
           (price < ma5 < ma10 < ma20) or \
           (trend_20d < -0.05) or \
           (rsi > 70 and trend_5d < -0.02) or \
           (volatility > 0.08 and trend_10d < -0.03):
            signal = 'SELL'
        # Moderate SELL conditions
        elif (trend_5d < -0.02 and price < ma10) or \
             (trend_10d < -0.03) or \
             (rsi > 75) or \
             (price < ma5 and trend_5d < -0.015):
            signal = 'SELL'
        # Strong BUY conditions
        elif (trend_5d > 0.04 and trend_10d > 0.02 and volume_ratio > 1.3) or \
             (price > ma5 > ma10 > ma20 and trend_5d > 0.03) or \
             (rsi < 30 and trend_5d > 0.02):
            signal = 'BUY'
        # Moderate BUY conditions  
        elif (trend_5d > 0.03 and price > ma10) or \
             (trend_10d > 0.03 and volume_ratio > 1.1) or \
             (price > ma5 and trend_5d > 0.025):
            signal = 'BUY'
        else:
            signal = 'HOLD'
        
        signals.append({
            'date': date,
            'price': price,
            'signal': signal,
            'rsi': rsi,
            'trend_5d': trend_5d,
            'trend_10d': trend_10d
        })
    
    # Convert to DataFrame for easier analysis
    signals_df = pd.DataFrame(signals)
    
    print(f"🤖 Generated {len(signals_df)} daily signals")
    
    # BACKTEST SIMULATION
    print("\n💰 RUNNING BACKTEST SIMULATION...")
    print("=" * 40)
    
    # Starting conditions
    initial_cash = 10000
    cash = initial_cash
    shares = 0
    position = None  # 'LONG', 'SHORT', or None
    
    # Track performance
    portfolio_values = []
    trades = []
    
    for i, row in signals_df.iterrows():
        date = row['date']
        price = row['price']
        signal = row['signal']
        
        # Calculate current portfolio value
        portfolio_value = cash + (shares * price)
        portfolio_values.append({
            'date': date,
            'value': portfolio_value,
            'price': price,
            'signal': signal,
            'cash': cash,
            'shares': shares
        })
        
        # Execute trades based on signals
        if signal == 'BUY' and position != 'LONG':
            # Buy signal - go long
            if position == 'SHORT':
                # Close short position first
                cash += shares * price  # Buy back shares
                shares = 0
            
            # Buy shares
            shares = cash / price
            cash = 0
            position = 'LONG'
            
            trades.append({
                'date': date,
                'action': 'BUY',
                'price': price,
                'shares': shares,
                'value': shares * price
            })
            
        elif signal == 'SELL' and position != 'SHORT':
            # Sell signal - go short (or close long)
            if position == 'LONG':
                # Close long position
                cash = shares * price
                shares = 0
            
            # Go short (for simplicity, we'll just hold cash)
            position = 'SHORT'
            
            trades.append({
                'date': date,
                'action': 'SELL',
                'price': price,
                'shares': shares,
                'value': cash
            })
    
    # Final portfolio value
    final_price = signals_df.iloc[-1]['price']
    final_portfolio_value = cash + (shares * final_price)
    
    # Buy-and-hold comparison
    start_price = float(stock_data['Close'].iloc[30])  # When signals started
    end_price = float(stock_data['Close'].iloc[-1])
    buy_hold_return = ((end_price - start_price) / start_price) * 100
    buy_hold_final_value = initial_cash * (end_price / start_price)
    
    # AI strategy performance
    ai_return = ((final_portfolio_value - initial_cash) / initial_cash) * 100
    
    # RESULTS
    print(f"\n📊 BACKTEST RESULTS:")
    print("=" * 30)
    print(f"💰 Initial Investment: ${initial_cash:,.2f}")
    print(f"📈 Start Price: ${start_price:.2f}")
    print(f"📉 End Price: ${end_price:.2f}")
    print()
    print(f"🤖 AI STRATEGY:")
    print(f"   Final Value: ${final_portfolio_value:,.2f}")
    print(f"   Return: {ai_return:+.1f}%")
    print(f"   Total Trades: {len(trades)}")
    print()
    print(f"📈 BUY & HOLD:")
    print(f"   Final Value: ${buy_hold_final_value:,.2f}")
    print(f"   Return: {buy_hold_return:+.1f}%")
    print()
    
    # Performance comparison
    outperformance = ai_return - buy_hold_return
    print(f"⚖️  PERFORMANCE COMPARISON:")
    if outperformance > 0:
        print(f"   🎯 AI Strategy OUTPERFORMED by {outperformance:+.1f}%")
    else:
        print(f"   ❌ AI Strategy UNDERPERFORMED by {outperformance:.1f}%")
    
    # Signal analysis
    buy_signals = len([t for t in trades if t['action'] == 'BUY'])
    sell_signals = len([t for t in trades if t['action'] == 'SELL'])
    
    print(f"\n📊 SIGNAL BREAKDOWN:")
    print(f"   🟢 BUY Signals: {buy_signals}")
    print(f"   🔴 SELL Signals: {sell_signals}")
    
    # Show recent trades
    print(f"\n💼 RECENT TRADES:")
    for trade in trades[-5:]:
        action_emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
        print(f"   {action_emoji} {trade['date'].date()}: {trade['action']} @ ${trade['price']:.2f}")
    
    # Create performance chart
    portfolio_df = pd.DataFrame(portfolio_values)
    
    plt.figure(figsize=(15, 8))
    
    # Plot portfolio value vs buy-and-hold
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_df['date'], portfolio_df['value'], linewidth=2, color='blue', label='AI Strategy')
    
    # Calculate buy-and-hold values for comparison
    buy_hold_values = [initial_cash * (price / start_price) for price in portfolio_df['price']]
    plt.plot(portfolio_df['date'], buy_hold_values, linewidth=2, color='gray', linestyle='--', label='Buy & Hold')
    
    plt.title('Tesla Trading Strategy Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot signals
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_df['date'], portfolio_df['price'], linewidth=2, color='black', alpha=0.7, label='TSLA Price')
    
    # Mark trades
    for trade in trades:
        color = 'green' if trade['action'] == 'BUY' else 'red'
        marker = '^' if trade['action'] == 'BUY' else 'v'
        plt.scatter(trade['date'], trade['price'], color=color, marker=marker, s=100, zorder=5)
    
    plt.title('Trading Signals on TSLA Price', fontsize=14, fontweight='bold')
    plt.ylabel('Price ($)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tesla_backtest_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Backtest chart saved as 'tesla_backtest_results.png'")
    
    return {
        'ai_return': ai_return,
        'buy_hold_return': buy_hold_return,
        'outperformance': outperformance,
        'trades': trades,
        'final_value': final_portfolio_value
    }

if __name__ == "__main__":
    print("🧪 TESLA AI STRATEGY BACKTEST")
    print("=" * 40)
    print("Testing if our 'improved' signals actually work...")
    
    results = backtest_tesla_signals()
    
    if results:
        print(f"\n🎯 FINAL VERDICT:")
        if results['outperformance'] > 5:
            print(f"   ✅ Strategy is GOOD - beats buy-and-hold!")
        elif results['outperformance'] > 0:
            print(f"   🟡 Strategy is OK - slightly beats buy-and-hold")
        else:
            print(f"   ❌ Strategy is BAD - loses to buy-and-hold!")
            print(f"   💡 Need to improve signal logic!")
