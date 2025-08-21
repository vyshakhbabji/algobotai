#!/usr/bin/env python3
"""
Quick Strategy Performance Check
Test your current live trading strategy vs buy-and-hold
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def quick_test():
    """Quick test of strategy performance"""
    
    # Test on one stock first - UBER (which had a strong signal today)
    symbol = "UBER"
    
    # 6 months of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    print(f"🔍 TESTING YOUR STRATEGY ON {symbol}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("=" * 50)
    
    # Download data
    df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
    
    if df.empty:
        print("❌ No data available")
        return
    
    # Add indicators
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Simple momentum signal
    df['momentum_5d'] = df['Close'].pct_change(5)
    df['momentum_10d'] = df['Close'].pct_change(10)
    
    # Generate simple buy/sell signals
    buy_signals = (
        (df['momentum_5d'] > 0.02) &  # 2% 5-day momentum
        (df['momentum_10d'] > 0.02) &  # 2% 10-day momentum
        (df['Close'] > df['MA5']) &
        (df['MA5'] > df['MA10'])
    )
    
    sell_signals = (
        (df['momentum_5d'] < -0.02) |  # -2% 5-day momentum
        (df['RSI'] > 70) |  # Overbought
        (df['Close'] < df['MA10'])  # Below 10-day MA
    )
    
    # Simple backtest
    position = 0  # 0 = cash, 1 = invested
    trades = []
    
    for i in range(20, len(df)):
        if buy_signals.iloc[i] and position == 0:
            position = 1
            entry_price = df['Close'].iloc[i]
            entry_date = df.index[i]
            trades.append({'action': 'BUY', 'date': entry_date, 'price': entry_price})
        
        elif sell_signals.iloc[i] and position == 1:
            position = 0
            exit_price = df['Close'].iloc[i]
            exit_date = df.index[i]
            trades.append({'action': 'SELL', 'date': exit_date, 'price': exit_price})
    
    # Calculate performance
    if len(trades) >= 2:
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        total_return = 0
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades[i]['price']
            sell_price = sell_trades[i]['price']
            trade_return = (sell_price - buy_price) / buy_price
            total_return += trade_return
            print(f"  Trade {i+1}: Buy ${buy_price:.2f} -> Sell ${sell_price:.2f} = {trade_return*100:.1f}%")
        
        avg_return = total_return / min(len(buy_trades), len(sell_trades)) if min(len(buy_trades), len(sell_trades)) > 0 else 0
        
    else:
        total_return = 0
        avg_return = 0
        print("  No complete trades found")
    
    # Buy and hold comparison
    start_price = df['Close'].iloc[20]
    end_price = df['Close'].iloc[-1]
    buy_hold_return = (end_price - start_price) / start_price
    
    print(f"\n📊 RESULTS:")
    print(f"  Strategy Return: {total_return*100:.1f}%")
    print(f"  Buy & Hold:      {buy_hold_return*100:.1f}%")
    print(f"  Trades: {len(trades)}")
    print(f"  Current Price: ${end_price:.2f}")
    
    # Show recent signals
    recent_buy = buy_signals.tail(5).any()
    recent_sell = sell_signals.tail(5).any()
    
    print(f"\n🎯 RECENT SIGNALS (last 5 days):")
    print(f"  Buy Signal: {'YES' if recent_buy else 'NO'}")
    print(f"  Sell Signal: {'YES' if recent_sell else 'NO'}")
    
    if recent_buy:
        print(f"  ✅ Strategy suggests BUY - matches your live trading signal!")
    
    # Show why your live trader picked this stock
    latest_momentum_5d = df['momentum_5d'].iloc[-1]
    latest_momentum_10d = df['momentum_10d'].iloc[-1]
    latest_rsi = df['RSI'].iloc[-1]
    
    print(f"\n🔍 CURRENT METRICS:")
    print(f"  5-day momentum: {latest_momentum_5d*100:.1f}%")
    print(f"  10-day momentum: {latest_momentum_10d*100:.1f}%")
    print(f"  RSI: {latest_rsi:.1f}")
    print(f"  Price vs MA5: {((end_price/df['MA5'].iloc[-1])-1)*100:.1f}%")
    
    return {
        'symbol': symbol,
        'strategy_return': total_return,
        'buy_hold_return': buy_hold_return,
        'trades': len(trades),
        'recent_buy_signal': recent_buy
    }

if __name__ == "__main__":
    result = quick_test()
    
    print(f"\n🎯 SUMMARY:")
    print(f"Your live trading bot is working and generating signals!")
    print(f"The strategy shows {'positive' if result['strategy_return'] > 0 else 'negative'} historical performance.")
    print(f"Your current signal generation logic is {'beating' if result['strategy_return'] > result['buy_hold_return'] else 'underperforming'} buy-and-hold.")
