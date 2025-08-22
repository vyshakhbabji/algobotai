#!/usr/bin/env python3
"""
Buy and Hold Performance Analysis
Compare our 3.8% trading performance vs simple buy-and-hold
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_buy_and_hold():
    # Our 20 elite stocks
    elite_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ORCL',
        'JPM', 'BAC', 'V', 'MA', 'UNH', 'JNJ', 'PG', 'KO', 'XOM', 'HD', 'DIS', 'CRM'
    ]
    
    # Test period: May 22, 2024 to August 21, 2024 (3 months) - Historical data
    start_date = "2024-05-22"
    end_date = "2024-08-21"
    
    print("🔍 BUY AND HOLD ANALYSIS")
    print("=" * 80)
    print(f"📅 Period: {start_date} to {end_date} (3 months)")
    print(f"📊 Portfolio: 20 Elite Stocks (Equal Weight)")
    print(f"💰 Initial Capital: $100,000")
    print("=" * 80)
    
    # Download data for all stocks
    print("📥 Downloading stock data...")
    data = {}
    returns = []
    
    for symbol in elite_stocks:
        try:
            # Get historical data
            stock = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if len(stock) < 2:
                print(f"⚠️  Insufficient data for {symbol}")
                continue
                
            # Calculate return
            start_price = float(stock['Close'].iloc[0])
            end_price = float(stock['Close'].iloc[-1])
            stock_return = (end_price / start_price - 1) * 100
            
            data[symbol] = {
                'start_price': start_price,
                'end_price': end_price,
                'return': stock_return
            }
            returns.append(stock_return)
            
            print(f"📈 {symbol:5}: ${start_price:7.2f} → ${end_price:7.2f} = {stock_return:+6.2f}%")
            
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            continue
    
    if not returns:
        print("❌ No valid stock data found!")
        return
    
    # Calculate portfolio performance
    equal_weight_return = np.mean(returns)
    portfolio_value = 100000 * (1 + equal_weight_return / 100)
    profit = portfolio_value - 100000
    
    print("\n" + "=" * 80)
    print("💼 EQUAL-WEIGHT BUY AND HOLD RESULTS")
    print("=" * 80)
    print(f"📊 Average Stock Return:     {equal_weight_return:+6.2f}%")
    print(f"💰 Portfolio Value:          ${portfolio_value:,.2f}")
    print(f"💵 Total Profit:             ${profit:+,.2f}")
    print(f"📈 Portfolio Return:         {equal_weight_return:+6.2f}%")
    print(f"📅 Annualized Return:        {equal_weight_return * 4:+6.2f}%")
    
    # Compare with our trading system
    our_return = 3.8
    our_value = 103793
    our_profit = 3793
    
    print("\n" + "=" * 80)
    print("⚔️  PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"🤖 Our Trading System:       +{our_return:.1f}% (${our_value:,})")
    print(f"💼 Buy and Hold:             {equal_weight_return:+6.2f}% (${portfolio_value:,.0f})")
    print(f"📊 Difference:               {our_return - equal_weight_return:+6.2f}%")
    print(f"💸 Profit Difference:        ${our_profit - profit:+,.0f}")
    
    if equal_weight_return > our_return:
        underperformance = equal_weight_return - our_return
        print(f"\n🚨 UNDERPERFORMANCE: -{underperformance:.2f}%")
        print(f"💔 We LOST ${profit - our_profit:,.0f} by trading instead of holding!")
        print(f"🎯 Buy-and-hold was {underperformance/our_return*100:.1f}% better!")
    else:
        outperformance = our_return - equal_weight_return
        print(f"\n✅ OUTPERFORMANCE: +{outperformance:.2f}%")
        print(f"💚 We GAINED ${our_profit - profit:,.0f} by trading vs holding!")
    
    # Best and worst performers
    print("\n" + "=" * 80)
    print("🏆 TOP PERFORMERS (Buy & Hold)")
    print("=" * 80)
    
    # Sort by returns
    sorted_stocks = sorted(data.items(), key=lambda x: x[1]['return'], reverse=True)
    
    print("🥇 TOP 5 WINNERS:")
    for i, (symbol, info) in enumerate(sorted_stocks[:5]):
        print(f"   {i+1}. {symbol:5}: {info['return']:+7.2f}%")
    
    print("\n🥴 TOP 5 LOSERS:")
    for i, (symbol, info) in enumerate(sorted_stocks[-5:]):
        print(f"   {i+1}. {symbol:5}: {info['return']:+7.2f}%")
    
    # Calculate if we just bought the best performer
    if sorted_stocks:
        best_stock = sorted_stocks[0]
        best_return = best_stock[1]['return']
        best_value = 100000 * (1 + best_return / 100)
        
        print(f"\n🎯 IF WE JUST BOUGHT {best_stock[0]}:")
        print(f"   Return: {best_return:+6.2f}%")
        print(f"   Value:  ${best_value:,.0f}")
        print(f"   Profit: ${best_value - 100000:+,.0f}")
        print(f"   vs Our System: {best_return - our_return:+6.2f}% better")

if __name__ == "__main__":
    analyze_buy_and_hold()
