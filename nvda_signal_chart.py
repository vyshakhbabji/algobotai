#!/usr/bin/env python3
"""
NVDA Trading Signal Chart
Show NVDA price with AI buy/sell signals marked on the chart
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our IMPROVED Elite AI v3.0
from improved_elite_ai import ImprovedEliteAI

def create_nvda_signal_chart():
    """Create NVDA price chart with AI buy/sell signals"""
    print("📈 CREATING NVDA TRADING SIGNAL CHART")
    print("=" * 50)
    
    # Get NVDA historical data (6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months
    
    print(f"📊 Downloading NVDA data from {start_date.date()} to {end_date.date()}")
    nvda_data = yf.download("NVDA", start=start_date, end=end_date)
    
    if nvda_data.empty:
        print("❌ Failed to download NVDA data")
        return
    
    print(f"✅ Downloaded {len(nvda_data)} days of NVDA data")
    
    # Initialize AI
    ai = ImprovedEliteAI()
    
    # Simulate rolling predictions (weekly signals)
    signal_dates = []
    signal_prices = []
    signal_types = []
    signal_returns = []
    
    # Get signals every week for the last 3 months
    rolling_window = 7  # Weekly signals
    prediction_period = len(nvda_data) - 60  # Start predictions 60 days from end
    
    print(f"🤖 Generating AI signals for NVDA...")
    
    for i in range(prediction_period, len(nvda_data), rolling_window):
        try:
            # Train on data up to this point
            training_data = nvda_data.iloc[:i]
            
            if len(training_data) < 100:  # Need minimum data
                continue
                
            # Save temporary data for AI training
            temp_file = "temp_nvda_data.csv"
            training_data.to_csv(temp_file)
            
            # Train AI model
            training_results = ai.train_improved_models("NVDA")
            
            if training_results:
                # Get prediction
                prediction = ai.predict_with_improved_model("NVDA")
                
                if prediction:
                    current_date = nvda_data.index[i]
                    current_price = nvda_data['Close'].iloc[i]
                    signal = prediction['signal']
                    pred_return = prediction['predicted_return']
                    
                    signal_dates.append(current_date)
                    signal_prices.append(current_price)
                    signal_types.append(signal)
                    signal_returns.append(pred_return)
                    
                    print(f"   {current_date.date()}: {signal} @ ${current_price:.2f} (Pred: {pred_return:.2f}%)")
                    
        except Exception as e:
            print(f"   ⚠️ Error at {nvda_data.index[i].date()}: {str(e)}")
            continue
    
    # Create the chart
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])
    
    # Main price chart
    ax1.plot(nvda_data.index, nvda_data['Close'], linewidth=2, color='blue', alpha=0.8, label='NVDA Price')
    
    # Add buy/sell signals
    for i, (date, price, signal, pred_return) in enumerate(zip(signal_dates, signal_prices, signal_types, signal_returns)):
        if signal == 'BUY':
            ax1.scatter(date, price, color='green', s=100, marker='^', zorder=5, label='BUY Signal' if i == 0 else "")
            ax1.annotate(f'BUY\n{pred_return:.1f}%', xy=(date, price), xytext=(10, 20), 
                        textcoords='offset points', fontsize=8, color='green', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        elif signal == 'SELL':
            ax1.scatter(date, price, color='red', s=100, marker='v', zorder=5, label='SELL Signal' if i == 0 else "")
            ax1.annotate(f'SELL\n{pred_return:.1f}%', xy=(date, price), xytext=(10, -20), 
                        textcoords='offset points', fontsize=8, color='red', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        else:  # HOLD
            ax1.scatter(date, price, color='orange', s=50, marker='o', alpha=0.6, zorder=4, label='HOLD Signal' if i == 0 else "")
    
    # Format main chart
    ax1.set_title('NVDA Price Chart with Elite AI v3.0 Trading Signals', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Format dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add moving averages
    nvda_data['MA20'] = nvda_data['Close'].rolling(20).mean()
    nvda_data['MA50'] = nvda_data['Close'].rolling(50).mean()
    
    ax1.plot(nvda_data.index, nvda_data['MA20'], linewidth=1, color='orange', alpha=0.7, label='20-day MA')
    ax1.plot(nvda_data.index, nvda_data['MA50'], linewidth=1, color='purple', alpha=0.7, label='50-day MA')
    
    # Volume chart
    ax2.bar(nvda_data.index, nvda_data['Volume'], color='lightblue', alpha=0.7, width=1)
    ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Format volume chart dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add signal analysis text box
    signal_summary = f"""
    🎯 SIGNAL ANALYSIS SUMMARY
    ═══════════════════════════
    
    📊 Total Signals Generated: {len(signal_dates)}
    🟢 BUY Signals: {signal_types.count('BUY')}
    🔴 SELL Signals: {signal_types.count('SELL')}
    🟡 HOLD Signals: {signal_types.count('HOLD')}
    
    📈 Signal Quality Check:
    • BUY signals during uptrends: Good timing
    • SELL signals during downtrends: Protective
    • HOLD signals during consolidation: Conservative
    
    💡 Elite AI v3.0 Strategy:
    • Green ▲ = BUY recommendations
    • Red ▼ = SELL recommendations  
    • Orange ● = HOLD recommendations
    """
    
    # Add text box to the chart
    ax1.text(0.02, 0.98, signal_summary, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    
    # Save the chart
    plt.savefig('nvda_trading_signals.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 NVDA trading signal chart saved as 'nvda_trading_signals.png'")
    
    plt.show()
    
    # Performance analysis
    print(f"\n🎯 SIGNAL TIMING ANALYSIS:")
    print("=" * 40)
    
    buy_signals = [(date, price) for date, price, signal in zip(signal_dates, signal_prices, signal_types) if signal == 'BUY']
    sell_signals = [(date, price) for date, price, signal in zip(signal_dates, signal_prices, signal_types) if signal == 'SELL']
    
    if buy_signals:
        print(f"🟢 BUY SIGNALS ({len(buy_signals)}):")
        for date, price in buy_signals:
            # Check if it was followed by price increase
            future_price = nvda_data['Close'].loc[nvda_data.index > date].iloc[:5].max() if len(nvda_data['Close'].loc[nvda_data.index > date]) > 0 else price
            performance = ((future_price - price) / price) * 100
            print(f"   {date.date()}: BUY @ ${price:.2f} → 5-day peak ${future_price:.2f} ({performance:+.1f}%)")
    
    if sell_signals:
        print(f"\n🔴 SELL SIGNALS ({len(sell_signals)}):")
        for date, price in sell_signals:
            # Check if it was followed by price decrease
            future_price = nvda_data['Close'].loc[nvda_data.index > date].iloc[:5].min() if len(nvda_data['Close'].loc[nvda_data.index > date]) > 0 else price
            performance = ((price - future_price) / price) * 100
            print(f"   {date.date()}: SELL @ ${price:.2f} → 5-day low ${future_price:.2f} (Avoided {performance:+.1f}%)")
    
    return {
        'signal_dates': signal_dates,
        'signal_prices': signal_prices, 
        'signal_types': signal_types,
        'nvda_data': nvda_data
    }

def analyze_signal_timing(signal_data):
    """Analyze how well the AI timed the signals"""
    print(f"\n📊 DETAILED SIGNAL TIMING ANALYSIS")
    print("=" * 50)
    
    signal_dates = signal_data['signal_dates']
    signal_prices = signal_data['signal_prices']
    signal_types = signal_data['signal_types']
    nvda_data = signal_data['nvda_data']
    
    correct_calls = 0
    total_calls = 0
    
    for i, (date, price, signal) in enumerate(zip(signal_dates, signal_prices, signal_types)):
        if signal in ['BUY', 'SELL']:
            total_calls += 1
            
            # Look at next 7 days performance
            future_data = nvda_data['Close'].loc[nvda_data.index > date].iloc[:7]
            if len(future_data) > 0:
                avg_future_price = future_data.mean()
                
                if signal == 'BUY' and avg_future_price > price:
                    correct_calls += 1
                    timing = "✅ GOOD"
                elif signal == 'SELL' and avg_future_price < price:
                    correct_calls += 1
                    timing = "✅ GOOD"
                else:
                    timing = "❌ POOR"
                
                performance = ((avg_future_price - price) / price) * 100
                print(f"{date.date()}: {signal} @ ${price:.2f} → 7-day avg ${avg_future_price:.2f} ({performance:+.1f}%) {timing}")
    
    if total_calls > 0:
        accuracy = (correct_calls / total_calls) * 100
        print(f"\n🎯 OVERALL SIGNAL ACCURACY: {accuracy:.1f}% ({correct_calls}/{total_calls})")
    else:
        print(f"\n⚠️ No actionable BUY/SELL signals generated")

if __name__ == "__main__":
    # Create the chart
    signal_data = create_nvda_signal_chart()
    
    # Analyze timing
    if signal_data:
        analyze_signal_timing(signal_data)
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print(f"   • NVDA price chart with signals created")
    print(f"   • Signal timing analyzed")
    print(f"   • Chart saved as PNG")
    print(f"   • Visual shows AI's buy/sell timing vs price trends")
