#!/usr/bin/env python3
"""
NVDA SIGNAL ANALYSIS
Check why no trades were made during the strong NVDA run
Analyze what the signal thresholds should be for momentum markets
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    data = data.copy()
    
    # Moving averages
    data['MA5'] = data['Close'].rolling(5).mean()
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA20'] = data['Close'].rolling(20).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

def analyze_nvda_signals():
    """Analyze NVDA signals during the 3-month period"""
    print("📊 NVDA SIGNAL ANALYSIS - WHY NO TRADES?")
    print("=" * 50)
    
    # Download data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=120)  # 3 months + buffer
    
    data = yf.download('NVDA', start=start_date, end=end_date, progress=False)
    data = calculate_technical_indicators(data)
    
    # Current config from optimizer
    config = {
        'trend_5d_buy_threshold': 0.025,
        'trend_5d_sell_threshold': -0.02,
        'trend_10d_buy_threshold': 0.025,
        'trend_10d_sell_threshold': -0.045,
        'rsi_overbought': 65,
        'rsi_oversold': 20,
        'volatility_threshold': 0.07,
        'volume_ratio_threshold': 1.6
    }
    
    print(f"🔧 CURRENT SIGNAL THRESHOLDS:")
    print(f"   📈 Trend 5d BUY: >{config['trend_5d_buy_threshold']:.1%}")
    print(f"   📉 Trend 5d SELL: <{config['trend_5d_sell_threshold']:.1%}")
    print(f"   📈 Trend 10d BUY: >{config['trend_10d_buy_threshold']:.1%}")
    print(f"   📉 Trend 10d SELL: <{config['trend_10d_sell_threshold']:.1%}")
    print(f"   🔴 RSI Overbought: >{config['rsi_overbought']}")
    print(f"   🟢 RSI Oversold: <{config['rsi_oversold']}")
    print(f"   🌊 Volatility Limit: <{config['volatility_threshold']:.1%}")
    print(f"   📊 Volume Ratio: >{config['volume_ratio_threshold']:.1f}x")
    
    # Analyze each day
    print(f"\n📅 DAILY SIGNAL ANALYSIS:")
    print("-" * 80)
    
    signals_that_could_trigger = []
    
    for i in range(20, len(data)):
        date = data.index[i]
        price = float(data['Close'].iloc[i])
        
        # Calculate signal components
        recent_5d = data['Close'].iloc[i-5:i]
        recent_10d = data['Close'].iloc[i-10:i]
        
        trend_5d = (price - float(recent_5d.mean())) / float(recent_5d.mean())
        trend_10d = (price - float(recent_10d.mean())) / float(recent_10d.mean())
        
        ma5 = float(data['MA5'].iloc[i]) if not pd.isna(data['MA5'].iloc[i]) else price
        ma10 = float(data['MA10'].iloc[i]) if not pd.isna(data['MA10'].iloc[i]) else price
        rsi = float(data['RSI'].iloc[i]) if not pd.isna(data['RSI'].iloc[i]) else 50
        
        volatility = float(recent_10d.std()) / float(recent_10d.mean()) if float(recent_10d.mean()) > 0 else 0
        
        # Volume analysis
        recent_volume = float(data['Volume'].iloc[i-10:i].mean())
        current_volume = float(data['Volume'].iloc[i])
        volume_ratio = current_volume / recent_volume if recent_volume > 0 else 1
        
        # Check which conditions would trigger
        buy_conditions = []
        sell_conditions = []
        
        # BUY CONDITIONS CHECK
        if trend_5d > config['trend_5d_buy_threshold'] and trend_10d > config['trend_10d_buy_threshold'] and volume_ratio > config['volume_ratio_threshold']:
            buy_conditions.append("Strong_Trend_+_Volume")
        
        if price > ma5 > ma10 and trend_5d > config['trend_5d_buy_threshold']:
            buy_conditions.append("MA_Crossover_+_Trend")
            
        if rsi < config['rsi_oversold'] and trend_5d > config['trend_5d_buy_threshold']/2:
            buy_conditions.append("RSI_Oversold_+_Trend")
        
        # SELL CONDITIONS CHECK
        if trend_5d < config['trend_5d_sell_threshold'] and trend_10d < config['trend_10d_sell_threshold']:
            sell_conditions.append("Dual_Trend_Breakdown")
            
        if price < ma5 < ma10:
            sell_conditions.append("MA_Breakdown")
            
        if rsi > config['rsi_overbought'] and trend_5d < config['trend_5d_sell_threshold']/2:
            sell_conditions.append("RSI_Overbought_+_Weak_Trend")
            
        if volatility > config['volatility_threshold'] and trend_10d < config['trend_10d_sell_threshold']:
            sell_conditions.append("High_Vol_+_Weak_Trend")
        
        # Determine signal
        signal = 'HOLD'
        if sell_conditions:
            signal = 'SELL'
        elif buy_conditions:
            signal = 'BUY'
        
        # Store potential signals for analysis
        signals_that_could_trigger.append({
            'date': date,
            'price': price,
            'signal': signal,
            'trend_5d': trend_5d,
            'trend_10d': trend_10d,
            'rsi': rsi,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'buy_conditions': buy_conditions,
            'sell_conditions': sell_conditions
        })
        
        # Print key days
        if i % 10 == 0 or buy_conditions or sell_conditions:
            trend_5d_str = f"{trend_5d:+.1%}"
            trend_10d_str = f"{trend_10d:+.1%}"
            vol_str = f"{volatility:.1%}"
            vol_ratio_str = f"{volume_ratio:.1f}x"
            
            print(f"{date.strftime('%Y-%m-%d')}: ${price:6.2f} | {signal:4s} | 5d:{trend_5d_str:>6s} 10d:{trend_10d_str:>6s} | RSI:{rsi:4.0f} | Vol:{vol_str:>5s} | VR:{vol_ratio_str:>5s}")
            
            if buy_conditions:
                print(f"               🟢 BUY triggers: {', '.join(buy_conditions)}")
            if sell_conditions:
                print(f"               🔴 SELL triggers: {', '.join(sell_conditions)}")
    
    # Summary analysis
    print(f"\n" + "="*50)
    print("🔍 THRESHOLD ANALYSIS")
    print("="*50)
    
    # Find maximum values during the period
    all_trend_5d = [s['trend_5d'] for s in signals_that_could_trigger]
    all_trend_10d = [s['trend_10d'] for s in signals_that_could_trigger]
    all_volume_ratios = [s['volume_ratio'] for s in signals_that_could_trigger]
    all_rsi = [s['rsi'] for s in signals_that_could_trigger]
    
    max_trend_5d = max(all_trend_5d)
    max_trend_10d = max(all_trend_10d)
    max_volume_ratio = max(all_volume_ratios)
    min_rsi = min(all_rsi)
    
    print(f"📊 ACTUAL RANGES DURING NVDA RUN:")
    print(f"   📈 Max 5d Trend: {max_trend_5d:+.1%} (need >{config['trend_5d_buy_threshold']:.1%})")
    print(f"   📈 Max 10d Trend: {max_trend_10d:+.1%} (need >{config['trend_10d_buy_threshold']:.1%})")
    print(f"   📊 Max Volume Ratio: {max_volume_ratio:.1f}x (need >{config['volume_ratio_threshold']:.1f}x)")
    print(f"   🔽 Min RSI: {min_rsi:.0f} (need <{config['rsi_oversold']})")
    
    # Suggest better thresholds
    print(f"\n💡 SUGGESTED MOMENTUM-FRIENDLY THRESHOLDS:")
    print(f"   📈 Trend 5d BUY: >{max_trend_5d * 0.7:.1%} (vs current {config['trend_5d_buy_threshold']:.1%})")
    print(f"   📈 Trend 10d BUY: >{max_trend_10d * 0.7:.1%} (vs current {config['trend_10d_buy_threshold']:.1%})")
    print(f"   📊 Volume Ratio: >{max_volume_ratio * 0.8:.1f}x (vs current {config['volume_ratio_threshold']:.1f}x)")
    print(f"   🟢 RSI Oversold: <{min_rsi + 10:.0f} (vs current {config['rsi_oversold']})")
    
    buy_signals = [s for s in signals_that_could_trigger if s['signal'] == 'BUY']
    sell_signals = [s for s in signals_that_could_trigger if s['signal'] == 'SELL']
    
    print(f"\n📈 SIGNAL SUMMARY:")
    print(f"   🟢 Buy signals generated: {len(buy_signals)}")
    print(f"   🔴 Sell signals generated: {len(sell_signals)}")
    print(f"   ⚪ Hold days: {len(signals_that_could_trigger) - len(buy_signals) - len(sell_signals)}")
    
    if len(buy_signals) == 0:
        print(f"\n❌ NO BUY SIGNALS = NO TRADES = MISSED 25.6% GAIN!")
        print(f"🎯 Strategy was too conservative for this momentum period")

if __name__ == "__main__":
    analyze_nvda_signals()
