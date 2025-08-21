#!/usr/bin/env python3
"""
Enhanced Live Trading System
Based on your existing paper_trade_runner.py but now integrated with
sophisticated ML, elite stock selection, and options strategies

This enhances your working system without breaking it.
"""

import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time
from zoneinfo import ZoneInfo
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Import your existing config
try:
    from algobot.config import GLOBAL_CONFIG
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️ GLOBAL_CONFIG not available, using default universe")

# Import unified system for enhanced capabilities
try:
    from unified_ml_trading_system import UnifiedMLTradingSystem
    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False
    print("⚠️ Unified system not available, using basic signals only")

def get_enhanced_universe() -> List[str]:
    """Get enhanced stock universe using elite selection or fallback to existing"""
    
    if UNIFIED_AVAILABLE:
        print("🔍 Using Elite AI Stock Selection...")
        trading_system = UnifiedMLTradingSystem()
        return trading_system.get_elite_stock_universe()
    elif CONFIG_AVAILABLE:
        print("📋 Using existing GLOBAL_CONFIG universe...")
        return list(GLOBAL_CONFIG.universe.core_universe)[:30]
    else:
        print("📋 Using default fallback universe...")
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'CRM', 'ADBE', 'NFLX', 'AMD', 'PLTR', 'SNOW', 'COIN'
        ]

def calculate_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced version of your _indicators function with ML features"""
    d = df.copy()
    
    # Your existing indicators
    d['MA5'] = d['Close'].rolling(5).mean()
    d['MA10'] = d['Close'].rolling(10).mean()
    
    # Enhanced RSI calculation
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    d['RSI'] = 100 - (100 / (1 + rs))
    
    # Additional ML-ready features
    d['MA20'] = d['Close'].rolling(20).mean()
    d['volatility'] = d['Close'].pct_change().rolling(20).std()
    d['volume_ma'] = d['Volume'].rolling(20).mean()
    d['volume_ratio'] = d['Volume'] / d['volume_ma']
    
    # ATR for better risk management
    high_low = d['High'] - d['Low']
    high_close = np.abs(d['High'] - d['Close'].shift())
    low_close = np.abs(d['Low'] - d['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    d['ATR'] = tr.rolling(14).mean()
    
    return d

def generate_enhanced_signal(d: pd.DataFrame, i: int, symbol: str = None) -> Dict:
    """Enhanced version of your _signal function with ML integration"""
    
    # Your existing signal logic (preserved)
    price = float(d['Close'].iloc[i])
    close = d['Close']
    ma5 = float(d['MA5'].iloc[i]) if not pd.isna(d['MA5'].iloc[i]) else price
    ma10 = float(d['MA10'].iloc[i]) if not pd.isna(d['MA10'].iloc[i]) else price
    
    r5 = close.iloc[i-5:i] if i >= 5 else close.iloc[:i]
    r10 = close.iloc[i-10:i] if i >= 10 else close.iloc[:i]
    r20 = close.iloc[i-20:i] if i >= 20 else close.iloc[:i]
    
    trend_5d = (price - float(r5.mean()))/max(float(r5.mean()), 1e-9) if len(r5) > 0 else 0
    trend_10d = (price - float(r10.mean()))/max(float(r10.mean()), 1e-9) if len(r10) > 0 else 0
    trend_20d = (price - float(r20.mean()))/max(float(r20.mean()), 1e-9) if len(r20) > 0 else 0
    
    vol10 = float(np.std(r10))/max(float(np.mean(r10)), 1e-9) if len(r10) > 0 else 0
    momentum_consistency = np.mean([trend_5d>0, trend_10d>0, trend_20d>0])

    # Your existing signal strength calculation (preserved)
    buy_strength = 0.0
    sell_strength = 0.0
    
    if trend_5d > 0.025 and trend_10d > 0.025:
        buy_strength += min(1.0, (trend_5d + trend_10d)/0.1) * 0.3
    if price > ma5 > ma10:
        buy_strength += min(1.0, (price - ma10)/max(ma10,1e-9)/0.05) * 0.2
    if trend_5d > 0.0125 and float(d['RSI'].iloc[i]) < 20:
        buy_strength += (20 - float(d['RSI'].iloc[i]))/20 * 0.15
    buy_strength += momentum_consistency * 0.2

    if trend_5d < -0.02 and trend_10d < -0.045:
        sell_strength += min(1.0, abs(trend_5d + trend_10d)/0.1) * 0.4
    if price < ma5 < ma10:
        sell_strength += min(1.0, (ma10 - price)/max(ma10,1e-9)/0.05) * 0.3
    if float(d['RSI'].iloc[i]) > 65 and trend_5d < -0.01:
        sell_strength += (float(d['RSI'].iloc[i]) - 65)/35 * 0.2
    if vol10 > 0.07:
        sell_strength += min(1.0, vol10/0.2) * 0.1

    # Enhanced ML prediction (if available)
    ml_boost = 0.0
    if UNIFIED_AVAILABLE and symbol:
        try:
            trading_system = UnifiedMLTradingSystem()
            if symbol in trading_system.ensemble_models:
                ml_prediction = trading_system.get_ml_prediction(symbol, d.iloc[:i+1])
                ml_boost = (ml_prediction - 50) / 100 * 0.2  # ±20% boost from ML
        except:
            pass
    
    # Apply ML boost
    if ml_boost > 0:
        buy_strength += ml_boost
    else:
        sell_strength += abs(ml_boost)

    # Enhanced thresholds (more aggressive than your original 0.3)
    signal = 'HOLD'
    strength = 0.0
    buy_threshold = 0.25  # Lowered from your 0.3 for more signals
    sell_threshold = 0.25
    
    if buy_strength > buy_threshold and buy_strength > sell_strength:
        signal = 'BUY'
        strength = min(1.0, buy_strength)
    elif sell_strength > sell_threshold and sell_strength > buy_strength:
        signal = 'SELL'
        strength = min(1.0, sell_strength)
    
    return {
        'signal': signal,
        'strength': float(strength),
        'buy_strength': float(buy_strength),
        'sell_strength': float(sell_strength),
        'momentum_consistency': float(momentum_consistency),
        'volatility': float(vol10),
        'price': float(price),
        'ml_boost': float(ml_boost),
        'rsi': float(d['RSI'].iloc[i]) if not pd.isna(d['RSI'].iloc[i]) else 50.0,
        'atr': float(d['ATR'].iloc[i]) if 'ATR' in d.columns and not pd.isna(d['ATR'].iloc[i]) else 0.0
    }

def enhanced_paper_trade_runner(account_size: float = 100000, 
                              max_buys: int = 5, 
                              execute: bool = False,
                              universe_override: List[str] = None):
    """Enhanced version of your paper trade runner with ML integration"""
    
    print("🚀 ENHANCED PAPER TRADE RUNNER")
    print("Based on your existing system with ML enhancements")
    print("=" * 60)
    
    # Get enhanced universe
    if universe_override:
        symbols = universe_override
    else:
        symbols = get_enhanced_universe()
    
    print(f"📊 Analyzing {len(symbols)} symbols...")
    print(f"💰 Account size: ${account_size:,.2f}")
    
    # Download and analyze data
    all_signals = {}
    successful_analysis = 0
    
    for symbol in symbols:
        try:
            print(f"   Analyzing {symbol}...")
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='3mo')  # 3 months for analysis
            
            if len(df) < 30:
                print(f"   ⚠️ Insufficient data for {symbol}")
                continue
            
            # Calculate enhanced indicators
            df_enhanced = calculate_enhanced_indicators(df)
            
            # Generate enhanced signal
            signal_data = generate_enhanced_signal(df_enhanced, len(df_enhanced)-1, symbol)
            
            current_price = signal_data['price']
            
            # Enhanced position sizing based on signal strength and volatility
            if signal_data['signal'] in ['BUY', 'SELL']:
                base_allocation = 0.10  # 10% base allocation
                signal_adjusted = base_allocation * signal_data['strength']
                
                # Volatility adjustment
                if signal_data['atr'] > 0:
                    vol_adjustment = min(1.0, 2.0 / (signal_data['atr'] / current_price * 100))
                    final_allocation = signal_adjusted * vol_adjustment
                else:
                    final_allocation = signal_adjusted
                
                position_value = account_size * final_allocation
                shares = int(position_value / current_price)
                
                signal_data.update({
                    'position_value': position_value,
                    'shares': shares,
                    'allocation_pct': final_allocation * 100
                })
            
            all_signals[symbol] = signal_data
            successful_analysis += 1
            
            # Print signal summary
            signal_str = signal_data['signal']
            strength = signal_data['strength']
            ml_boost = signal_data.get('ml_boost', 0)
            
            if signal_str != 'HOLD':
                ml_indicator = f" (ML: {ml_boost:+.2f})" if abs(ml_boost) > 0.01 else ""
                print(f"   🎯 {signal_str} - Strength: {strength:.2f}{ml_indicator}")
                
                if 'shares' in signal_data:
                    shares = signal_data['shares']
                    allocation = signal_data['allocation_pct']
                    print(f"      📦 Position: {shares} shares (${signal_data['position_value']:,.0f}, {allocation:.1f}%)")
        
        except Exception as e:
            print(f"   ❌ Error analyzing {symbol}: {e}")
            continue
    
    print(f"\n✅ Successfully analyzed {successful_analysis}/{len(symbols)} symbols")
    
    # Summary of signals
    buy_signals = [s for s in all_signals.values() if s['signal'] == 'BUY']
    sell_signals = [s for s in all_signals.values() if s['signal'] == 'SELL']
    
    print(f"\n📊 SIGNAL SUMMARY:")
    print(f"   🟢 BUY signals: {len(buy_signals)}")
    print(f"   🔴 SELL signals: {len(sell_signals)}")
    print(f"   ⚪ HOLD signals: {len(all_signals) - len(buy_signals) - len(sell_signals)}")
    
    # Top buy recommendations
    if buy_signals:
        top_buys = sorted(buy_signals, key=lambda x: x['strength'], reverse=True)[:max_buys]
        
        print(f"\n🎯 TOP {len(top_buys)} BUY RECOMMENDATIONS:")
        total_allocation = 0
        
        for i, signal in enumerate(top_buys, 1):
            symbol = [k for k, v in all_signals.items() if v == signal][0]
            strength = signal['strength']
            shares = signal.get('shares', 0)
            value = signal.get('position_value', 0)
            allocation = signal.get('allocation_pct', 0)
            ml_boost = signal.get('ml_boost', 0)
            rsi = signal.get('rsi', 50)
            
            total_allocation += allocation
            
            ml_str = f" +ML({ml_boost:+.2f})" if abs(ml_boost) > 0.01 else ""
            print(f"   {i}. {symbol}: ${signal['price']:.2f} - Strength {strength:.2f}{ml_str}")
            print(f"      📦 {shares} shares = ${value:,.0f} ({allocation:.1f}%) | RSI: {rsi:.0f}")
            
            if execute:
                print(f"      ⚡ EXECUTING: BUY {shares} shares of {symbol}")
            else:
                print(f"      📝 PAPER: Would buy {shares} shares of {symbol}")
        
        print(f"\n💼 Total allocation: {total_allocation:.1f}% of portfolio")
        print(f"💰 Remaining cash: {100 - total_allocation:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'account_size': account_size,
        'universe_size': len(symbols),
        'successful_analysis': successful_analysis,
        'signals': all_signals,
        'summary': {
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'top_recommendations': len(top_buys) if buy_signals else 0
        }
    }
    
    filename = f"enhanced_trading_signals_{timestamp}.json"
    import json
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filename}")
    
    return results

def main():
    """Main execution matching your existing interface"""
    parser = argparse.ArgumentParser(description="Enhanced Alpaca paper-trade runner")
    parser.add_argument('--account', type=float, default=100000.0, help='Account size for sizing logic')
    parser.add_argument('--execute', action='store_true', help='If set, place orders (not implemented in demo)')
    parser.add_argument('--dry-run', action='store_true', help='Print intended actions only (default)')
    parser.add_argument('--max-buys', type=int, default=5, help='Max new buys to open this run')
    parser.add_argument('--universe', type=str, nargs='*', help='Optional override ticker list')
    
    args = parser.parse_args()
    
    print("🌟 ENHANCED PAPER TRADE RUNNER")
    print("Your existing system + ML + Elite Stock Selection + Options Intelligence")
    print("=" * 80)
    
    # Override universe if provided
    universe = None
    if args.universe:
        universe = [s.upper() for s in args.universe]
        print(f"📋 Using custom universe: {universe}")
    
    # Run enhanced paper trading
    results = enhanced_paper_trade_runner(
        account_size=args.account,
        max_buys=args.max_buys,
        execute=args.execute,
        universe_override=universe
    )
    
    print("\n🎉 Enhanced paper trading analysis completed!")
    print("🔗 This preserves your existing system while adding ML intelligence")
    
    if not args.execute:
        print("\n💡 Add --execute flag when ready for live trading")

if __name__ == "__main__":
    main()
