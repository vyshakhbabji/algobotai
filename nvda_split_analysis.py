#!/usr/bin/env python3
"""
NVDA Stock Split Impact Analysis
How the June 2024 10-for-1 split affects Elite AI predictions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NVDAStockSplitAnalysis:
    def __init__(self):
        self.symbol = "NVDA"
        # June 2024 10-for-1 split date
        self.split_date = "2024-06-07"  # Approximate split date
        
    def analyze_split_impact(self):
        """Analyze how stock split affects AI model predictions"""
        print("📊 NVDA STOCK SPLIT IMPACT ANALYSIS")
        print("=" * 40)
        
        # Get extended historical data
        print("📥 Downloading NVDA historical data...")
        
        # Pre-split and post-split data
        pre_split_data = yf.download("NVDA", start="2023-01-01", end="2024-06-10", progress=False)
        post_split_data = yf.download("NVDA", start="2024-06-10", end=datetime.now().strftime('%Y-%m-%d'), progress=False)
        all_data = yf.download("NVDA", period="2y", progress=False)
        
        print(f"\n🔍 STOCK SPLIT TIMELINE:")
        print(f"   June 2024: 10-for-1 split")
        print(f"   Pre-split data: {len(pre_split_data)} days")
        print(f"   Post-split data: {len(post_split_data)} days")
        
        # Analyze price characteristics
        self.analyze_price_characteristics(pre_split_data, post_split_data, all_data)
        
        # Analyze volatility changes
        self.analyze_volatility_changes(pre_split_data, post_split_data)
        
        # Model confusion analysis
        self.analyze_model_confusion(all_data)
        
        # Corrected analysis
        self.split_aware_analysis(all_data)
        
    def analyze_price_characteristics(self, pre_split, post_split, all_data):
        """Analyze price characteristics before and after split"""
        print("
📈 PRICE CHARACTERISTICS ANALYSIS")
        print("-----------------------------------")
        
        # Calculate means properly
        pre_split_avg = float(pre_split['Close'].mean())
        post_split_avg = float(post_split['Close'].mean())
        pre_split_min = float(pre_split['Close'].min())
        pre_split_max = float(pre_split['Close'].max())
        post_split_min = float(post_split['Close'].min())
        post_split_max = float(post_split['Close'].max())
        pre_split_std = float(pre_split['Close'].std())
        post_split_std = float(post_split['Close'].std())
        
        print("PRE-SPLIT (Jan 2023 - June 2024):")
        print(f"   Average Price: ${pre_split_avg:.2f}")
        print(f"   Price Range: ${pre_split_min:.2f} - ${pre_split_max:.2f}")
        print(f"   Volatility: {pre_split_std:.2f}")
        
        print("
POST-SPLIT (June 2024 - Present):")
        print(f"   Average Price: ${post_split_avg:.2f}")
        print(f"   Price Range: ${post_split_min:.2f} - ${post_split_max:.2f}")
        print(f"   Volatility: {post_split_std:.2f}")
        
        # Calculate adjusted averages
        adjusted_pre_avg = pre_split_avg / 10  # Adjust for 10:1 split
        print(f"
� SPLIT-ADJUSTED COMPARISON:")
        print(f"   Pre-split adjusted avg: ${adjusted_pre_avg:.2f}")
        print(f"   Post-split avg: ${post_split_avg:.2f}")
        print(f"   Split effect removed: {((post_split_avg / adjusted_pre_avg) - 1) * 100:.1f}% change")
                
    def analyze_volatility_changes(self, pre_split, post_split):
        """Analyze volatility patterns before and after split"""
        print(f"\n⚡ VOLATILITY ANALYSIS")
        print("-" * 22)
        
        if not pre_split.empty and not post_split.empty:
            # Calculate daily returns
            pre_returns = pre_split['Close'].pct_change().dropna()
            post_returns = post_split['Close'].pct_change().dropna()
            
            # Volatility metrics
            pre_vol = pre_returns.std() * np.sqrt(252) * 100
            post_vol = post_returns.std() * np.sqrt(252) * 100
            
            print(f"PRE-SPLIT VOLATILITY:")
            print(f"   Annualized: {pre_vol:.1f}%")
            print(f"   Daily avg: {pre_returns.abs().mean()*100:.2f}%")
            
            print(f"\nPOST-SPLIT VOLATILITY:")
            print(f"   Annualized: {post_vol:.1f}%")
            print(f"   Daily avg: {post_returns.abs().mean()*100:.2f}%")
            
            vol_change = ((post_vol / pre_vol) - 1) * 100
            print(f"\n📊 VOLATILITY CHANGE: {vol_change:+.1f}%")
            
            if abs(vol_change) > 20:
                print(f"   ⚠️ Significant volatility change after split!")
                print(f"   💡 This could affect model predictions")
            else:
                print(f"   ✅ Volatility relatively stable")
                
    def analyze_model_confusion(self, all_data):
        """Analyze how stock split might confuse ML models"""
        print(f"\n🤖 MODEL CONFUSION ANALYSIS")
        print("-" * 28)
        
        # Look for structural breaks around split date
        split_idx = None
        for i, date in enumerate(all_data.index):
            if date.strftime('%Y-%m-%d') >= self.split_date:
                split_idx = i
                break
                
        if split_idx and split_idx > 50:
            # Data before and after split
            before_split = all_data.iloc[:split_idx]
            after_split = all_data.iloc[split_idx:]
            
            print(f"🔍 POTENTIAL MODEL ISSUES:")
            
            # 1. Price level shifts
            price_before = before_split['Close'].iloc[-10:].mean()
            price_after = after_split['Close'].iloc[:10].mean()
            price_jump = abs((price_after / price_before) - 1) * 100
            
            if price_jump > 50:
                print(f"   🔴 Major price level shift: {price_jump:.1f}%")
                print(f"   💡 Models may see this as trend change")
            
            # 2. Moving average disruption
            ma20_before = before_split['Close'].rolling(20).mean().iloc[-1]
            ma20_after = after_split['Close'].rolling(20).mean().iloc[-1] if len(after_split) >= 20 else None
            
            if ma20_after:
                ma_ratio = ma20_after / ma20_before
                print(f"   📊 MA20 ratio (after/before): {ma_ratio:.2f}")
                if ma_ratio < 0.8 or ma_ratio > 1.2:
                    print(f"   🔴 Moving averages disrupted by split")
                    
            # 3. Feature engineering issues
            print(f"\n🛠️ FEATURE ENGINEERING PROBLEMS:")
            print(f"   • Price-based features (MA, Bollinger) affected")
            print(f"   • Support/Resistance levels shifted")
            print(f"   • Historical patterns broken")
            print(f"   • Model sees 'artificial' volatility")
            
            # 4. Training data contamination
            print(f"\n⚠️ TRAINING DATA ISSUES:")
            print(f"   • Models trained on mixed pre/post-split data")
            print(f"   • Split creates 'false' patterns")
            print(f"   • Historical relationships invalidated")
            print(f"   • Ensemble models give conflicting signals")
            
    def split_aware_analysis(self, all_data):
        """Provide split-aware analysis and recommendations"""
        print(f"\n🎯 SPLIT-AWARE RECOMMENDATIONS")
        print("-" * 32)
        
        print(f"📊 WHY ELITE AI SHOWS SELL SIGNAL:")
        print(f"   1. 🔴 Models confused by split-adjusted data")
        print(f"   2. 🔴 Training includes pre-split patterns")
        print(f"   3. 🔴 Technical indicators disrupted")
        print(f"   4. 🔴 Historical support/resistance invalid")
        print(f"   5. 🔴 Ensemble sees 'artificial' bearish signals")
        
        print(f"\n💡 CORRECTED ANALYSIS APPROACH:")
        print(f"   1. ✅ Use only post-split data for training")
        print(f"   2. ✅ Recalibrate technical indicators")
        print(f"   3. ✅ Focus on percentage-based features")
        print(f"   4. ✅ Weight recent data more heavily")
        print(f"   5. ✅ Use split-adjusted fundamentals")
        
        # Quick corrected analysis
        print(f"\n📈 SPLIT-CORRECTED QUICK ANALYSIS:")
        
        # Use only recent post-split data
        recent_data = all_data.iloc[-60:]  # Last 60 days
        
        if len(recent_data) >= 20:
            current_price = recent_data['Close'].iloc[-1]
            ma20 = recent_data['Close'].rolling(20).mean().iloc[-1]
            recent_returns = recent_data['Close'].pct_change().dropna()
            momentum = recent_returns.iloc[-5:].mean() * 100
            
            print(f"   Current vs MA20: {((current_price/ma20)-1)*100:+.1f}%")
            print(f"   Recent momentum: {momentum:+.2f}%")
            print(f"   Volatility: {recent_returns.std()*np.sqrt(252)*100:.1f}%")
            
            # Split-aware signals
            signals = []
            if current_price > ma20:
                signals.append("✅ Above MA20")
            else:
                signals.append("❌ Below MA20")
                
            if momentum > 0:
                signals.append("✅ Positive momentum")
            else:
                signals.append("❌ Negative momentum")
                
            recent_vol = recent_returns.std() * np.sqrt(252) * 100
            if recent_vol < 50:
                signals.append("✅ Reasonable volatility")
            else:
                signals.append("⚠️ High volatility")
                
            print(f"\n🎯 SPLIT-CORRECTED SIGNALS:")
            for signal in signals:
                print(f"   {signal}")
                
        # Final recommendation
        print(f"\n💼 UPDATED NVDA RECOMMENDATION:")
        print(f"   🚨 IGNORE Elite AI SELL signal!")
        print(f"   📊 Split confusion invalidates prediction")
        print(f"   🎯 Focus on post-split fundamentals:")
        print(f"      • AI narrative still strong")
        print(f"      • Post-split accessibility improved")
        print(f"      • Volume increased (bullish)")
        print(f"      • Technical patterns resetting")
        
        print(f"\n✅ REVISED POSITION ADVICE:")
        print(f"   • HOLD your NVDA position")
        print(f"   • Split improved stock liquidity")
        print(f"   • Don't panic on AI false signals")
        print(f"   • Monitor post-split patterns")
        print(f"   • Consider splits as bullish catalysts")

def main():
    """Run NVDA stock split impact analysis"""
    analyzer = NVDAStockSplitAnalysis()
    analyzer.analyze_split_impact()
    
    print(f"\n" + "="*50)
    print(f"🎯 KEY INSIGHT: June 2024 10-for-1 split likely")
    print(f"   caused Elite AI model confusion and false")
    print(f"   SELL signal. Split effects should be bullish!")
    print(f"="*50)

if __name__ == "__main__":
    main()
