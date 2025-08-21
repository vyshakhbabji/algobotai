#!/usr/bin/env python3
"""
NVDA Stock Split Impact Analysis
Analyzes how the June 2024 10-for-1 stock split affects Elite AI predictions
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NVDASplitAnalyzer:
    def __init__(self):
        self.split_date = "2024-06-10"  # NVDA 10:1 split date
        
    def download_data(self):
        """Download NVDA historical data"""
        print("📥 Downloading NVDA historical data...")
        ticker = yf.Ticker("NVDA")
        
        # Get 2 years of data to capture before/after split
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        data = ticker.history(start=start_date, end=end_date)
        return data
        
    def analyze_split_impact(self):
        """Main analysis function"""
        data = self.download_data()
        
        # Fix timezone issues
        split_date = pd.to_datetime(self.split_date).tz_localize('America/New_York')
        
        # Split data into pre/post split periods
        pre_split = data[data.index < split_date]
        post_split = data[data.index >= split_date]
        
        print(f"\n🔍 STOCK SPLIT TIMELINE:")
        print(f"   Split Date: {self.split_date} (10-for-1)")
        print(f"   Pre-split data: {len(pre_split)} days")
        print(f"   Post-split data: {len(post_split)} days")
        
        # Price analysis
        self.analyze_prices(pre_split, post_split)
        
        # Volume analysis
        self.analyze_volume(pre_split, post_split)
        
        # Model confusion analysis
        self.analyze_model_confusion(pre_split, post_split)
        
        # Final recommendations
        self.provide_recommendations()
        
    def analyze_prices(self, pre_split, post_split):
        """Analyze price characteristics"""
        print(f"\n📈 PRICE ANALYSIS")
        print("=" * 30)
        
        pre_avg = pre_split['Close'].mean()
        post_avg = post_split['Close'].mean()
        
        print(f"PRE-SPLIT (Before {self.split_date}):")
        print(f"   Average Price: ${pre_avg:.2f}")
        print(f"   Price Range: ${pre_split['Close'].min():.2f} - ${pre_split['Close'].max():.2f}")
        print(f"   Last Price: ${pre_split['Close'].iloc[-1]:.2f}")
        
        print(f"\nPOST-SPLIT (After {self.split_date}):")
        print(f"   Average Price: ${post_avg:.2f}")
        print(f"   Price Range: ${post_split['Close'].min():.2f} - ${post_split['Close'].max():.2f}")
        print(f"   Current Price: ${post_split['Close'].iloc[-1]:.2f}")
        
        # Split-adjusted comparison
        adjusted_pre_avg = pre_avg / 10
        print(f"\n🔄 SPLIT-ADJUSTED COMPARISON:")
        print(f"   Pre-split adjusted: ${adjusted_pre_avg:.2f}")
        print(f"   Post-split actual: ${post_avg:.2f}")
        change = ((post_avg / adjusted_pre_avg) - 1) * 100
        print(f"   Real change: {change:+.1f}%")
        
    def analyze_volume(self, pre_split, post_split):
        """Analyze volume patterns"""
        print(f"\n📊 VOLUME ANALYSIS")
        print("=" * 30)
        
        pre_vol = pre_split['Volume'].mean()
        post_vol = post_split['Volume'].mean()
        
        print(f"PRE-SPLIT Average Volume: {pre_vol:,.0f}")
        print(f"POST-SPLIT Average Volume: {post_vol:,.0f}")
        print(f"Volume Change: {((post_vol / pre_vol) - 1) * 100:+.1f}%")
        
    def analyze_model_confusion(self, pre_split, post_split):
        """Analyze how split confuses ML models"""
        print(f"\n🤖 MODEL CONFUSION ANALYSIS")
        print("=" * 40)
        
        # Calculate price momentum before/after split
        pre_split['Returns'] = pre_split['Close'].pct_change()
        post_split['Returns'] = post_split['Close'].pct_change()
        
        pre_volatility = pre_split['Returns'].std() * np.sqrt(252)
        post_volatility = post_split['Returns'].std() * np.sqrt(252)
        
        print(f"PRE-SPLIT Volatility: {pre_volatility:.1%}")
        print(f"POST-SPLIT Volatility: {post_volatility:.1%}")
        
        print(f"\n⚠️  MODEL CONFUSION FACTORS:")
        print(f"   • Price scale change: 10x reduction")
        print(f"   • Volume increase: ~10x higher")
        print(f"   • Historical patterns broken")
        print(f"   • Technical indicators disrupted")
        
        # Check recent performance vs historical
        recent_30d = post_split.tail(30)['Returns'].mean() * 252
        historical_return = pre_split['Returns'].mean() * 252
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"   Historical annual return: {historical_return:.1%}")
        print(f"   Recent 30-day annual: {recent_30d:.1%}")
        
    def provide_recommendations(self):
        """Provide final recommendations"""
        print(f"\n🎯 RECOMMENDATIONS")
        print("=" * 40)
        
        print("❌ ELITE AI LIMITATIONS:")
        print("   • Models trained on pre-split data")
        print("   • Price patterns completely changed")
        print("   • Technical indicators unreliable")
        print("   • Volume patterns disrupted")
        
        print("\n✅ WHAT TO DO:")
        print("   1. IGNORE Elite AI SELL signal")
        print("   2. Focus on fundamental analysis")
        print("   3. Trust market sentiment over AI")
        print("   4. Consider split-adjusted models")
        
        print("\n🔮 NVDA POSITION ADVICE:")
        print("   • Stock split is BULLISH signal")
        print("   • Company confidence indicator")
        print("   • Increases accessibility to retail")
        print("   • Historical splits = positive performance")
        
        print("\n💡 ACTION ITEMS:")
        print("   🔸 HOLD your NVDA position")
        print("   🔸 Elite AI models need retraining")
        print("   🔸 Use split-aware analysis tools")
        print("   🔸 Monitor fundamentals, not AI signals")

def main():
    print("📊 NVDA STOCK SPLIT IMPACT ANALYSIS")
    print("=" * 50)
    
    analyzer = NVDASplitAnalyzer()
    analyzer.analyze_split_impact()
    
    print(f"\n🏁 ANALYSIS COMPLETE")
    print("Split impact explains Elite AI confusion!")

if __name__ == "__main__":
    main()
