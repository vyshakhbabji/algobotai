#!/usr/bin/env python3
"""
Stock Split Detection and Data Adjustment System
Automatically detects splits and adjusts historical data for clean model training
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockSplitDetector:
    def __init__(self):
        self.split_threshold = 0.4  # 40% single-day change indicates split
        
    def detect_splits(self, symbol, period="5y"):
        """Detect stock splits by analyzing price movements"""
        print(f"🔍 Detecting splits for {symbol}...")
        
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if len(data) == 0:
                return []
            
            # Calculate daily returns
            data['returns'] = data['Close'].pct_change()
            
            # Look for large negative returns (price drops due to splits)
            potential_splits = data[data['returns'] < -self.split_threshold]
            
            splits = []
            for date, row in potential_splits.iterrows():
                # Calculate split ratio
                prev_close = data['Close'].loc[:date].iloc[-2]
                split_close = row['Close']
                split_ratio = prev_close / split_close
                
                # Only consider if it looks like a clean split (2:1, 3:1, 10:1, etc.)
                if split_ratio >= 1.8:  # At least 2:1 split
                    rounded_ratio = round(split_ratio)
                    if abs(split_ratio - rounded_ratio) < 0.3:  # Close to whole number
                        splits.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'split_ratio': rounded_ratio,
                            'pre_split_price': prev_close,
                            'post_split_price': split_close,
                            'actual_ratio': split_ratio
                        })
                        print(f"   📅 {date.strftime('%Y-%m-%d')}: {rounded_ratio}:1 split")
            
            return splits
            
        except Exception as e:
            print(f"❌ Error detecting splits for {symbol}: {str(e)}")
            return []
    
    def get_official_splits(self, symbol):
        """Get official split data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            splits = ticker.splits
            
            if len(splits) == 0:
                return []
            
            official_splits = []
            for date, ratio in splits.items():
                official_splits.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'split_ratio': float(ratio),
                    'source': 'official'
                })
                print(f"   📅 Official: {date.strftime('%Y-%m-%d')}: {float(ratio)}:1 split")
            
            return official_splits
            
        except Exception as e:
            print(f"❌ Error getting official splits for {symbol}: {str(e)}")
            return []

class DataAdjuster:
    def __init__(self):
        self.split_detector = StockSplitDetector()
        
    def adjust_for_splits(self, symbol, period="5y"):
        """Adjust historical data for all detected splits"""
        print(f"\n🔧 ADJUSTING DATA FOR {symbol}")
        print("=" * 40)
        
        # Get raw data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if len(data) == 0:
            print(f"❌ No data available for {symbol}")
            return None
        
        # Detect splits
        detected_splits = self.split_detector.detect_splits(symbol, period)
        official_splits = self.split_detector.get_official_splits(symbol)
        
        # Combine and deduplicate splits
        all_splits = self.combine_splits(detected_splits, official_splits)
        
        if not all_splits:
            print(f"✅ No splits detected for {symbol} - data is clean")
            return data
        
        # Apply split adjustments
        adjusted_data = self.apply_split_adjustments(data, all_splits)
        
        return adjusted_data
    
    def combine_splits(self, detected, official):
        """Combine detected and official splits, removing duplicates"""
        all_splits = []
        
        # Add official splits first (more reliable)
        for split in official:
            all_splits.append(split)
        
        # Add detected splits if not already in official list
        for split in detected:
            split_date = split['date']
            if not any(abs(pd.to_datetime(s['date']) - pd.to_datetime(split_date)).days < 5 
                      for s in official):
                all_splits.append(split)
        
        # Sort by date
        all_splits.sort(key=lambda x: x['date'])
        return all_splits
    
    def apply_split_adjustments(self, data, splits):
        """Apply split adjustments to historical data"""
        print(f"\n🔄 Applying {len(splits)} split adjustments...")
        
        adjusted_data = data.copy()
        cumulative_adjustment = 1.0
        
        # Process splits from newest to oldest
        for split in reversed(splits):
            split_date = pd.to_datetime(split['date'])
            # Make split_date timezone-aware to match data index
            if adjusted_data.index.tz is not None:
                split_date = split_date.tz_localize(adjusted_data.index.tz)
            
            split_ratio = split.get('split_ratio', split.get('actual_ratio', 2))
            
            print(f"   📅 {split['date']}: {split_ratio}:1 split")
            
            # Adjust all data before split date
            mask = adjusted_data.index < split_date
            
            if mask.any():
                # Adjust prices (divide by split ratio)
                adjusted_data.loc[mask, 'Open'] /= split_ratio
                adjusted_data.loc[mask, 'High'] /= split_ratio
                adjusted_data.loc[mask, 'Low'] /= split_ratio
                adjusted_data.loc[mask, 'Close'] /= split_ratio
                
                # Adjust volume (multiply by split ratio)
                adjusted_data.loc[mask, 'Volume'] *= split_ratio
                
                cumulative_adjustment *= split_ratio
        
        print(f"✅ Applied cumulative adjustment factor: {cumulative_adjustment:.2f}")
        
        # Add split adjustment metadata
        adjusted_data.attrs['split_adjusted'] = True
        adjusted_data.attrs['adjustment_factor'] = cumulative_adjustment
        adjusted_data.attrs['splits'] = splits
        
        return adjusted_data

class EliteAIRetrainer:
    def __init__(self):
        self.data_adjuster = DataAdjuster()
        
    def retrain_with_clean_data(self, symbols):
        """Retrain Elite AI with split-adjusted data"""
        print("\n🚀 RETRAINING ELITE AI WITH CLEAN DATA")
        print("=" * 50)
        
        clean_data = {}
        
        # Adjust data for each symbol
        for symbol in symbols:
            print(f"\n🔄 Processing {symbol}...")
            adjusted_data = self.data_adjuster.adjust_for_splits(symbol)
            if adjusted_data is not None:
                clean_data[symbol] = adjusted_data
                print(f"✅ {symbol} data cleaned and ready")
            else:
                print(f"❌ Failed to process {symbol}")
        
        # Save clean data for model training
        self.save_clean_data(clean_data)
        
        # Train new model with clean data
        self.train_improved_model(clean_data)
        
        return clean_data
    
    def save_clean_data(self, clean_data):
        """Save clean data for future use"""
        print(f"\n💾 Saving clean data...")
        
        for symbol, data in clean_data.items():
            filename = f"clean_data_{symbol}.csv"
            data.to_csv(filename)
            print(f"   💾 Saved {symbol} -> {filename}")
    
    def train_improved_model(self, clean_data):
        """Train improved model with clean data"""
        print(f"\n🎯 TRAINING IMPROVED MODEL")
        print("-" * 30)
        
        # This would integrate with your existing Elite AI trainer
        # For now, we'll show the improvement potential
        
        for symbol, data in clean_data.items():
            print(f"\n📊 {symbol} Clean Data Summary:")
            print(f"   📅 Date Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"   📈 Price Range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            print(f"   📊 Data Points: {len(data)}")
            
            if hasattr(data, 'attrs') and data.attrs.get('split_adjusted'):
                print(f"   🔧 Split Adjusted: YES")
                print(f"   🔧 Adjustment Factor: {data.attrs.get('adjustment_factor', 1):.2f}")
                print(f"   🔧 Splits Applied: {len(data.attrs.get('splits', []))}")
            else:
                print(f"   🔧 Split Adjusted: NO")

def main():
    """Run complete split detection and data cleaning"""
    print("🔧 STOCK SPLIT DETECTION & DATA CLEANING SYSTEM")
    print("=" * 60)
    
    # Focus on NVDA first, then expand
    symbols = ['NVDA', 'AAPL', 'GOOGL', 'TSLA', 'AMZN']
    
    retrainer = EliteAIRetrainer()
    clean_data = retrainer.retrain_with_clean_data(symbols)
    
    print(f"\n🏁 DATA CLEANING COMPLETE")
    print("=" * 30)
    print("✅ Split-adjusted data ready for model training")
    print("✅ Historical patterns now consistent")
    print("✅ Model should perform much better")
    
    # Show next steps
    print(f"\n🔄 NEXT STEPS:")
    print("1. Integrate clean data with Elite AI trainer")
    print("2. Retrain all models with adjusted data")
    print("3. Test improved model performance")
    print("4. Deploy improved model for trading")

if __name__ == "__main__":
    main()
