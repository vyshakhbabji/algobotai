#!/usr/bin/env python3
"""
Simple test of Elite ML system to debug data flow issues
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the backtrader_system directory to the path
sys.path.append('/Users/vyshakhbabji/Desktop/AlgoTradingBot')

def test_data_flow():
    """Test the data flow of our Elite ML system"""
    
    print("🔧 Testing Elite ML System Data Flow...")
    
    # Download test data
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1000)  # About 3 years
    
    print(f"📈 Downloading {symbol} data from {start_date.date()} to {end_date.date()}")
    
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        print(f"✅ Downloaded {len(data)} bars for {symbol}")
        
        # Basic data info
        print(f"   Data range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Sample data:")
        print(data.head(2))
        
        # Test feature calculation
        from backtrader_system.strategies.enhanced_ml_signal_generator import EnhancedMLSignalGenerator
        
        print("\n🧠 Testing ML Signal Generator...")
        
        # Create minimal config
        config = {
            'signal_threshold': 0.15,
            'max_position_size': 0.15,
            'use_ensemble': True
        }
        
        ml_gen = EnhancedMLSignalGenerator(config)
        
        # Prepare data format
        df = data.copy()
        df.reset_index(inplace=True)
        
        # Handle multi-level column names from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]  # Take first level
        
        # Ensure we have the right column names
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = df.columns.tolist()
        print(f"   Available columns: {available_cols}")
        
        # Rename columns if needed
        if 'Adj Close' in df.columns and 'Close' not in df.columns:
            df['Close'] = df['Adj Close']
        
        # Select only the columns we need
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.set_index('Date', inplace=True)
        
        print(f"   Prepared dataframe shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Test feature calculation
        print("\n⚙️  Testing feature calculation...")
        try:
            df_with_features = ml_gen.calculate_advanced_features(df.copy(), symbol)
            print(f"✅ Features calculated: {df_with_features.shape}")
            print(f"   New columns: {len(df_with_features.columns) - len(df.columns)}")
            print(f"   NaN count: {df_with_features.isna().sum().sum()}")
            
            # Test target creation
            print("\n🎯 Testing target creation...")
            df_with_targets = ml_gen.create_enhanced_targets(df_with_features.copy())
            print(f"✅ Targets created: {df_with_targets.shape}")
            
            # Check data after cleaning
            clean_data = df_with_targets.dropna()
            print(f"   Clean data: {clean_data.shape}")
            print(f"   Dropped rows: {len(df_with_targets) - len(clean_data)}")
            
            if len(clean_data) > 50:
                print("✅ Sufficient data for training!")
                
                # Test feature selection
                feature_cols = [col for col in clean_data.columns 
                              if not col.startswith('future_') 
                              and col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'ml_signal_strength', 'direction_target', 'regime_target', 'prob_profit']]
                
                print(f"   Feature columns: {len(feature_cols)}")
                print(f"   Target columns: ['ml_signal_strength', 'direction_target', 'regime_target']")
                
                if len(feature_cols) > 10:
                    print("✅ Elite ML System data flow working correctly!")
                    return True
                else:
                    print("❌ Not enough features generated")
                    return False
            else:
                print("❌ Insufficient clean data after processing")
                return False
                
        except Exception as e:
            print(f"❌ Error in feature calculation: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return False

if __name__ == "__main__":
    success = test_data_flow()
    if success:
        print("\n🚀 Elite ML System is ready for backtesting!")
    else:
        print("\n🔧 Need to fix issues before backtesting")
