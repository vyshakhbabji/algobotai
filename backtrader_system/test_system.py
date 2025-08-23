#!/usr/bin/env python3
"""
Test script for Backtrader ML Trading System
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError:
        print("❌ pandas - run: pip install pandas")
        return False
    
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError:
        print("❌ numpy - run: pip install numpy")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn")
    except ImportError:
        print("❌ scikit-learn - run: pip install scikit-learn")
        return False
    
    try:
        import yfinance as yf
        print("✅ yfinance")
    except ImportError:
        print("❌ yfinance - run: pip install yfinance")
        return False
    
    try:
        import yaml
        print("✅ PyYAML")
    except ImportError:
        print("❌ PyYAML - run: pip install PyYAML")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib")
    except ImportError:
        print("❌ matplotlib - run: pip install matplotlib")
        return False
    
    try:
        import backtrader as bt
        print("✅ backtrader")
    except ImportError:
        print("❌ backtrader - run: pip install backtrader")
        return False
    
    return True

def test_signal_generator():
    """Test the ML signal generator"""
    print("\nTesting signal generator...")
    
    try:
        from strategies.signal_generator import MLSignalGenerator
        
        # Create test config
        config = {
            'min_training_days': 60,
            'feature_columns': [
                'price_vs_ma5', 'price_vs_ma20', 'rsi_normalized'
            ]
        }
        
        generator = MLSignalGenerator(config)
        symbols = generator.get_elite_stocks()
        
        assert len(symbols) == 20, f"Expected 20 stocks, got {len(symbols)}"
        assert 'AAPL' in symbols, "AAPL should be in elite stocks"
        
        print(f"✅ Signal generator works - {len(symbols)} elite stocks loaded")
        return True
        
    except Exception as e:
        print(f"❌ Signal generator failed: {e}")
        return False

def test_data_download():
    """Test data downloading"""
    print("\nTesting data download...")
    
    try:
        import yfinance as yf
        import pandas as pd
        
        # Test downloading a single stock
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1mo")
        
        assert not data.empty, "No data downloaded"
        assert len(data) > 10, f"Too little data: {len(data)} bars"
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            assert col in data.columns, f"Missing column: {col}"
        
        print(f"✅ Data download works - {len(data)} bars for AAPL")
        return True
        
    except Exception as e:
        print(f"❌ Data download failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        import yaml
        
        config_path = Path(__file__).parent / "config" / "backtest_config.yaml"
        
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['backtest', 'strategy', 'ml_config']
        for section in required_sections:
            assert section in config, f"Missing config section: {section}"
        
        # Check key parameters
        assert 'initial_capital' in config['backtest']
        assert 'signal_threshold' in config['strategy']
        
        print("✅ Configuration loading works")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_technical_indicators():
    """Test technical indicator calculation"""
    print("\nTesting technical indicators...")
    
    try:
        from strategies.signal_generator import MLSignalGenerator
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        
        data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        generator = MLSignalGenerator({})
        result = generator.calculate_technical_indicators(data)
        
        # Check that indicators were added
        expected_indicators = [
            'ma_5', 'ma_20', 'rsi', 'price_vs_ma5', 
            'bb_position', 'macd_histogram'
        ]
        
        for indicator in expected_indicators:
            assert indicator in result.columns, f"Missing indicator: {indicator}"
        
        print("✅ Technical indicators calculation works")
        return True
        
    except Exception as e:
        print(f"❌ Technical indicators failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Backtrader ML Trading System")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loading,
        test_signal_generator,
        test_data_download,
        test_technical_indicators
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python backtest_runner.py")
        print("2. Check results in ./results/ directory")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed. Please fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
