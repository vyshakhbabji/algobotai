"""
🎯 OPTIMAL ALPACA TEST WITH CORRECT DATES
Test our optimal strategy with:
- 15+ months training data (2023-01-01 to 2024-05-01)
- 3 months trading simulation (2024-05-01 to 2024-08-01)
- 100% Alpaca data within subscription limits
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from enhanced_alpaca_backtester_with_filtering import EnhancedAlpacaBacktesterWithFiltering

def main():
    print("🎯 OPTIMAL ALPACA STRATEGY TEST")
    print("=" * 60)
    
    # Initialize backtester
    backtester = EnhancedAlpacaBacktesterWithFiltering(initial_capital=100000)
    
    # OPTIMAL STRATEGY with available data
    data_start = "2023-01-01"      # 15+ months of training data
    trading_start = "2024-05-01"   # Start trading simulation here
    trading_end = "2024-08-01"     # 3 months of recent available data
    
    print(f"📊 Training period: {data_start} to {trading_start} (15+ months)")
    print(f"📈 Trading period: {trading_start} to {trading_end} (3 months)")
    print(f"🔥 Strategy: Long training + Recent trading with Alpaca data")
    print()
    
    # Run backtest
    try:
        results = backtester.run_enhanced_backtest(
            start_date=data_start,
            end_date=trading_end,
            max_symbols=50
        )
        
        if results:
            print(f"\n🎯 OPTIMAL ALPACA TEST RESULTS:")
            print(f"   📈 Total Return: {results.get('total_return', 0):.2%}")
            print(f"   💰 Final Value: ${results.get('final_value', 100000):,.2f}")
            print(f"   📊 Total Trades: {results.get('total_trades', 0)}")
            print(f"   🎯 Symbols Traded: {results.get('symbols_traded', 0)}")
            print(f"   🤖 Models Trained: {results.get('models_trained', 0)}")
            
            if results.get('model_stats'):
                print(f"   📈 Avg Model Accuracy: {results.get('avg_model_accuracy', 0):.1%}")
                
                # Show top performing models
                print(f"\n🏆 TOP PERFORMING MODELS:")
                sorted_models = sorted(results['model_stats'], key=lambda x: x['accuracy'], reverse=True)[:5]
                for i, model in enumerate(sorted_models, 1):
                    print(f"   {i}. {model['symbol']}: {model['accuracy']:.1%}")
        else:
            print("❌ No results returned")
            
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
