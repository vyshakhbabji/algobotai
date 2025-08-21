#!/usr/bin/env python3
"""
Strategy Debug Analysis
Figure out why the strategy isn't trading at all
"""
import json
import sys
import os

def check_results_detail():
    """Check detailed results to understand why no trading occurred"""
    
    print("🔍 DETAILED RESULTS ANALYSIS")
    print("=" * 60)
    
    try:
        with open('two_year_batch/batch_results.json', 'r') as f:
            results = json.load(f)
        
        print(f"📊 Full Results Structure:")
        print(json.dumps(results, indent=2))
        
        # Check for individual files
        batch_dir = "two_year_batch"
        if os.path.exists(batch_dir):
            files = os.listdir(batch_dir)
            print(f"\n📁 Files in {batch_dir}:")
            for file in files:
                print(f"   {file}")
                
                # If it's a JSON file for NVDA, read it
                if 'nvda' in file.lower() and file.endswith('.json'):
                    try:
                        with open(os.path.join(batch_dir, file), 'r') as f:
                            nvda_detail = json.load(f)
                        print(f"\n📋 {file} content:")
                        print(json.dumps(nvda_detail, indent=2)[:1000] + "..." if len(str(nvda_detail)) > 1000 else json.dumps(nvda_detail, indent=2))
                    except Exception as e:
                        print(f"   ❌ Could not read {file}: {e}")
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")

def diagnose_trading_issues():
    """Diagnose why the strategy isn't trading"""
    
    print(f"\n🔧 POTENTIAL ISSUES:")
    print("=" * 60)
    
    print(f"1. SIGNAL GENERATION PROBLEMS:")
    print(f"   • ML model may not be generating any buy/sell signals")
    print(f"   • Training data might be insufficient")
    print(f"   • Model confidence too low to trigger trades")
    
    print(f"\n2. THRESHOLD PROBLEMS:")
    print(f"   • Even with extreme thresholds (22%), no signals")
    print(f"   • Probability calculations might be broken")
    print(f"   • Signal strength below minimum requirements")
    
    print(f"\n3. DATA PROBLEMS:")
    print(f"   • NVDA data might be missing or corrupted")
    print(f"   • Forward test period might have no valid trading days")
    print(f"   • Price data quality issues")
    
    print(f"\n4. CONFIGURATION PROBLEMS:")
    print(f"   • Risk management preventing all trades")
    print(f"   • Position sizing calculations failing")
    print(f"   • Conflicting parameters causing logic errors")

def create_minimal_test():
    """Create a minimal test to isolate the problem"""
    
    print(f"\n🧪 MINIMAL TEST APPROACH:")
    print("=" * 60)
    
    print(f"Step 1: Test with no constraints")
    print(f"   • Set all thresholds to 0%")
    print(f"   • Remove all risk limits")
    print(f"   • Force trading regardless of signals")
    
    print(f"\nStep 2: Test signal generation directly")
    print(f"   • Run just the ML model part")
    print(f"   • Check if predictions are being generated")
    print(f"   • Verify probability outputs")
    
    print(f"\nStep 3: Test with buy-and-hold fallback")
    print(f"   • If no signals, just buy and hold")
    print(f"   • This should at least match market performance")

def suggest_immediate_fixes():
    """Suggest immediate things to try"""
    
    print(f"\n⚡ IMMEDIATE FIXES TO TRY:")
    print("=" * 60)
    
    print(f"1. RESTORE AND TRY CONSERVATIVE:")
    print(f"   python3 extreme_config_final.py --restore")
    print(f"   python -m algobot.portfolio.two_year_batch_runner --topk 1 --workers 1")
    
    print(f"\n2. CHECK IF ANY STOCKS TRADE:")
    print(f"   python -m algobot.portfolio.two_year_batch_runner --topk 10 --workers 1")
    
    print(f"\n3. TRY DIFFERENT TIME PERIODS:")
    print(f"   python -m algobot.portfolio.two_year_batch_runner --topk 1 --workers 1 --years 2 --fwd-days 30")
    
    print(f"\n4. ENABLE ALL TRADING (if possible):")
    print(f"   # Force buy-and-hold if no signals")
    print(f"   # Override minimum confidence requirements")
    
    print(f"\n5. CHECK LOGS:")
    print(f"   # Look for error messages in strategy execution")
    print(f"   # Check if ML model is working")

def main():
    check_results_detail()
    diagnose_trading_issues()
    create_minimal_test()
    suggest_immediate_fixes()
    
    print(f"\n🎯 NEXT STEPS:")
    print("=" * 60)
    print(f"The strategy appears to have a fundamental issue where it's not trading at all.")
    print(f"This is likely NOT a threshold problem but a deeper issue with:")
    print(f"   • Signal generation")
    print(f"   • Data quality") 
    print(f"   • Model training")
    print(f"   • Configuration conflicts")
    
    print(f"\nI recommend:")
    print(f"   1. First restore original settings and see if ANYTHING trades")
    print(f"   2. If nothing trades, there's a bug in the strategy itself")
    print(f"   3. May need to debug the ML model and signal generation")
    print(f"   4. Consider a simpler buy-and-hold test to verify basic functionality")

if __name__ == "__main__":
    main()
