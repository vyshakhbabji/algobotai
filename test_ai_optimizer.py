#!/usr/bin/env python3
"""
Quick Test of AI Self-Optimizer
"""

import sys
import os
sys.path.append('pages')

from ai_optimizer import SelfImprovingAI

def test_ai_optimizer():
    print("🧠 Testing AI Self-Optimizer...")
    
    # Initialize
    ai_optimizer = SelfImprovingAI()
    
    # Test with AAPL
    print("\n📊 Testing performance evaluation for AAPL...")
    performance = ai_optimizer.evaluate_model_performance('AAPL', days_back=14)
    
    if performance:
        print(f"✅ AAPL Performance Evaluation:")
        print(f"   - Performance Score: {performance['performance_score']:.3f}")
        print(f"   - Direction Accuracy: {performance['direction_accuracy']:.1%}")
        print(f"   - Win Rate: {performance['win_rate']:.1%}")
        print(f"   - Buy Signals: {performance['buy_signals']}")
        print(f"   - Needs Improvement: {performance['needs_improvement']}")
        
        # Test optimization if needed
        if performance['needs_improvement']:
            print(f"\n⚡ Running optimization for AAPL...")
            optimization = ai_optimizer.optimize_model_parameters('AAPL', performance)
            if optimization:
                print(f"✅ Optimization complete!")
                print(f"   - New Model: {optimization['new_model']}")
                print(f"   - R² Improvement: {optimization['improvement']:+.3f}")
            else:
                print("❌ Optimization failed")
        else:
            print("✅ AAPL performing well - no optimization needed")
    else:
        print("❌ Could not evaluate AAPL performance")
    
    print("\n🎯 AI Optimizer test complete!")

if __name__ == "__main__":
    test_ai_optimizer()
