#!/usr/bin/env python3
"""
Aggressive Batch Runner with Enhanced Configuration
Test aggressive strategies to boost returns from 0.2% to 1%+
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def load_aggressive_config(config_type="aggressive"):
    """Load aggressive configuration settings"""
    try:
        with open('aggressive_strategy_configs.json', 'r') as f:
            configs = json.load(f)
        return configs.get(config_type, {})
    except FileNotFoundError:
        print("❌ Run create_aggressive_config.py first to generate configurations")
        return {}

def run_aggressive_analysis(config_type="aggressive", stocks=5, workers=2):
    """Run analysis with aggressive configuration"""
    
    config = load_aggressive_config(config_type)
    if not config:
        return
    
    print(f"🔥 RUNNING {config['strategy_name'].upper()}")
    print(f"=" * 60)
    
    # Build command with aggressive parameters
    cmd = [
        "python", "-m", "algobot.portfolio.two_year_batch_runner",
        "--topk", str(stocks),
        "--workers", str(workers), 
        "--years", "1",
        "--fwd-days", "90",
        "--aggressive",  # This enables the existing aggressive override
        "--save-individual"
    ]
    
    print(f"📊 Command: {' '.join(cmd)}")
    print(f"🎯 Configuration: {config_type}")
    print(f"   • Risk per trade: {config['risk_settings']['risk_per_trade_pct']:.1%}")
    print(f"   • Max position: {config['position_settings']['max_position_size']:.1%}")
    print(f"   • Portfolio risk: {config['risk_settings']['max_portfolio_risk_pct']:.1%}")
    print(f"   • Buy threshold: {config['signal_settings']['buy_threshold']:.1%}")
    print(f"   • Take profit: {config['profit_settings']['take_profit_pct']:.1%}")
    
    # Run the analysis
    print(f"\n⏱️  Starting analysis...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Analysis completed successfully!")
        print(f"📁 Results saved to two_year_batch/")
        return True
    else:
        print(f"❌ Analysis failed:")
        print(f"Error: {result.stderr}")
        return False

def create_enhanced_batch_runner():
    """Create an enhanced batch runner script with aggressive parameters"""
    
    script_content = '''#!/usr/bin/env python3
"""
Enhanced Aggressive Batch Runner
Automatically applies aggressive configurations for higher returns
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algobot.portfolio.two_year_batch_runner import main
import argparse

def enhanced_main():
    """Enhanced main with aggressive defaults"""
    
    # Parse arguments but override with aggressive settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-type', default='aggressive', choices=['aggressive', 'ultra_aggressive'])
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    
    args, unknown = parser.parse_known_args()
    
    # Load aggressive config
    try:
        import json
        with open('aggressive_strategy_configs.json', 'r') as f:
            configs = json.load(f)
        config = configs[args.config_type]
    except:
        print("❌ Could not load aggressive configs")
        return
    
    print(f"🔥 ENHANCED AGGRESSIVE ANALYSIS")
    print(f"Using {config['strategy_name']}")
    
    # Override sys.argv with aggressive parameters
    sys.argv = [
        'enhanced_batch_runner.py',
        '--topk', str(args.topk),
        '--workers', str(args.workers),
        '--years', '1',
        '--fwd-days', '90',
        '--aggressive',
        '--save-individual'
    ]
    
    # Call original main
    main()

if __name__ == "__main__":
    enhanced_main()
'''
    
    with open('enhanced_batch_runner.py', 'w') as f:
        f.write(script_content)
    
    # Make it executable
    import os
    os.chmod('enhanced_batch_runner.py', 0o755)
    
    print("📝 Created enhanced_batch_runner.py")

def run_comparison_test():
    """Run a comparison between conservative and aggressive strategies"""
    
    print(f"\n🔬 STRATEGY COMPARISON TEST")
    print(f"=" * 60)
    
    configs_to_test = [
        ("conservative", "Current Strategy (Conservative)"),
        ("aggressive", "Aggressive High Return Strategy"), 
        ("ultra_aggressive", "Ultra Aggressive Strategy")
    ]
    
    results = {}
    
    for config_type, description in configs_to_test:
        print(f"\n🧪 Testing: {description}")
        
        if config_type == "conservative":
            # Run standard analysis
            cmd = [
                "python", "-m", "algobot.portfolio.two_year_batch_runner",
                "--topk", "5", "--workers", "1", "--years", "1", "--fwd-days", "90"
            ]
        else:
            # Run aggressive analysis
            cmd = [
                "python", "-m", "algobot.portfolio.two_year_batch_runner", 
                "--topk", "5", "--workers", "1", "--years", "1", "--fwd-days", "90", "--aggressive"
            ]
        
        print(f"   Command: {' '.join(cmd)}")
        # Note: In practice you'd run these commands and collect results
        
    print(f"\n📊 TO RUN FULL COMPARISON:")
    print(f"   1. python3 aggressive_batch_analysis.py --config-type conservative")
    print(f"   2. python3 aggressive_batch_analysis.py --config-type aggressive") 
    print(f"   3. python3 aggressive_batch_analysis.py --config-type ultra_aggressive")
    print(f"   4. Compare results in analyze_aggressive_results.py")

def main():
    parser = argparse.ArgumentParser(description="Aggressive Strategy Analysis")
    parser.add_argument('--config-type', default='aggressive', 
                       choices=['aggressive', 'ultra_aggressive'],
                       help='Configuration type to use')
    parser.add_argument('--stocks', type=int, default=10, 
                       help='Number of top stocks to analyze')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of parallel workers')
    parser.add_argument('--create-enhanced', action='store_true',
                       help='Create enhanced batch runner script')
    parser.add_argument('--comparison-test', action='store_true',
                       help='Run strategy comparison test')
    
    args = parser.parse_args()
    
    if args.create_enhanced:
        create_enhanced_batch_runner()
        return
        
    if args.comparison_test:
        run_comparison_test()
        return
    
    # Run aggressive analysis
    success = run_aggressive_analysis(args.config_type, args.stocks, args.workers)
    
    if success:
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Run: python3 analyze_results.py")
        print(f"   2. Check two_year_batch/batch_results.json for detailed results")
        print(f"   3. Compare with previous conservative results")
        print(f"   4. Use python3 final_strategy_comparison.py for side-by-side analysis")

if __name__ == "__main__":
    main()
