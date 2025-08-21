#!/usr/bin/env python3
"""
Ultra Aggressive Batch Runner Override
Patch the batch runner to use truly aggressive parameters for maximum returns
"""
import shutil
import os
from pathlib import Path

def create_ultra_aggressive_batch_runner():
    """Create a modified batch runner with ultra-aggressive parameters"""
    
    # Make a backup of the original
    original_file = "algobot/portfolio/two_year_batch_runner.py"
    backup_file = "algobot/portfolio/two_year_batch_runner.py.backup"
    
    if not os.path.exists(backup_file):
        shutil.copy2(original_file, backup_file)
        print(f"✅ Created backup: {backup_file}")
    
    # Read the original file
    with open(original_file, 'r') as f:
        content = f.read()
    
    # Define ultra-aggressive parameters
    ultra_aggressive_params = '''                # ULTRA-AGGRESSIVE parameter overrides for maximum returns
                res = run_nvda_institutional(symbol=sym,
                                             train_start=str(derived_train_start.date()),
                                             train_end=str(derived_train_end.date()),
                                             fwd_start=str(derived_fwd_start.date()),
                                             fwd_end=str(end_date.date()),
                                             chart=False,
                                             classification_map=class_map,
                                             symbol_classification='auto',
                                             tier_mode='static',
                                             tier_probs=(0.42,0.47,0.52,0.58),  # Lower thresholds for more trades
                                             breakout_prob=0.42,               # Much lower breakout threshold
                                             hard_exit_prob=0.35,              # Lower exit threshold
                                             soft_exit_prob=0.40,              # Lower soft exit
                                             early_core_fraction=0.90,         # Higher initial position
                                             atr_initial_mult=1.3,             # Tighter stops for faster action
                                             atr_trail_mult=1.8,               # Tighter trailing stops
                                             performance_guard_days=10,         # Shorter performance guard
                                             performance_guard_min_capture=0.03, # Lower capture requirement
                                             performance_guard_max_fraction=0.40, # More aggressive guard
                                             enable_expectancy_guard=False,
                                             risk_per_trade_pct=0.12,          # 12% risk per trade (vs 5%)
                                             risk_ceiling=0.15,                # 15% max risk (vs 8%)
                                             risk_increment=0.02,              # 2% increments (vs 1%)
                                             profit_ladder=(0.08,0.18,0.35,0.60), # Lower profit thresholds, more levels
                                             profit_trim_fractions=(0.05,0.08,0.12,0.15), # Smaller trims to keep positions
                                             fast_scale_gain_threshold=0.04,   # 4% for fast scaling (vs 10%)
                                             min_holding_days=1,               # 1 day minimum (vs 7)
                                             time_scalein_days=1,              # Immediate scaling
                                             momentum_20d_threshold=0.04,      # 4% momentum threshold (vs 8%)
                                             adjust_trim_ladder=True,          # Enable trim adjustments
                                             enable_quality_filter=False,
                                             enable_prob_ema_exit=False,
                                             enable_vol_adaptive_trail=True,   # Enable adaptive trailing
                                             vol_trail_floor_mult=0.7,         # More aggressive trail floor
                                             adaptive_trail_mult_after_gain=1.2, # Less loosening after gains
                                             adaptive_trail_gain_threshold=0.15, # 15% threshold for adaptive trail
                                             stale_days=30,                    # 30 days vs 60
                                             stale_min_runup=0.06,             # 6% vs 10%
                                             enable_pullback_reentry=True)     # Enable reentry'''
    
    # Replace the aggressive section
    old_aggressive_start = "            if args.aggressive:"
    old_aggressive_end = "                                             enable_prob_ema_exit=False)"
    
    start_idx = content.find(old_aggressive_start)
    if start_idx == -1:
        print("❌ Could not find aggressive section to replace")
        return False
    
    # Find the end of the aggressive section
    end_idx = content.find(old_aggressive_end, start_idx)
    if end_idx == -1:
        print("❌ Could not find end of aggressive section")
        return False
    
    end_idx += len(old_aggressive_end)
    
    # Replace the section
    new_content = (
        content[:start_idx] + 
        "            if args.aggressive:\n" +
        ultra_aggressive_params + 
        content[end_idx:]
    )
    
    # Write the modified file
    with open(original_file, 'w') as f:
        f.write(new_content)
    
    print(f"🔥 ULTRA-AGGRESSIVE BATCH RUNNER CREATED")
    print(f"=" * 60)
    print(f"Modified parameters:")
    print(f"   • Risk per trade: 12% (was 5%)")
    print(f"   • Tier probabilities: (0.42,0.47,0.52,0.58) (was 0.48-0.66)")
    print(f"   • Breakout threshold: 42% (was ~50%)")
    print(f"   • Hard exit threshold: 35% (was ~45%)")
    print(f"   • Profit ladder: (8%,18%,35%,60%) (was 20%,40%,60%)")
    print(f"   • Fast scale threshold: 4% (was 10%)")
    print(f"   • Min holding days: 1 (was 7)")
    print(f"   • Momentum threshold: 4% (was 8%)")
    
    return True

def restore_original_batch_runner():
    """Restore the original batch runner from backup"""
    
    original_file = "algobot/portfolio/two_year_batch_runner.py"
    backup_file = "algobot/portfolio/two_year_batch_runner.py.backup"
    
    if os.path.exists(backup_file):
        shutil.copy2(backup_file, original_file)
        print(f"✅ Restored original batch runner from backup")
        return True
    else:
        print(f"❌ No backup found: {backup_file}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra Aggressive Batch Runner Modifier")
    parser.add_argument('--create', action='store_true', help='Create ultra-aggressive version')
    parser.add_argument('--restore', action='store_true', help='Restore original version')
    parser.add_argument('--test', action='store_true', help='Test ultra-aggressive configuration')
    
    args = parser.parse_args()
    
    if args.restore:
        restore_original_batch_runner()
    elif args.test:
        if create_ultra_aggressive_batch_runner():
            print(f"\n🧪 TESTING ULTRA-AGGRESSIVE CONFIGURATION")
            print(f"Run: python -m algobot.portfolio.two_year_batch_runner --topk 5 --workers 1 --years 1 --fwd-days 90 --aggressive")
            print(f"Then: python3 analyze_aggressive_results.py")
    elif args.create:
        create_ultra_aggressive_batch_runner()
    else:
        print(f"Usage:")
        print(f"  --create   : Create ultra-aggressive version")
        print(f"  --restore  : Restore original version")
        print(f"  --test     : Create and show test command")

if __name__ == "__main__":
    main()
