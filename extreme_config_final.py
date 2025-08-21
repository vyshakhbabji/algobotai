#!/usr/bin/env python3
"""
EXTREME MAXIMUM RETURN CONFIGURATION
Final attempt to create a high-return trading strategy
WARNING: This is extremely aggressive and for testing only
"""
import shutil
import os

def create_extreme_batch_runner():
    """Create the most extreme aggressive parameters possible"""
    
    # Read current file
    original_file = "algobot/portfolio/two_year_batch_runner.py"
    
    with open(original_file, 'r') as f:
        content = f.read()
    
    # Define EXTREME parameters - designed to trade frequently with large positions
    extreme_params = '''                # EXTREME MAXIMUM RETURN - trades aggressively with large positions
                res = run_nvda_institutional(symbol=sym,
                                             train_start=str(derived_train_start.date()),
                                             train_end=str(derived_train_end.date()),
                                             fwd_start=str(derived_fwd_start.date()),
                                             fwd_end=str(end_date.date()),
                                             chart=False,
                                             classification_map=class_map,
                                             symbol_classification='auto',
                                             tier_mode='static',
                                             tier_probs=(0.25,0.30,0.35,0.42),  # VERY LOW - trades on any signal
                                             breakout_prob=0.22,               # 22% breakout (vs 42%)
                                             hard_exit_prob=0.20,              # 20% exit (vs 35%)
                                             soft_exit_prob=0.25,              # 25% soft exit
                                             partial_derisk_prob=0.30,         # Later de-risking
                                             early_core_fraction=0.95,         # 95% initial position
                                             atr_initial_mult=0.6,             # VERY tight stops (0.6x ATR)
                                             atr_trail_mult=1.0,               # Tight trail (1.0x ATR)
                                             performance_guard_days=5,          # Short guard period
                                             performance_guard_min_capture=0.01, # 1% capture minimum
                                             performance_guard_max_fraction=0.30, # More aggressive guard
                                             enable_expectancy_guard=False,
                                             risk_per_trade_pct=0.20,          # 20% risk per trade
                                             risk_ceiling=0.25,                # 25% max risk
                                             risk_increment=0.05,              # 5% increments
                                             profit_ladder=(0.05,0.15,0.40,1.00), # Let winners run to 100%
                                             profit_trim_fractions=(0.02,0.03,0.05,0.08), # Tiny trims
                                             fast_scale_gain_threshold=0.015,  # 1.5% for fast scaling
                                             min_holding_days=0,               # Same-day exits allowed
                                             time_scalein_days=0,              # Immediate full position
                                             momentum_20d_threshold=0.015,     # 1.5% momentum threshold
                                             adjust_trim_ladder=True,
                                             enable_quality_filter=False,
                                             enable_prob_ema_exit=False,
                                             enable_vol_adaptive_trail=False,  # Disable adaptive (keep tight)
                                             stale_days=15,                    # 15 days max hold
                                             stale_min_runup=0.03,             # 3% minimum for stale
                                             enable_pullback_reentry=True,
                                             finalize_at_end=True)'''
    
    # Replace the aggressive section
    old_aggressive_start = "            if args.aggressive:"
    
    start_idx = content.find(old_aggressive_start)
    if start_idx == -1:
        print("❌ Could not find aggressive section")
        return False
    
    # Find where the aggressive section ends (look for 'else:')
    else_idx = content.find("            else:", start_idx)
    if else_idx == -1:
        print("❌ Could not find else section")
        return False
    
    # Replace everything between if and else
    new_content = (
        content[:start_idx] + 
        "            if args.aggressive:\n" +
        extreme_params + "\n" +
        content[else_idx:]
    )
    
    # Write the modified file
    with open(original_file, 'w') as f:
        f.write(new_content)
    
    print(f"⚡ EXTREME CONFIGURATION APPLIED")
    print(f"=" * 60)
    print(f"🔥 MAXIMUM AGGRESSIVENESS SETTINGS:")
    print(f"   • Tier probabilities: (25%, 30%, 35%, 42%) - trades on weak signals")
    print(f"   • Breakout threshold: 22% (was 42%) - enters on small moves")
    print(f"   • Exit threshold: 20% (was 35%) - holds longer")
    print(f"   • Risk per trade: 20% (was 12%) - large positions")
    print(f"   • Initial position: 95% (was 85%) - almost full size immediately")
    print(f"   • Stop loss: 0.6x ATR (was 1.3x) - very tight stops")
    print(f"   • Fast scaling: 1.5% gain (was 4%) - doubles position quickly")
    print(f"   • Min holding: 0 days (was 1) - same-day exits allowed")
    print(f"   • Profit targets: 5%, 15%, 40%, 100% - let winners run big")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='Apply extreme configuration')
    parser.add_argument('--test-single', action='store_true', help='Test on single best stock')
    parser.add_argument('--restore', action='store_true', help='Restore from backup')
    
    args = parser.parse_args()
    
    if args.restore:
        backup_file = "algobot/portfolio/two_year_batch_runner.py.backup"
        original_file = "algobot/portfolio/two_year_batch_runner.py"
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, original_file)
            print("✅ Restored original configuration")
        else:
            print("❌ No backup found")
        return
    
    if args.apply or args.test_single:
        success = create_extreme_batch_runner()
        
        if success and args.test_single:
            print(f"\n🧪 TESTING ON SINGLE STOCK (NVDA):")
            print(f"Command: python -m algobot.portfolio.two_year_batch_runner --topk 1 --universe-file <(echo 'NVDA') --workers 1 --aggressive")
            print(f"\n📊 Expected results with extreme config:")
            print(f"   • Market exposure: 60-90% (vs current 17%)")
            print(f"   • More frequent trades")
            print(f"   • Larger position sizes") 
            print(f"   • Higher volatility but potentially much higher returns")
            
        print(f"\n⚠️  EXTREME RISK WARNING:")
        print(f"   • This config can cause 50%+ losses in bad markets")
        print(f"   • Only use small amounts for testing")
        print(f"   • Monitor positions closely")
        print(f"   • Be prepared to exit manually if needed")
        
        print(f"\n🎯 SUCCESS CRITERIA:")
        print(f"   • Target exposure: >60%")
        print(f"   • Target returns: >1% in 3-month period")
        print(f"   • Target capture ratio: >40%")
    else:
        print(f"Usage:")
        print(f"  --apply        Apply extreme configuration")
        print(f"  --test-single  Apply config and show test command")
        print(f"  --restore      Restore original configuration")

if __name__ == "__main__":
    main()
