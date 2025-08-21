#!/usr/bin/env python3
"""
Debug script to see what signals are being generated
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algobot.live.paper_trade_runner import download_recent, _signal
from algobot.config import GLOBAL_CONFIG
import pandas as pd

def debug_signals():
    # Get the universe
    symbols = list(GLOBAL_CONFIG.universe.core_universe)[:GLOBAL_CONFIG.universe.max_universe]
    print(f"Universe: {symbols}")
    
    # Download recent data
    print("Downloading recent data...")
    data = download_recent(symbols)
    print(f"Got data for {len(data)} symbols: {list(data.keys())}")
    
    if not data:
        print("No data available!")
        return
    
    # Common latest date
    common = None
    for df in data.values():
        idx = set(df.index)
        common = idx if common is None else common.intersection(idx)
    dates = sorted(list(common))
    today = dates[-1]
    print(f"Latest common date: {today}")
    
    # Build signals and show details
    signals = {}
    print("\n===== SIGNAL ANALYSIS =====")
    for sym, df in data.items():
        try:
            i = df.index.get_loc(today)
            if i >= 30:
                sig = _signal(df, i)
                signals[sym] = sig
                print(f"\n{sym}:")
                print(f"  Signal: {sig['signal']}")
                print(f"  Strength: {sig['strength']:.3f}")
                print(f"  Buy Strength: {sig['buy_strength']:.3f}")
                print(f"  Sell Strength: {sig['sell_strength']:.3f}")
                print(f"  Momentum Consistency: {sig['momentum_consistency']:.3f}")
                print(f"  Volatility: {sig['volatility']:.3f}")
                print(f"  Price: ${sig['price']:.2f}")
        except Exception as e:
            print(f"Error processing {sym}: {e}")
    
    # Show buy candidates
    print(f"\n===== BUY CANDIDATES =====")
    buy_cands = []
    for sym, sig in signals.items():
        if sig['signal'] == 'BUY' and sig['strength'] >= 0.4:  # Updated threshold
            score = sig['strength'] * max(0.0, sig['momentum_consistency'])
            buy_cands.append((score, sym, sig))
    
    buy_cands.sort(reverse=True)
    if buy_cands:
        print(f"Found {len(buy_cands)} BUY candidates with strength >= 0.4:")
        for score, sym, sig in buy_cands:
            print(f"  {sym}: score={score:.3f}, strength={sig['strength']:.3f}, momentum={sig['momentum_consistency']:.3f}")
    else:
        print("No BUY candidates found (need signal='BUY' AND strength >= 0.4)")
        # Show close misses
        close_buys = [(sym, sig) for sym, sig in signals.items() if sig['signal'] == 'BUY']
        if close_buys:
            print("BUY signals but with low strength:")
            for sym, sig in close_buys:
                print(f"  {sym}: strength={sig['strength']:.3f} (need >= 0.4)")
    
    print(f"\n===== SUMMARY =====")
    signal_counts = {}
    for sig in signals.values():
        signal_counts[sig['signal']] = signal_counts.get(sig['signal'], 0) + 1
    
    for signal_type, count in signal_counts.items():
        print(f"{signal_type}: {count}")

if __name__ == "__main__":
    debug_signals()
