#!/usr/bin/env python3
"""CLI runner to execute a full model scan across the configured universe.

Usage:
  python run_model_scan.py
"""
from algobot.analysis.model_scan import scan_universe


def main():
    df = scan_universe()
    if df is None or df.empty:
        print("No results produced.")
        return 1
    print("Scan complete. Rows:", len(df))
    # Basic summary
    signals = df['signal'].dropna()
    if not signals.empty:
        print("Signal counts:")
        print(signals.value_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
