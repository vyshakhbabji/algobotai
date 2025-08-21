"""CLI to run walk-forward evaluation across a configured universe.

Usage (example):
  python run_walk_forward.py --symbols AAPL MSFT NVDA --data-dir . --train 200 --test 20 --step 20 --out walk_forward_results.csv

If clean data CSVs like clean_data_SYMBOL.csv exist in data-dir, they are used.
Otherwise attempts to download via yfinance.
"""
from __future__ import annotations
import argparse
import os
import sys
import pandas as pd
import yfinance as yf
from typing import Dict

from algobot.forward_test.walk_forward_eval import evaluate_universe_walk_forward


def load_symbol(symbol: str, data_dir: str) -> pd.DataFrame:
    fname_variants = [
        f"clean_data_{symbol}.csv",
        f"clean_data_{symbol.upper()}.csv",
        f"clean_data_{symbol.lower()}.csv",
    ]
    for fn in fname_variants:
        path = os.path.join(data_dir, fn)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
            return df
    # fallback download 2y
    df = yf.download(symbol, period='2y', interval='1d', auto_adjust=True, progress=False)
    df = df.rename_axis('Date')
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', required=True)
    parser.add_argument('--data-dir', default='.')
    parser.add_argument('--train', type=int, default=200)
    parser.add_argument('--test', type=int, default=20)
    parser.add_argument('--step', type=int, default=20)
    parser.add_argument('--out', type=str, default='walk_forward_results.csv')
    args = parser.parse_args()

    price_data: Dict[str, pd.DataFrame] = {}
    for sym in args.symbols:
        try:
            price_data[sym] = load_symbol(sym, args.data_dir)
        except Exception as e:  # noqa
            print(f"Failed to load {sym}: {e}", file=sys.stderr)

    if not price_data:
        print("No symbols loaded. Exiting.")
        return 1

    df_results = evaluate_universe_walk_forward(price_data, train_size=args.train, test_size=args.test, step=args.step)
    if df_results.empty:
        print("No results produced.")
        return 1

    df_results.to_csv(args.out, index=False)
    print(f"Saved {len(df_results)} window rows to {args.out}")
    print(df_results.groupby('symbol')['directional_accuracy'].mean().sort_values(ascending=False).head())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
