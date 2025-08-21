"""Daily orchestration pipeline.

Steps:
1. Load / fetch data (placeholder expects existing clean_data_SYMBOL.csv).
2. Run model scan -> record snapshot metrics.
3. Run walk-forward evaluation append -> record metrics.
4. Run multi-symbol backtest (universe) -> record metrics.
5. Persist metrics via metrics.store.
"""
from __future__ import annotations
import glob
import pandas as pd
from pathlib import Path
from algobot.analysis.model_scan import scan_universe
from algobot.forward_test.walk_forward_eval import evaluate_universe_walk_forward
from algobot.backtest.multi_engine import run_multi_backtest
from algobot.metrics.store import record_backtest, record_walk_forward
from algobot.config import GLOBAL_CONFIG


def _load_price(symbol: str) -> pd.DataFrame:
    for pattern in [f"clean_data_{symbol}.csv", f"clean_data_{symbol.upper()}.csv", f"clean_data_{symbol.lower()}.csv"]:
        p = Path(pattern)
        if p.exists():
            return pd.read_csv(p, parse_dates=['Date'], index_col='Date')
    raise FileNotFoundError(symbol)


def run_pipeline():
    universe = GLOBAL_CONFIG.universe.core_universe
    # 1 model scan
    scan_df = scan_universe(universe)
    # 2 walk-forward evaluation (light parameters for daily update)
    price_data = {}
    for sym in universe:
        try:
            price_data[sym] = _load_price(sym)
        except Exception:
            pass
    wf_df = evaluate_universe_walk_forward(price_data, train_size=150, test_size=20, step=20)
    if not wf_df.empty:
        record_walk_forward(wf_df)
    # 3 backtest
    # Use only symbols with data
    subset = {k: v for k, v in price_data.items() if not v.empty}
    if subset:
        bt = run_multi_backtest(subset)
        record_backtest("daily_multi", bt.metrics)
    return {
        'scan_rows': len(scan_df),
        'wf_rows': 0 if wf_df is None else len(wf_df),
        'backtest_final_equity': bt.metrics['final_equity'] if subset else None
    }

if __name__ == "__main__":
    out = run_pipeline()
    print(out)
