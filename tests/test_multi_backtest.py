import pandas as pd
import numpy as np
from algobot.backtest.multi_engine import run_multi_backtest


def make_price_df(seed: int, periods: int = 260):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2024-01-01', periods=periods, freq='B')
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=len(dates))))
    vol = rng.integers(1_000_000, 2_000_000, size=len(dates))
    return pd.DataFrame({'Close': price, 'Volume': vol}, index=dates)


def test_run_multi_backtest_smoke():
    data = {sym: make_price_df(i) for i, sym in enumerate(['AAA','BBB','CCC'], start=1)}
    result = run_multi_backtest(data, initial_capital=50_000, max_symbol_weight=0.5)
    assert result.equity_curve.is_monotonic_increasing is False  # equity should move
    assert 'final_equity' in result.metrics
    assert len(result.daily_records) > 10
    # ensure trades recorded
    assert len(result.trades) > 0
