import pandas as pd
import numpy as np
from algobot.forward_test.walk_forward_eval import rolling_windows_index, evaluate_symbol_walk_forward

def test_rolling_windows_index_counts():
    dates = pd.date_range('2023-01-01', periods=300, freq='B')
    df = pd.DataFrame({'Close': np.linspace(100, 120, len(dates))}, index=dates)
    windows = list(rolling_windows_index(df, train_size=200, test_size=20, step=20))
    # Expected windows: starts at 0 then 20, 40, 60 until train_end+test_size <= 300
    # last start where train_end+test_size= start+200+20 <=300 => start <=80
    # starts: 0,20,40,60,80 -> 5 windows
    assert len(windows) == 5
    for train_idx, test_idx in windows:
        assert len(train_idx) == 200
        assert len(test_idx) == 20

def test_evaluate_symbol_walk_forward_smoke():
    dates = pd.date_range('2023-01-01', periods=260, freq='B')
    rng = np.random.default_rng(0)
    # simulate GBM-like series
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=len(dates))))
    df = pd.DataFrame({'Close': price}, index=dates)
    results = evaluate_symbol_walk_forward('TEST', df, train_size=200, test_size=20, step=20)
    # With 260 rows: windows where start <= 260 - 200 - 20 = 40 -> starts 0,20,40 -> 3 windows
    assert len(results) >= 1  # tolerate variance-based skips in synthetic data
    for r in results:
        assert r.symbol == 'TEST'
        assert r.n_train == 200
        assert r.n_test == 20
        assert 0 <= r.directional_accuracy <= 1
