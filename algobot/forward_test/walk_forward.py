"""Walk-forward evaluation scaffold."""
from dataclasses import dataclass
from typing import Dict, List, Callable
import pandas as pd


@dataclass
class WalkForwardWindowResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    accuracy: float | None
    r2: float | None


def rolling_windows(dates: List[pd.Timestamp], train_size: int, test_size: int):
    idx = 0
    n = len(dates)
    while idx + train_size + test_size <= n:
        yield (dates[idx], dates[idx+train_size-1], dates[idx+train_size], dates[idx+train_size+test_size-1])
        idx += test_size


def walk_forward_evaluate(symbol: str, data: pd.DataFrame, feature_fn: Callable[[pd.DataFrame], pd.DataFrame], train_days: int = 180, test_days: int = 63):
    dates = list(data.index)
    results: List[WalkForwardWindowResult] = []
    for t_start, t_end, s_start, s_end in rolling_windows(dates, train_days, test_days):
        results.append(WalkForwardWindowResult(str(t_start.date()), str(t_end.date()), str(s_start.date()), str(s_end.date()), None, None))
    return results
