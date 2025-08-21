"""Walk-forward evaluation producing rolling train/test metrics for each symbol.

Goals:
- Given historical price/feature data, create sequential train/test windows.
- Train model (simple ensemble adapter) on each train window; evaluate on immediate next test window.
- Collect metrics: directional accuracy, MAE, MAPE (where valid), R^2, average predicted return, realized return.
- Support evaluating a universe and aggregating stats.

Assumptions:
- DataFrame includes 'Close' price column and features produced by features.basic.build_basic_features.
- Prediction target: next-day log return (or simple return) derived internally.

This is a lightweight placeholder; can be extended with more complex models or hyperparameter tuning.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from algobot.features.basic import build_basic_features, FEATURE_COLUMNS

@dataclass
class WindowResult:
    symbol: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    directional_accuracy: float
    mae: float
    mape: float | None
    r2: float | None
    avg_pred_return: float
    avg_realized_return: float


def rolling_windows_index(df: pd.DataFrame, train_size: int, test_size: int, step: int) -> Iterable[Tuple[pd.Index, pd.Index]]:
    """Generate rolling (expanding-style slide) windows of positional indices.

    Parameters
    ----------
    df : DataFrame
        Indexed by datetime.
    train_size : int
        Number of rows in each train window.
    test_size : int
        Number of rows in each test window (forward segment).
    step : int
        Step size to advance the window start.
    """
    n = len(df)
    start = 0
    while True:
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > n:
            break
        train_idx = df.index[start:train_end]
        test_idx = df.index[train_end:test_end]
        if len(train_idx) == train_size and len(test_idx) == test_size:
            yield train_idx, test_idx
        start += step


def _prep_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    # If Volume missing (e.g., synthetic data in tests) create a neutral synthetic series
    if 'Volume' not in df.columns:
        df['Volume'] = 1_000_000  # constant volume placeholder
    # Basic features (idempotent if existing columns present)
    df = build_basic_features(df)
    # Target: next day simple return
    df['target'] = df['Close'].pct_change().shift(-1)
    df = df.dropna(subset=['target'])
    return df


def evaluate_symbol_walk_forward(symbol: str, data: pd.DataFrame, train_size: int = 200, test_size: int = 20, step: int = 20) -> List[WindowResult]:
    df = _prep_dataframe(data)
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    results: List[WindowResult] = []

    for train_idx, test_idx in rolling_windows_index(df, train_size, test_size, step):
        train = df.loc[train_idx]
        test = df.loc[test_idx]
        X_train = train[feature_cols].values
        y_train = train['target'].values
        X_test = test[feature_cols].values
        y_test = test['target'].values

        if np.allclose(np.std(y_train), 0):  # insufficient variance
            continue

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        signs_correct = np.sign(preds) == np.sign(y_test)
        directional_accuracy = float(np.mean(signs_correct)) if len(signs_correct) else math.nan
        mae = float(mean_absolute_error(y_test, preds))
        mape = float(np.mean(np.abs((y_test - preds) / (y_test + 1e-12)))) if np.all(y_test != 0) else None
        try:
            r2 = float(r2_score(y_test, preds))
        except ValueError:
            r2 = None
        avg_pred = float(np.mean(preds))
        avg_realized = float(np.mean(y_test))

        results.append(WindowResult(
            symbol=symbol,
            train_start=str(train.index[0].date()),
            train_end=str(train.index[-1].date()),
            test_start=str(test.index[0].date()),
            test_end=str(test.index[-1].date()),
            n_train=len(train),
            n_test=len(test),
            directional_accuracy=directional_accuracy,
            mae=mae,
            mape=mape,
            r2=r2,
            avg_pred_return=avg_pred,
            avg_realized_return=avg_realized,
        ))
    return results


def evaluate_universe_walk_forward(price_data: Dict[str, pd.DataFrame], train_size: int = 200, test_size: int = 20, step: int = 20) -> pd.DataFrame:
    """Evaluate each symbol and aggregate window results into a single DataFrame."""
    all_rows: List[WindowResult] = []
    for symbol, df in price_data.items():
        try:
            all_rows.extend(evaluate_symbol_walk_forward(symbol, df, train_size, test_size, step))
        except Exception as e:  # noqa
            # Skip problematic symbol for now
            continue
    if not all_rows:
        return pd.DataFrame()
    records = [r.__dict__ for r in all_rows]
    return pd.DataFrame(records)


if __name__ == "__main__":  # basic smoke test using random data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=400, freq='B')
    rng = np.random.default_rng(42)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=len(dates))))
    df_sample = pd.DataFrame({'Close': price}, index=dates)
    out = evaluate_symbol_walk_forward('DEMO', df_sample)
    print(f"Generated {len(out)} window results")
