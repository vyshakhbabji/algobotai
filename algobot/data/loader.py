"""Data loading utilities with corporate action handling.

Adds:
- load_equity: split-aware downloader (adjusted or manual adjustment)
- detect_corporate_action_jumps: flag potential unadjusted split/dividend gaps
- Preserves legacy helpers (load_clean_daily, download_daily)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf

DATA_DIR = Path('.')


def load_clean_daily(symbol: str) -> Optional[pd.DataFrame]:
    csv_path = DATA_DIR / f"clean_data_{symbol}.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except Exception:
            return None
    return None


def download_daily(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            return df
    except Exception:
        return None
    return None


def _get_actions(symbol: str) -> pd.DataFrame:
    try:
        acts = yf.Ticker(symbol).actions
        if acts is None or acts.empty:
            return pd.DataFrame(columns=['Dividends','Stock Splits'])
        return acts
    except Exception:
        return pd.DataFrame(columns=['Dividends','Stock Splits'])


def load_equity(symbol: str, start: str, end: str, adjusted: bool = True) -> pd.DataFrame:
    """Load daily equity data with optional manual split adjustment.

    If adjusted=True, rely on yfinance auto_adjust and annotate split_factor.
    If adjusted=False, pull raw prices and manually back-adjust historical data to be forward-comparable.
    """
    if adjusted:
        df = yf.download(symbol, start=start, end=end, interval='1d', auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        df.index.name = 'Date'
        actions = _get_actions(symbol)
        splits = actions[actions.get('Stock Splits', 0) > 0]['Stock Splits'] if not actions.empty else pd.Series(dtype=float)
        df['split_factor'] = 1.0
        if not splits.empty:
            for dt, factor in splits.items():
                dt_norm = pd.Timestamp(dt).tz_localize(None)
                if dt_norm in df.index:
                    df.loc[dt_norm, 'split_factor'] = factor
        return df
    # Manual path
    raw = yf.download(symbol, start=start, end=end, interval='1d', auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"No data for {symbol}")
    raw.index.name = 'Date'
    actions = _get_actions(symbol)
    splits = actions[actions.get('Stock Splits', 0) > 0]['Stock Splits'] if not actions.empty else pd.Series(dtype=float)
    adj = pd.Series(1.0, index=raw.index)
    if not splits.empty:
        for dt, factor in splits.items():
            dt_norm = pd.Timestamp(dt).tz_localize(None)
            adj.loc[adj.index < dt_norm] *= factor
    adjusted_prices = raw[['Open','High','Low','Close','Adj Close']].div(adj, axis=0)
    adjusted_prices['Volume'] = raw['Volume'] * adj
    adjusted_prices['split_factor'] = 1.0
    if not splits.empty:
        for dt, factor in splits.items():
            dt_norm = pd.Timestamp(dt).tz_localize(None)
            if dt_norm in adjusted_prices.index:
                adjusted_prices.loc[dt_norm, 'split_factor'] = factor
    return adjusted_prices


def detect_corporate_action_jumps(df: pd.DataFrame, price_col: str = 'Close', threshold: float = 0.5) -> pd.DataFrame:
    """Return rows where absolute log return > threshold (potential unadjusted corporate action)."""
    if price_col not in df.columns:
        raise ValueError(f"{price_col} not in DataFrame")
    log_ret = np.log(df[price_col] / df[price_col].shift(1))
    mask = log_ret.abs() > threshold
    out = df.loc[mask].copy()
    out['log_ret'] = log_ret[mask]
    return out

__all__ = [
    'load_clean_daily','download_daily','load_equity','detect_corporate_action_jumps'
]
