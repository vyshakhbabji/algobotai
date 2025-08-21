"""Minimal daily backtest engine (long-only full allocation prototype)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Dict]
    metrics: Dict


def simple_long_only_backtest(price_series: pd.Series, signals: pd.Series, initial_capital: float = 100000.0) -> BacktestResult:
    position = 0
    cash = initial_capital
    shares = 0
    trades: List[Dict] = []
    equity_curve = []
    for date, price in price_series.items():
        signal = signals.loc[date]
        if signal == 'BUY' and position == 0:
            shares = cash / price
            cash = 0
            position = 1
            trades.append({'date': date, 'action': 'BUY', 'price': float(price), 'shares': shares})
        elif signal == 'SELL' and position == 1:
            cash = shares * price
            trades.append({'date': date, 'action': 'SELL', 'price': float(price), 'shares': shares})
            shares = 0
            position = 0
        equity = cash + shares * price
        equity_curve.append((date, float(equity)))
    curve = pd.Series({d: v for d, v in equity_curve})
    returns = curve.pct_change().fillna(0)
    metrics = {
        'final_equity': float(curve.iloc[-1]),
        'total_return_pct': (curve.iloc[-1] / curve.iloc[0]) - 1,
        'max_drawdown_pct': _max_drawdown(curve),
        'sharpe': _sharpe(returns)
    }
    return BacktestResult(curve, trades, metrics)


def _max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    dd = (series / roll_max) - 1
    return float(dd.min())


def _sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    std = returns.std()
    if std == 0:
        return 0.0
    return float((returns.mean() - risk_free/252) / std * np.sqrt(252))
