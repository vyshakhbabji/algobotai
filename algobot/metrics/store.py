"""Metrics persistence layer (lightweight).

Persists backtest, walk-forward, and scan metrics into:
- SQLite database for structured querying.
- Optional Parquet append for large time-series metrics.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Dict, Any, Iterable
import pandas as pd

_DB_FILE = Path("metrics.db")

_SCHEMA = {
    'backtests': "CREATE TABLE IF NOT EXISTS backtests (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, label TEXT, final_equity REAL, total_return REAL, max_dd REAL, sharpe REAL)",
    'walk_forward': "CREATE TABLE IF NOT EXISTS walk_forward (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT, train_start TEXT, train_end TEXT, test_start TEXT, test_end TEXT, directional_accuracy REAL, mae REAL, r2 REAL)"
}


def _ensure_schema():
    with sqlite3.connect(_DB_FILE) as conn:
        cur = conn.cursor()
        for ddl in _SCHEMA.values():
            cur.execute(ddl)
        conn.commit()


def record_backtest(label: str, metrics: Dict[str, Any]):
    _ensure_schema()
    with sqlite3.connect(_DB_FILE) as conn:
        conn.execute(
            "INSERT INTO backtests (ts,label,final_equity,total_return,max_dd,sharpe) VALUES (datetime('now'),?,?,?,?,?)",
            [label, metrics.get('final_equity'), metrics.get('total_return_pct'), metrics.get('max_drawdown_pct'), metrics.get('sharpe')]
        )
        conn.commit()


def record_walk_forward(df: pd.DataFrame):
    _ensure_schema()
    cols = ['symbol','train_start','train_end','test_start','test_end','directional_accuracy','mae','r2']
    with sqlite3.connect(_DB_FILE) as conn:
        for _, row in df[cols].iterrows():
            conn.execute(
                "INSERT INTO walk_forward (ts,symbol,train_start,train_end,test_start,test_end,directional_accuracy,mae,r2) VALUES (datetime('now'),?,?,?,?,?,?,?,?)",
                list(row.values)
            )
        conn.commit()


def load_backtests() -> pd.DataFrame:
    _ensure_schema()
    with sqlite3.connect(_DB_FILE) as conn:
        return pd.read_sql("SELECT * FROM backtests ORDER BY id DESC", conn)


def load_walk_forward(symbol: str | None = None) -> pd.DataFrame:
    _ensure_schema()
    with sqlite3.connect(_DB_FILE) as conn:
        if symbol:
            return pd.read_sql("SELECT * FROM walk_forward WHERE symbol=? ORDER BY id DESC", conn, params=[symbol])
        return pd.read_sql("SELECT * FROM walk_forward ORDER BY id DESC", conn)

__all__ = ["record_backtest","record_walk_forward","load_backtests","load_walk_forward"]
