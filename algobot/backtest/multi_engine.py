"""Vectorized multi-symbol backtester (daily bar, end-of-day decisions).

Design Goals:
- Accept dict of price DataFrames (symbol->DataFrame with at least Close, Volume) or single wide DataFrame.
- Generate features per symbol (reuse build_basic_features) and simple model (Ridge) predictions.
- Convert predictions to position weights with risk constraints (max per symbol, gross exposure).
- Apply transaction costs (bps per side) & optional slippage (bps) on fills.
- Track portfolio equity curve using Portfolio abstraction.

Simplifications:
- Daily rebalancing at close using predictions from data up to current day (no lookahead: prediction for next day uses features up to current day; trade executed at close for next day open proxy).
- Long-only weights: negative predictions => 0 weight.

Future Enhancements (not implemented yet):
- Shorting, borrow costs.
- Intraday fills (use open next day).
- More sophisticated weighting (mean-variance, risk parity) and risk model integration.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import warnings

from algobot.features.basic import build_basic_features, FEATURE_COLUMNS
from algobot.portfolio.portfolio import Portfolio


@dataclass
class MultiBacktestResult:
    equity_curve: pd.Series
    daily_records: pd.DataFrame
    trades: List[dict]
    metrics: Dict[str, float]


def _prepare_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if 'Volume' not in df.columns:
        df['Volume'] = 1_000_000
    return build_basic_features(df[['Close', 'Volume']])


def _align_universe(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DatetimeIndex, Dict[str, pd.DataFrame]]:
    # union of all indices, forward fill missing
    all_index = sorted(set().union(*[d.index for d in data.values()]))
    idx = pd.DatetimeIndex(all_index)
    out = {}
    for sym, df in data.items():
        df2 = df.reindex(idx).ffill()
        out[sym] = df2
    return idx, out


def run_multi_backtest(price_data: Dict[str, pd.DataFrame], initial_capital: float = 100_000.0,
                       max_symbol_weight: float = 0.15,  # 15% per symbol cap
                       target_gross_exposure: float = 1.0,  # fully invested target
                       cost_bps: float = 5.0,  # round-trip approximated per trade side cost in bps
                       slippage_bps: float = 2.0,
                       min_feature_history: int = 40) -> MultiBacktestResult:
    # Align indices & prep features
    index, aligned = _align_universe(price_data)
    feature_store: Dict[str, pd.DataFrame] = {}
    for sym, df in aligned.items():
        feature_store[sym] = _prepare_symbol(df)

    portfolio = Portfolio(initial_cash=initial_capital)
    trades: List[dict] = []
    daily_rows = []

    cost_rate = cost_bps / 10_000.0
    slip_rate = slippage_bps / 10_000.0

    model_cache: Dict[str, Ridge] = {}

    for current_date in index:
        # Build per-symbol predictions using data up to (and including) current_date -1 (avoid lookahead)
        pred_info = []
        for sym, feats in feature_store.items():
            if current_date not in feats.index:
                continue
            # Use features up to previous row for model training/prediction of current date+1
            loc = feats.index.get_loc(current_date)
            if isinstance(loc, slice) or isinstance(loc, np.ndarray):
                continue
            if loc < min_feature_history:
                continue
            train = feats.iloc[:loc]  # exclude current row
            if len(train) < min_feature_history:
                continue
            X = train[FEATURE_COLUMNS]
            y = train['target']
            if sym not in model_cache:
                model_cache[sym] = Ridge(alpha=1.0)
            model = model_cache[sym]
            model.fit(X, y)
            current_feats = feats.iloc[[loc]][FEATURE_COLUMNS]  # keep columns to avoid warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress feature names warning
                pred = float(model.predict(current_feats)[0])  # expected next-day return
            pred_info.append((sym, pred))

        if not pred_info:
            # Mark portfolio
            prices_today = {sym: aligned[sym].loc[current_date, 'Close'] for sym in aligned if current_date in aligned[sym].index}
            snapshot = portfolio.mark_to_market(prices_today)
            daily_rows.append({'date': current_date, **{k: snapshot[k] for k in ('equity','cash','positions_value')}})
            continue

        # Convert predictions to provisional weights (linear positive scaling)
        preds = pd.DataFrame(pred_info, columns=['symbol', 'pred'])
        preds['positive'] = preds['pred'].clip(lower=0)
        if preds['positive'].sum() == 0:
            target_weights = {sym: 0.0 for sym in preds['symbol']}
        else:
            preds['raw_w'] = preds['positive'] / preds['positive'].sum()
            # Cap per symbol
            preds['capped_w'] = preds['raw_w'].clip(upper=max_symbol_weight)
            total_capped = preds['capped_w'].sum()
            if total_capped > 0:
                preds['final_w'] = preds['capped_w'] / total_capped * target_gross_exposure
            else:
                preds['final_w'] = 0.0
            target_weights = dict(zip(preds['symbol'], preds['final_w']))

        # Compute desired shares vs current, execute trades at closing price with slippage/costs
        prices_today = {sym: aligned[sym].loc[current_date, 'Close'] for sym in aligned if current_date in aligned[sym].index}
        equity_before = portfolio.mark_to_market(prices_today)['equity']
        for sym, target_w in target_weights.items():
            price = prices_today.get(sym)
            if price is None or price <= 0:
                continue
            target_value = target_w * equity_before
            existing = portfolio.positions.get(sym)
            existing_value = existing.market_value(price) if existing else 0.0
            delta_value = target_value - existing_value
            if abs(delta_value) / equity_before < 0.005:  # ignore tiny rebalances <0.5% equity
                continue
            if delta_value > 0:  # buy
                shares = delta_value / price
                # Apply slippage upward
                exec_price = price * (1 + slip_rate)
                cost = shares * exec_price
                if cost > portfolio.cash:
                    shares = portfolio.cash / exec_price
                    cost = shares * exec_price
                if shares <= 0:
                    continue
                if existing:
                    # Simplified: close and reopen (placeholder for partial adjust) -> improvement later
                    portfolio.close_position(sym, exec_price)
                portfolio.open_position(sym, shares, exec_price)
                fee = cost * cost_rate
                portfolio.cash -= fee
                trades.append({'date': current_date, 'symbol': sym, 'action': 'BUY', 'shares': shares, 'price': exec_price, 'fee': fee})
            else:  # sell or reduce
                shares_to_sell_value = -delta_value
                if not existing:
                    continue
                shares_to_sell = min(existing.shares, shares_to_sell_value / price)
                if shares_to_sell <= 0:
                    continue
                exec_price = price * (1 - slip_rate)
                # For simplicity: if partial reduce, emulate by closing and reopening remaining
                remaining_shares = existing.shares - shares_to_sell
                portfolio.close_position(sym, exec_price)
                fee = shares_to_sell * exec_price * cost_rate
                portfolio.cash -= fee
                trades.append({'date': current_date, 'symbol': sym, 'action': 'SELL', 'shares': shares_to_sell, 'price': exec_price, 'fee': fee})
                if remaining_shares > 0:
                    portfolio.open_position(sym, remaining_shares, exec_price)

        snapshot = portfolio.mark_to_market(prices_today)
        daily_rows.append({'date': current_date, **{k: snapshot[k] for k in ('equity','cash','positions_value')}})

    equity_curve = pd.Series({row['date']: row['equity'] for row in daily_rows})
    returns = equity_curve.pct_change().fillna(0)
    # Metrics
    roll_max = equity_curve.cummax()
    dd = (equity_curve / roll_max) - 1
    metrics = {
        'final_equity': float(equity_curve.iloc[-1]) if len(equity_curve) else initial_capital,
        'total_return_pct': float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 1 else 0.0,
        'max_drawdown_pct': float(dd.min()) if len(dd) else 0.0,
        'sharpe': float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() else 0.0)
    }

    return MultiBacktestResult(
        equity_curve=equity_curve,
        daily_records=pd.DataFrame(daily_rows),
        trades=trades,
        metrics=metrics
    )


__all__ = ["run_multi_backtest", "MultiBacktestResult"]
