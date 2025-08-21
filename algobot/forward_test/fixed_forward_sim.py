"""Fixed training window (1y) + forward test (3+ months) simulation.

Trains a simple Ridge regression model per symbol on a specified training range
(e.g., 2024-01-01 -> 2024-12-31) using basic features, then simulates
forward period trading decisions (e.g., 2025-05-01 -> 2025-08-08) with no
retraining and no future data leakage.

Trading Rule:
- At each forward day t, build features using all data strictly prior to t.
- Predict next-day return. If predicted_return > buy_threshold -> desired state = LONG.
- If predicted_return < sell_threshold (<= 0 by default) -> desired state = FLAT.
- Positions entered/exited at the close of day t (next day performance realized via price change).

Allocation:
- Equal-weight the set of symbols with LONG signal that day, rebalancing daily.
- Initial capital distributed; if no symbols active, sit in cash.

Outputs:
- Equity curve DataFrame
- Trades list
- Per-symbol trade charts (price with BUY/SELL markers) optional.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from sklearn.linear_model import Ridge
from algobot.models.ensemble import train_ensemble, TrainedEnsemble
import matplotlib.pyplot as plt

from algobot.features.basic import build_basic_features, FEATURE_COLUMNS


@dataclass
class ForwardResult:
    equity_curve: pd.DataFrame
    trades: List[Dict[str, Any]]
    per_symbol_positions: Dict[str, pd.DataFrame]


def _download(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval='1d', auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    df.index.name = 'Date'
    df['Volume'] = df.get('Volume', 1_000_000)
    return df


def _prepare_training(df: pd.DataFrame) -> pd.DataFrame:
    feat = build_basic_features(df[['Close','Volume']].copy())
    return feat


def _fit_model(train_feat: pd.DataFrame) -> Ridge:
    X = train_feat[FEATURE_COLUMNS]
    y = train_feat['target']
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model


def run_fixed_forward(symbols: List[str], train_start: str, train_end: str,
                      fwd_start: str, fwd_end: str, initial_capital: float = 100_000.0,
                      buy_threshold: float | None = 0.003, sell_threshold: float | None = 0.0,
                      hold_threshold: float | None = None,
                      dynamic_threshold_quantiles: Tuple[float,float] | None = (0.6, 0.4),
                      pred_smoothing_window: int = 3,
                      min_trade_value: float = 500.0,
                      min_holding_days: int = 3,
                      chart: bool = True, out_dir: str = "forward_results") -> ForwardResult:
    Path(out_dir).mkdir(exist_ok=True)

    # 1. Download data ranges and split
    raw_train: Dict[str, pd.DataFrame] = {}
    raw_forward: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df_all = _download(sym, start=train_start, end=fwd_end)
        raw_train[sym] = df_all.loc[train_start:train_end]
        raw_forward[sym] = df_all.loc[fwd_start:fwd_end]

    # 2. Train models & derive dynamic thresholds per symbol
    models: Dict[str, TrainedEnsemble | Ridge] = {}
    train_features: Dict[str, pd.DataFrame] = {}
    symbol_thresholds: Dict[str, Dict[str, float]] = {}
    for sym, df in raw_train.items():
        feats = _prepare_training(df)
        if feats.empty:
            continue
        # Train ensemble (Ridge + RF) for richer prediction distribution
        try:
            ens = train_ensemble(feats, FEATURE_COLUMNS)
            models[sym] = ens
            preds_train = ens.train_pred.values
        except Exception:
            model = _fit_model(feats)
            models[sym] = model
            preds_train = model.predict(feats[FEATURE_COLUMNS])
        train_features[sym] = feats
        # dynamic thresholds (quantiles of predicted next-day return) if requested
        if dynamic_threshold_quantiles is not None:
            # preds_train already defined from ensemble training above
            q_buy, q_sell = dynamic_threshold_quantiles
            dyn_buy = float(pd.Series(preds_train).quantile(q_buy))
            dyn_sell = float(pd.Series(preds_train).quantile(q_sell))
            symbol_thresholds[sym] = {
                'buy': dyn_buy if buy_threshold is None else buy_threshold,
                'sell': dyn_sell if sell_threshold is None else sell_threshold,
            }
        else:
            symbol_thresholds[sym] = {
                'buy': buy_threshold if buy_threshold is not None else 0.0,
                'sell': sell_threshold if sell_threshold is not None else 0.0,
            }
    if hold_threshold is None:
        # default hold threshold halfway between buy & sell for each symbol
        for sym, th in symbol_thresholds.items():
            th['hold'] = (th['buy'] + th['sell']) / 2
    else:
        for sym, th in symbol_thresholds.items():
            th['hold'] = hold_threshold

    dates_forward = sorted(set().union(*[df.index for df in raw_forward.values()]))

    cash = initial_capital
    holdings: Dict[str, float] = {s: 0.0 for s in symbols}
    trades: List[Dict[str, Any]] = []
    equity_rows = []
    per_symbol_rows: Dict[str, List[Tuple[pd.Timestamp, float]]] = {s: [] for s in symbols}

    history_prices: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        history_prices[sym] = pd.concat([raw_train[sym], raw_forward[sym]])

    # State tracking for min holding period & smoothed predictions
    last_buy_date: Dict[str, pd.Timestamp | None] = {s: None for s in symbols}
    recent_preds: Dict[str, List[float]] = {s: [] for s in symbols}

    for current_date in dates_forward:
        # Build prediction inputs using data strictly before current_date
        desired_longs = set()
        prices_today: Dict[str, float] = {}
        for sym in symbols:
            df_hist = history_prices[sym]
            if current_date not in df_hist.index:
                continue
            # yfinance can occasionally return duplicate index rows; .loc may yield a Series
            close_val = df_hist.loc[current_date, 'Close']
            if isinstance(close_val, (pd.Series, pd.DataFrame)):
                # take first occurrence deterministically
                try:
                    close_val = float(np.asarray(close_val).flatten()[0])
                except Exception:
                    close_val = float(close_val.iloc[0])  # fallback
            else:
                close_val = float(close_val)
            prices_today[sym] = close_val
            past_df = df_hist.loc[:current_date].iloc[:-1]  # exclude current day
            if len(past_df) < 25:
                continue
            past_feat = build_basic_features(past_df[['Close','Volume']].copy())
            if past_feat.empty or sym not in models:
                continue
            latest_row = past_feat.iloc[[-1]][FEATURE_COLUMNS]
            model_obj = models[sym]
            if isinstance(model_obj, TrainedEnsemble):
                raw_pred = float(model_obj.predict(latest_row)[0])
            else:
                raw_pred = float(model_obj.predict(latest_row)[0])
            # smoothing
            rp = recent_preds[sym]
            rp.append(raw_pred)
            if len(rp) > pred_smoothing_window:
                rp.pop(0)
            pred = float(np.mean(rp))
            th = symbol_thresholds.get(sym, {'buy': 0.0, 'sell': 0.0, 'hold': 0.0})
            current_holding = holdings[sym] > 0
            # Decision logic with hold zone & min holding days
            if pred >= th['buy']:
                desired_longs.add(sym)
            elif pred <= th['sell']:
                # allow exit only if min holding satisfied
                if current_holding:
                    lb = last_buy_date[sym]
                    if lb is None or (current_date - lb).days >= min_holding_days:
                        # explicit not adding to desired longs (will be flat)
                        pass
                    else:
                        # still within min holding period -> keep
                        desired_longs.add(sym)
            else:  # hold zone
                if current_holding:
                    desired_longs.add(sym)
        # Determine target equal weights
        target_syms = desired_longs
        target_weight = 1/len(target_syms) if target_syms else 0.0
        equity_before = cash + sum(holdings[s]*prices_today.get(s,0) for s in symbols)
        # Rebalance daily
        transaction_cost_bps = 5  # cost per side (0.05%) realistic friction
        for sym in symbols:
            price = prices_today.get(sym)
            if price is None:
                continue
            target_value = equity_before * target_weight if sym in target_syms else 0.0
            current_value = holdings[sym]*price
            delta = target_value - current_value
            if abs(delta) < min_trade_value:
                continue
            if abs(delta) / max(equity_before,1) < 0.001:
                continue
            if delta > 0 and cash > 0:
                buy_shares = delta / price
                cost = buy_shares * price
                if cost > cash:
                    buy_shares = cash / price
                    cost = buy_shares * price
                latest_pred = recent_preds[sym][-1] if recent_preds[sym] else 0.0
                est_cost = (transaction_cost_bps/10000.0) * cost
                if latest_pred * cost < est_cost:
                    continue
                if buy_shares > 0:
                    holdings[sym] += buy_shares
                    cash -= cost
                    trades.append({'date': current_date, 'symbol': sym, 'action': 'BUY', 'shares': buy_shares, 'price': price})
                    last_buy_date[sym] = current_date
            elif delta < 0 and holdings[sym] > 0:
                sell_value = -delta
                sell_shares = min(holdings[sym], sell_value / price)
                if sell_shares > 0:
                    est_cost = (transaction_cost_bps/10000.0) * (sell_shares * price)
                    holdings[sym] -= sell_shares
                    proceeds = sell_shares * price
                    cash += (proceeds - est_cost)
                    trades.append({'date': current_date, 'symbol': sym, 'action': 'SELL', 'shares': sell_shares, 'price': price})
        equity = cash + sum(holdings[s]*prices_today.get(s,0) for s in symbols)
        equity_rows.append({'date': current_date, 'equity': equity, 'cash': cash})
        for sym in symbols:
            per_symbol_rows[sym].append((current_date, holdings[sym]))

    equity_df = pd.DataFrame(equity_rows).set_index('date')
    # Export trades & equity
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(Path(out_dir)/'trades.csv', index=False)
    equity_df.to_csv(Path(out_dir)/'equity_curve.csv')

    # Summary metrics
    if not equity_df.empty:
        final_equity = float(equity_df['equity'].iloc[-1])
        ret_pct = final_equity / initial_capital - 1.0
        rolling_max = equity_df['equity'].cummax()
        drawdown = equity_df['equity'] / rolling_max - 1.0
        max_drawdown = float(drawdown.min())
        summary = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'return_pct': ret_pct,
            'max_drawdown': max_drawdown,
            'start_date': str(equity_df.index.min().date()),
            'end_date': str(equity_df.index.max().date()),
            'num_trades': int(len(trades_df)),
            'symbols': symbols,
        }
        import json
        with open(Path(out_dir)/'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    # Charts
    if chart:
        for sym in symbols:
            price_df = history_prices[sym].loc[fwd_start:fwd_end]
            fig, ax1 = plt.subplots(figsize=(10,4))
            ax1.plot(price_df.index, price_df['Close'], label='Close', color='blue', linewidth=1.25)
            # trades for this symbol within forward window
            t_sym = trades_df[trades_df.symbol==sym].copy()
            if not t_sym.empty:
                t_sym['date'] = pd.to_datetime(t_sym['date'])
                t_sym = t_sym.set_index('date').loc[fwd_start:fwd_end].reset_index()
                buys = t_sym[t_sym.action=='BUY']
                sells = t_sym[t_sym.action=='SELL']
                if not buys.empty:
                    ax1.scatter(buys.date, buys.price, marker='^', color='green', s=60, edgecolors='k', linewidths=0.5, zorder=5, label='BUY')
                if not sells.empty:
                    ax1.scatter(sells.date, sells.price, marker='v', color='red', s=60, edgecolors='k', linewidths=0.5, zorder=5, label='SELL')
            ax1.set_title(f"{sym} Forward Trades ({fwd_start} to {fwd_end})")
            ax1.grid(alpha=0.3)
            ax1.legend(loc='best', fontsize=8)
            fig.tight_layout()
            fig.savefig(Path(out_dir)/f"{sym}_trades.png")
            plt.close(fig)

    per_symbol_positions = {s: pd.DataFrame(r, columns=['date','shares']).set_index('date') for s,r in per_symbol_rows.items()}
    trades_df.to_json(Path(out_dir)/'trades.json', orient='records', indent=2, date_format='iso')

    return ForwardResult(equity_curve=equity_df, trades=trades, per_symbol_positions=per_symbol_positions)


if __name__ == "__main__":
    # Example usage
    result = run_fixed_forward([
        'NVDA','MSFT','AAPL','META','AMZN'
    ], train_start='2024-01-01', train_end='2024-12-31', fwd_start='2025-05-01', fwd_end='2025-08-08')
    print(result.equity_curve.tail())
