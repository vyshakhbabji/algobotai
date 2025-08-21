"""Single-symbol NVDA focused forward test (train 2024, forward 2025-05-01..2025-08-08).

Goal: Demonstrate how the system would have traded NVDA alone with an optimized
low-churn, high-conviction long-only strategy using advanced features.

Strategy Outline:
- Train ensemble classifier (GradientBoosting + Logistic) on 2024 data (advanced features).
- For each forward date t, compute features up to t-1 and probability p_up.
- Apply smoothing (moving average) over last N probs.
- Entry: p_smooth >= prob_buy AND price > SMA50.
- Exit (after min holding): (p_smooth <= prob_exit) OR price < SMA50 OR trailing stop hit.
- Trailing stop: 10% below peak close since entry.
- Transaction costs: 5 bps per side.
- Position size: 100% notional when in.

Outputs:
- trades.csv, equity_curve.csv, nvda_trades.png, summary.json in nvda_single_results/
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from algobot.features.advanced import build_advanced_features, ADV_FEATURE_COLUMNS


def _download(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval='1d', auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    df.index.name = 'Date'
    return df


def _train_classifier(df_feat: pd.DataFrame):
    X = df_feat[ADV_FEATURE_COLUMNS]
    y = df_feat['target_up']
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(max_iter=500)
    gb.fit(X, y)
    lr.fit(X, y)
    p_gb = gb.predict_proba(X)[:,1]
    p_lr = lr.predict_proba(X)[:,1]
    p = 0.5*p_gb + 0.5*p_lr
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float('nan')
    return {'gb': gb, 'lr': lr, 'auc': auc}


def _predict(models, X: pd.DataFrame) -> float:
    return float(0.5*models['gb'].predict_proba(X)[:,1] + 0.5*models['lr'].predict_proba(X)[:,1])


def run_nvda_single(train_start='2024-01-01', train_end='2025-04-30',
                     fwd_start='2025-05-01', fwd_end='2025-07-31',
                     initial_capital: float = 100_000.0,
                     prob_buy: float = 0.58, prob_exit: float = 0.50,
                     smoothing_window: int = 3, min_holding_days: int = 5,
                     trail_pct: float = 0.10,
                     transaction_cost_bps: float = 5,
                     out_dir: str = 'nvda_single_results', chart: bool = True):
    Path(out_dir).mkdir(exist_ok=True)
    raw = _download('NVDA', train_start, fwd_end)
    # Safety check
    if pd.to_datetime(train_end) < pd.to_datetime(fwd_start):
        train_df = raw.loc[train_start:train_end]
    else:
        # If overlapping or equal, still permit but slice strictly before forward start
        train_df = raw.loc[train_start:pd.to_datetime(fwd_start) - pd.Timedelta(days=1)]
    fwd_df = raw.loc[fwd_start:fwd_end]
    feat_train = build_advanced_features(train_df)
    models = _train_classifier(feat_train)

    probs_window = []
    in_position = False
    shares = 0.0
    cash = initial_capital
    peak_price = None
    entry_date = None
    entry_prob = None
    trades = []
    equity_rows = []

    # Precompute features for full range (will use historical portion each day)
    full_feat = build_advanced_features(raw)
    close_series = raw['Close']
    sma50 = close_series.rolling(50).mean()

    fwd_dates = fwd_df.index
    for current_date in fwd_dates:
        # Build feature row using data strictly before current_date
        hist_feat = full_feat.loc[:current_date].iloc[:-1]
        if len(hist_feat) < 60:
            continue
        X_latest = hist_feat[ADV_FEATURE_COLUMNS].iloc[[-1]]
        p_raw = _predict(models, X_latest)
        probs_window.append(p_raw)
        if len(probs_window) > smoothing_window:
            probs_window.pop(0)
        p_smooth = float(np.mean(probs_window))
        price_raw = close_series.loc[current_date]
        if isinstance(price_raw, (pd.Series, pd.DataFrame, np.ndarray)):
            try:
                price = float(np.asarray(price_raw).flatten()[0])
            except Exception:
                price = float(price_raw.iloc[0])
        else:
            price = float(price_raw)
        sma_raw = sma50.loc[current_date]
        if isinstance(sma_raw, (pd.Series, pd.DataFrame, np.ndarray)):
            try:
                sma50_val = float(np.asarray(sma_raw).flatten()[0])
            except Exception:
                sma50_val = float(sma_raw.iloc[0]) if hasattr(sma_raw, 'iloc') else float('nan')
        else:
            sma50_val = float(sma_raw) if pd.notna(sma_raw) else np.nan

        # Update trailing peak
        if in_position:
            if peak_price is None or price > peak_price:
                peak_price = price
        action = None

        # Entry logic
        if not in_position and p_smooth >= prob_buy and not np.isnan(sma50_val) and price > sma50_val:
            # Enter full notional
            buy_value = cash
            if buy_value > 0:
                shares = buy_value / price
                cost = buy_value * (transaction_cost_bps/10000.0)
                cash -= (buy_value + cost)
                in_position = True
                peak_price = price
                entry_date = current_date
                entry_prob = p_smooth
                trades.append({'date': current_date, 'action':'BUY', 'price': price, 'shares': shares, 'prob': p_smooth})
                action = 'BUY'
        # Exit logic
        elif in_position:
            held_days = (current_date - entry_date).days if entry_date else 0
            trail_stop_hit = peak_price is not None and price <= peak_price * (1 - trail_pct)
            exit_condition = (held_days >= min_holding_days and p_smooth <= prob_exit) or (not np.isnan(sma50_val) and price < sma50_val) or trail_stop_hit
            if exit_condition:
                sell_value = shares * price
                proceeds = sell_value
                cost = proceeds * (transaction_cost_bps/10000.0)
                cash += (proceeds - cost)
                trades.append({'date': current_date, 'action':'SELL', 'price': price, 'shares': shares, 'prob': p_smooth})
                in_position = False
                shares = 0.0
                peak_price = None
                action = 'SELL'
        equity = cash + shares*price
        equity_rows.append({'date': current_date, 'equity': equity, 'cash': cash, 'shares': shares, 'prob': p_smooth})

    equity_df = pd.DataFrame(equity_rows).set_index('date')
    trades_df = pd.DataFrame(trades)

    # Metrics
    if not equity_df.empty:
        final_equity = float(equity_df['equity'].iloc[-1])
        ret_pct = final_equity / initial_capital - 1
        dd = equity_df['equity']/equity_df['equity'].cummax() - 1
        max_dd = float(dd.min())
        daily_ret = equity_df['equity'].pct_change().dropna()
        sharpe = float(daily_ret.mean()/(daily_ret.std()+1e-9)*np.sqrt(252)) if not daily_ret.empty else float('nan')
    else:
        final_equity = initial_capital
        ret_pct = 0.0
        max_dd = 0.0
        sharpe = float('nan')

    # Buy & hold baseline for forward window
    if not fwd_df.empty:
        bh_start_price = float(fwd_df['Close'].iloc[0])
        bh_end_price = float(fwd_df['Close'].iloc[-1])
        bh_return = (bh_end_price/bh_start_price) - 1
    else:
        bh_return = float('nan')

    summary = {
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'strategy_return_pct': ret_pct,
        'buy_hold_return_pct': bh_return,
        'alpha_vs_buy_hold_pct': ret_pct - bh_return,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'num_trades': int(len(trades_df)),
        'auc_train': models.get('auc'),
        'train_start': train_start,
        'train_end': train_end,
        'forward_start': fwd_start,
        'forward_end': fwd_end
    }

    # Persist
    equity_df.to_csv(Path(out_dir)/'equity_curve.csv')
    trades_df.to_csv(Path(out_dir)/'trades.csv', index=False)
    import json
    with open(Path(out_dir)/'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    if chart and not equity_df.empty:
        # Price + trades + probability
        fig, ax1 = plt.subplots(figsize=(11,5))
        ax1.plot(raw.loc[fwd_start:fwd_end].index, raw.loc[fwd_start:fwd_end,'Close'], label='Close', color='blue')
        # Trades
        if not trades_df.empty:
            t = trades_df.copy(); t.date = pd.to_datetime(t.date); t = t.set_index('date')
            buys = t[t.action=='BUY']; sells = t[t.action=='SELL']
            if not buys.empty:
                ax1.scatter(buys.index, buys.price, marker='^', color='green', s=80, edgecolors='k', linewidths=0.6, label='BUY')
            if not sells.empty:
                ax1.scatter(sells.index, sells.price, marker='v', color='red', s=80, edgecolors='k', linewidths=0.6, label='SELL')
        ax1.set_ylabel('Price')
        ax2 = ax1.twinx()
        ax2.plot(equity_df.index, equity_df['prob'], color='orange', alpha=0.6, label='Prob Up')
        ax2.axhline(prob_buy, color='green', linestyle='--', linewidth=0.9)
        ax2.axhline(prob_exit, color='red', linestyle='--', linewidth=0.9)
        ax2.set_ylabel('Probability')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='upper left')
        ax1.set_title(f'NVDA Single Strategy Trades ({fwd_start} to {fwd_end})')
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(Path(out_dir)/'nvda_trades.png')
        plt.close(fig)
        # Equity curve
        fig2, ax = plt.subplots(figsize=(10,4))
        ax.plot(equity_df.index, equity_df['equity'], label='Strategy Equity', color='navy')
        # Buy & hold equity baseline
        if not fwd_df.empty:
            bh_equity = initial_capital * (fwd_df['Close']/fwd_df['Close'].iloc[0])
            ax.plot(fwd_df.index, bh_equity, label='Buy & Hold Equity', color='gray', linestyle='--')
        ax.set_title('NVDA Strategy vs Buy & Hold')
        ax.legend(); ax.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(Path(out_dir)/'equity_curve.png')
        plt.close(fig2)

    return summary


if __name__ == '__main__':
    res = run_nvda_single()
    print(res)
