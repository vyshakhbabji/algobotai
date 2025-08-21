"""Advanced forward simulation with rolling retrain, classification, conviction sizing.

Features:
- Monthly rolling retrain (train window 252 trading days by default)
- Advanced feature set + standardization
- Gradient Boosting + Logistic Regression stacking for up probability
- Conviction-based position sizing (cap weights)
- Volatility targeting to keep annualized vol near target
- Transaction costs & min holding days
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from algobot.features.advanced import build_advanced_features, ADV_FEATURE_COLUMNS

@dataclass
class AdvForwardResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    per_symbol_positions: Dict[str, pd.DataFrame]
    metrics: Dict[str, Any]


def _dl(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval='1d', auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data {symbol}")
    df.index.name = 'Date'
    return df


def _train_classifier(df_feat: pd.DataFrame):
    X = df_feat[ADV_FEATURE_COLUMNS]
    y = df_feat['target_up']
    # Simple ensemble: GradientBoosting + Logistic
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(max_iter=500)
    gb.fit(X, y)
    lr.fit(X, y)
    # Blend probabilities equally; store individual for diagnostics
    p_gb = gb.predict_proba(X)[:,1]
    p_lr = lr.predict_proba(X)[:,1]
    p_blend = 0.5*p_gb + 0.5*p_lr
    try:
        auc = roc_auc_score(y, p_blend)
    except Exception:
        auc = float('nan')
    return {'gb': gb, 'lr': lr, 'train_auc': auc}


def _predict(models: Dict[str, Any], X: pd.DataFrame) -> float:
    p = 0.5*models['gb'].predict_proba(X)[:,1] + 0.5*models['lr'].predict_proba(X)[:,1]
    return float(p[-1])


def advanced_forward(symbols: List[str], start: str, end: str,
                     lookback_days: int = 252, retrain_interval: str = 'M',
                     prob_buy: float = 0.60, prob_exit: float = 0.50, prob_hard_exit: float = 0.42,
                     smoothing_window: int = 3,
                     vol_target_annual: float = 0.18, min_holding_days: int = 5,
                     max_symbol_weight: float = 0.25, transaction_cost_bps: float = 5,
                     rebalance_weekdays: tuple = (0,),  # only Mondays by default (0=Mon)
                     allow_midweek_hard_exits: bool = True,
                     use_regime_filter: bool = True,
                     regime_symbol: str = 'SPY', regime_fast: int = 20, regime_slow: int = 100,
                     out_dir: str = 'adv_forward_results', chart: bool = True,
                     gross_target: float = 1.0, allow_leverage: bool = False) -> AdvForwardResult:
    Path(out_dir).mkdir(exist_ok=True)
    raw: Dict[str, pd.DataFrame] = {s: _dl(s, start, end) for s in symbols}
    # Market regime data (simple trend filter): only allow new entries when fast SMA above slow SMA and slope positive
    regime_df = None
    if use_regime_filter:
        try:
            regime_df = _dl(regime_symbol, start, end)
            regime_df['fast'] = regime_df['Close'].rolling(regime_fast).mean()
            regime_df['slow'] = regime_df['Close'].rolling(regime_slow).mean()
            regime_df['slope'] = regime_df['fast'].diff(5)
        except Exception:
            regime_df = None
    all_dates = sorted(set().union(*[df.index for df in raw.values()]))
    # Pre-build feature stores
    feat_cache: Dict[str, pd.DataFrame] = {s: build_advanced_features(df) for s, df in raw.items()}

    models: Dict[str, Dict[str, Any]] = {s: {} for s in symbols}
    last_train_date: Dict[str, pd.Timestamp | None] = {s: None for s in symbols}

    holdings = {s: 0.0 for s in symbols}
    cash = 100000.0
    trades = []
    equity_rows = []
    last_buy_date = {s: None for s in symbols}
    # Track daily holdings & probability history for charting
    holdings_history: Dict[str, list] = {s: [] for s in symbols}
    prob_history: Dict[str, list] = {s: [] for s in symbols}
    # Probability smoothing, peak probability and position bookkeeping
    prob_windows: Dict[str, list] = {s: [] for s in symbols}
    peak_prob: Dict[str, float] = {s: 0.0 for s in symbols}
    entry_prob: Dict[str, float] = {s: float('nan') for s in symbols}
    open_pos: Dict[str, Dict[str, Any]] = {s: {} for s in symbols}
    round_trips: List[Dict[str, Any]] = []
    trade_notional_values: List[float] = []  # for turnover

    def _annualized_vol(returns: pd.Series) -> float:
        try:
            vals = pd.to_numeric(returns, errors='coerce').dropna()
        except Exception:
            return 0.0
        if vals.empty:
            return 0.0
        return float(vals.std()) * np.sqrt(252)

    price_history = {s: raw[s]['Close'] for s in symbols}
    portfolio_daily_returns = []

    for current_date in all_dates:
        # Skip until have lookback window
        window_start = current_date - pd.Timedelta(days=lookback_days*1.5)
        # Gather predictions
        probs: Dict[str, float] = {}
        for s in symbols:
            fdf = feat_cache[s]
            if current_date not in fdf.index:
                continue
            hist_window = fdf.loc[:current_date].iloc[-lookback_days:]
            if len(hist_window) < lookback_days//2:
                continue
            # Retrain monthly (or first time)
            if (last_train_date[s] is None or
                current_date.to_period('M') != last_train_date[s].to_period('M')):
                models[s] = _train_classifier(hist_window)
                last_train_date[s] = current_date
            # Predict using most recent features excluding target columns
            raw_p = _predict(models[s], hist_window[ADV_FEATURE_COLUMNS].iloc[[-1]])
            pw = prob_windows[s]
            pw.append(raw_p)
            if len(pw) > smoothing_window:
                pw.pop(0)
            smooth_p = float(np.mean(pw))
            probs[s] = smooth_p
        prices_today = {}
        for s in symbols:
            if current_date in price_history[s].index:
                val = price_history[s].loc[current_date]
                if isinstance(val, (pd.Series, pd.DataFrame)):
                    try:
                        val = float(np.asarray(val).flatten()[0])
                    except Exception:
                        val = float(val.iloc[0])
                prices_today[s] = float(val)
        equity_before = cash + sum(holdings[s]*prices_today.get(s,0) for s in symbols)

        desired_weights: Dict[str, float] = {}
        # Regime check
        regime_ok = True
        if use_regime_filter and regime_df is not None and current_date in regime_df.index:
            rrow = regime_df.loc[current_date]
            try:
                fast_val = float(np.asarray(rrow['fast']).flatten()[0])
                slow_val = float(np.asarray(rrow['slow']).flatten()[0])
                slope_val = float(np.asarray(rrow['slope']).flatten()[0])
                if not (fast_val > slow_val and slope_val > 0):
                    regime_ok = False
            except Exception:
                pass
        is_rebalance_day = current_date.weekday() in rebalance_weekdays
        for s, p in probs.items():
            current_weight = (holdings[s]*prices_today.get(s,0))/max(equity_before,1)
            # Update peak probability while in position
            if holdings[s] > 0 and p > peak_prob[s]:
                peak_prob[s] = p
            if holdings[s] == 0:
                peak_prob[s] = max(peak_prob[s], p)
            # Entry condition
            if p >= prob_buy and is_rebalance_day and regime_ok:
                conviction = (p - prob_buy) / max(1e-6, (1 - prob_buy))
                target_w = min(max_symbol_weight, max(0.0, conviction*max_symbol_weight))
                desired_weights[s] = target_w
            elif holdings[s] > 0:
                # Hard exit condition (allow midweek if enabled)
                lb = last_buy_date[s]
                held_days = (current_date - lb).days if lb else 0
                if p <= prob_hard_exit and (allow_midweek_hard_exits or is_rebalance_day):
                    if held_days >= min_holding_days:
                        desired_weights[s] = 0.0
                    else:
                        desired_weights[s] = current_weight
                elif p <= prob_exit and is_rebalance_day:
                    # Partial de-risk: cut position in half if holding period satisfied
                    if held_days >= min_holding_days:
                        desired_weights[s] = current_weight * 0.5
                    else:
                        desired_weights[s] = current_weight
                else:  # hold
                    desired_weights[s] = current_weight
        # Normalize weights to meet gross_target (up or down). If sum==0, skip.
        total_w = sum(desired_weights.values())
        if total_w > 0:
            # Cap per-name first (already applied in target generation), then scale to gross_target
            scale = 1.0
            if total_w != gross_target:
                scale = gross_target / total_w
            for s in list(desired_weights.keys()):
                desired_weights[s] *= scale

        # Volatility targeting (scale exposures if realized port vol > target)
        if portfolio_daily_returns:
            realized_vol = _annualized_vol(pd.Series(portfolio_daily_returns[-60:]))
            if realized_vol > vol_target_annual and realized_vol > 0:
                scale = vol_target_annual / realized_vol
                for s in desired_weights:
                    desired_weights[s] *= scale

        # Execute rebalancing
        for s, w in desired_weights.items():
            price = prices_today.get(s)
            if price is None:
                continue
            if isinstance(price, (pd.Series, pd.DataFrame)):
                price = float(np.asarray(price).flatten()[0])
            price = float(price)
            target_value = float(w) * float(equity_before)
            current_value = float(holdings[s]) * price
            delta_value = float(target_value - current_value)
            if abs(delta_value) < 400:  # skip tiny
                continue
            if delta_value > 0:
                buy_value = delta_value if allow_leverage else min(delta_value, cash)
                shares = buy_value / price
                if shares > 0:
                    cost = buy_value * (transaction_cost_bps/10000.0)
                    holdings[s] += shares
                    cash -= (buy_value + cost)
                    trades.append({'date': current_date, 'symbol': s, 'action': 'BUY', 'shares': shares, 'price': price, 'prob': probs.get(s, None)})
                    last_buy_date[s] = current_date
                    trade_notional_values.append(buy_value)
                    # Position bookkeeping
                    if not open_pos[s]:
                        open_pos[s] = {'shares': shares, 'cost_basis': price, 'cost_basis_per_share': price, 'entry_date': current_date}
                        entry_prob[s] = probs.get(s, float('nan'))
                        peak_prob[s] = probs.get(s, 0.0)
                    else:
                        prev = open_pos[s]
                        tot_sh = prev['shares'] + shares
                        new_cps = (prev['shares']*prev['cost_basis_per_share'] + shares*price)/tot_sh
                        prev['shares'] = tot_sh
                        prev['cost_basis_per_share'] = new_cps
                        prev['cost_basis'] = tot_sh * new_cps
            elif delta_value < 0 and holdings[s] > 0:
                sell_value = min(-delta_value, holdings[s]*price)
                shares = sell_value / price
                if shares > 0:
                    proceeds = sell_value
                    cost = proceeds * (transaction_cost_bps/10000.0)
                    holdings[s] -= shares
                    cash += (proceeds - cost)
                    trades.append({'date': current_date, 'symbol': s, 'action': 'SELL', 'shares': shares, 'price': price, 'prob': probs.get(s, None)})
                    trade_notional_values.append(proceeds)
                    # Round trip detection
                    if open_pos[s]:
                        pos = open_pos[s]
                        if holdings[s] <= 1e-6:  # fully exited
                            pnl = (price - pos['cost_basis_per_share']) * pos['shares']
                            hold_days = (current_date - pos['entry_date']).days
                            round_trips.append({
                                'symbol': s,
                                'entry_date': pos['entry_date'],
                                'exit_date': current_date,
                                'pnl': pnl,
                                'return_pct': pnl/(pos['cost_basis_per_share']*pos['shares']+1e-9),
                                'holding_days': hold_days,
                                'entry_prob': entry_prob[s],
                                'peak_prob': peak_prob[s],
                                'exit_prob': probs.get(s, float('nan'))
                            })
                            open_pos[s] = {}
        equity = cash + sum(holdings[s]*prices_today.get(s,0) for s in symbols)
        equity_rows.append({'date': current_date, 'equity': equity, 'cash': cash})
        # Track portfolio daily return
        if len(equity_rows) > 1:
            r = equity_rows[-1]['equity']/equity_rows[-2]['equity'] - 1
            portfolio_daily_returns.append(r)
        # Record per-symbol holdings & probability for this day
        for s in symbols:
            holdings_history[s].append({'date': current_date, 'shares': holdings[s]})
            prob_val = probs.get(s, float('nan'))
            prob_history[s].append({'date': current_date, 'prob': prob_val})

    equity_df = pd.DataFrame(equity_rows).set_index('date')
    trades_df = pd.DataFrame(trades)

    # Metrics
    if not equity_df.empty:
        final_equity = float(equity_df['equity'].iloc[-1])
        ret_pct = final_equity / 100000.0 - 1
        dd = equity_df['equity']/equity_df['equity'].cummax() - 1
        max_dd = float(dd.min())
        daily_ret_series = equity_df['equity'].pct_change().dropna()
        sharpe = float(daily_ret_series.mean()/ (daily_ret_series.std()+1e-9) * np.sqrt(252)) if not daily_ret_series.empty else float('nan')
        downside = daily_ret_series[daily_ret_series<0]
        sortino = float(daily_ret_series.mean()/ (downside.std()+1e-9) * np.sqrt(252)) if not downside.empty else float('nan')
        turnover = float(sum(trade_notional_values)/( (equity_df['equity'].mean()+1e-9))) if trade_notional_values else 0.0
        rt_df = pd.DataFrame(round_trips)
        win_rate = float((rt_df['pnl']>0).mean()) if not rt_df.empty else float('nan')
        avg_hold = float(rt_df['holding_days'].mean()) if not rt_df.empty else float('nan')
        profit_factor = float(rt_df.loc[rt_df.pnl>0,'pnl'].sum()/abs(rt_df.loc[rt_df.pnl<0,'pnl'].sum())) if (not rt_df.empty and (rt_df.loc[rt_df.pnl<0,'pnl'].sum()!=0)) else float('nan')
        metrics = {
            'initial_capital': 100000.0,
            'final_equity': final_equity,
            'return_pct': ret_pct,
            'max_drawdown': max_dd,
            'num_trades': int(len(trades_df)),
            'sharpe': sharpe,
            'sortino': sortino,
            'turnover': turnover,
            'round_trips': len(round_trips),
            'win_rate': win_rate,
            'avg_holding_days': avg_hold,
            'profit_factor': profit_factor
        }
    else:
        metrics = {}

    # Charts
    if chart and not equity_df.empty:
        # Equity curve
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(equity_df.index, equity_df['equity'], label='Equity', color='navy')
        ax.set_title('Advanced Forward Equity Curve')
        ax.grid(alpha=0.3)
        ax.legend(); fig.tight_layout()
        fig.savefig(Path(out_dir)/'equity_curve.png'); plt.close(fig)
        # Per-symbol trade charts
        trades_df_local = trades_df.copy()
        if not trades_df_local.empty:
            trades_df_local['date'] = pd.to_datetime(trades_df_local['date'])
        for s in symbols:
            price_series = raw[s]['Close']
            fig, ax1 = plt.subplots(figsize=(10,4))
            ax1.plot(price_series.index, price_series.values, label='Close', color='blue', linewidth=1.1)
            # Trades
            if not trades_df_local.empty:
                t_sym = trades_df_local[trades_df_local.symbol==s].set_index('date')
                if not t_sym.empty:
                    buys = t_sym[t_sym.action=='BUY']
                    sells = t_sym[t_sym.action=='SELL']
                    if not buys.empty:
                        ax1.scatter(buys.index, buys.price, marker='^', color='green', s=55, edgecolors='k', linewidths=0.5, label='BUY', zorder=5)
                    if not sells.empty:
                        ax1.scatter(sells.index, sells.price, marker='v', color='red', s=55, edgecolors='k', linewidths=0.5, label='SELL', zorder=5)
            ax1.set_ylabel('Price')
            ax1.grid(alpha=0.25)
            # Probability overlay
            ph_df = pd.DataFrame(prob_history[s])
            if not ph_df.empty:
                ph_df = ph_df.set_index('date')
                ax2 = ax1.twinx()
                ax2.plot(ph_df.index, ph_df['prob'], color='orange', alpha=0.6, label='Prob Long')
                ax2.axhline(prob_buy, color='green', linestyle='--', linewidth=0.8)
                ax2.axhline(prob_exit, color='red', linestyle='--', linewidth=0.8)
                ax2.set_ylabel('Prob')
                # Combine legends
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines+lines2, labels+labels2, fontsize=8, loc='upper left')
            else:
                ax1.legend(fontsize=8)
            ax1.set_title(f"{s} Advanced Forward Trades")
            fig.tight_layout()
            fig.savefig(Path(out_dir)/f"{s}_trades.png")
            plt.close(fig)

    equity_df.to_csv(Path(out_dir)/'equity_curve.csv')
    trades_df.to_csv(Path(out_dir)/'trades.csv', index=False)

    per_symbol_positions = {s: pd.DataFrame(holdings_history[s]).set_index('date') for s in symbols}

    return AdvForwardResult(equity_curve=equity_df, trades=trades_df, per_symbol_positions=per_symbol_positions, metrics=metrics)

if __name__ == '__main__':
    res = advanced_forward(['NVDA','MSFT','AAPL','META','AMZN'], start='2024-01-01', end='2025-08-08')
    print(res.metrics)
