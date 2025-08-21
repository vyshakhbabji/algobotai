"""Institutional-style single-symbol NVDA strategy.

Objectives:
- Rolling monthly retrain (252-day lookback) with internal validation and probability calibration (Platt scaling).
- Ensemble classifier (GradientBoosting + Logistic) -> blended prob -> calibrated prob.
- Trend + breakout gating: require (Close > SMA50 > SMA100) OR 20-day breakout for any entry.
- Multi-tier pyramiding: probability tiers expand target risked notional (0.55/0.60/0.65).
- Early breakout starter position if breakout & prob >= 0.52.
- ATR(14)-based initial and trailing stop.
- Risk-based position sizing: max 1% of equity at risk per full position; shares sized by (risk_per_trade / stop_distance).
- Transaction costs: 5 bps per side; slippage placeholder.
- Exit rules: trailing stop, hard prob drop (<0.45), soft prob exit (<0.50 after min hold), trend breakdown (Close < SMA100), or time-based (>90 days).
- Advanced metrics: CAGR, volatility, Sharpe, Sortino, hit rate, expectancy, max adverse excursion.

Forward window configured for NVDA: trade May-Jul 2025 after training history up to Apr 30 2025.

DISCLAIMER: This is an illustrative research framework; real institutional strategies require far deeper data hygiene, execution modeling, and risk aggregation.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

from algobot.features.advanced import build_advanced_features, ADV_FEATURE_COLUMNS
from algobot.data.loader import load_equity

@dataclass
class TradeRecord:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    entry_price: float
    exit_price: float | None
    shares: float
    pnl: float | None
    return_pct: float | None
    max_run_up: float
    max_drawdown: float
    entry_prob: float
    peak_prob: float
    exit_prob: float | None


def _download(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Deprecated direct downloader kept for fallback; prefer load_equity."""
    df = yf.download(symbol, start=start, end=end, interval='1d', auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f'No data for {symbol}')
    df.index.name = 'Date'
    return df


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def _train_calibrated_classifier(feat: pd.DataFrame) -> dict:
    # Split train/validation (time-based) last 20% for calibration
    n = len(feat)
    if n < 100:
        raise ValueError('Not enough data to train')
    split = int(n * 0.8)
    train = feat.iloc[:split]
    valid = feat.iloc[split:]
    X_tr, y_tr = train[ADV_FEATURE_COLUMNS], train['target_up']
    X_va, y_va = valid[ADV_FEATURE_COLUMNS], valid['target_up']
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(max_iter=500)
    gb.fit(X_tr, y_tr)
    lr.fit(X_tr, y_tr)
    p_gb_tr = gb.predict_proba(X_tr)[:,1]
    p_lr_tr = lr.predict_proba(X_tr)[:,1]
    p_blend_tr = 0.5*p_gb_tr + 0.5*p_lr_tr
    p_gb_va = gb.predict_proba(X_va)[:,1]
    p_lr_va = lr.predict_proba(X_va)[:,1]
    p_blend_va = 0.5*p_gb_va + 0.5*p_lr_va
    try:
        auc = roc_auc_score(y_va, p_blend_va)
    except Exception:
        auc = float('nan')
    # Calibrate by fitting logistic regression on validation probabilities
    calib_lr = LogisticRegression()
    calib_lr.fit(p_blend_va.reshape(-1,1), y_va)
    return {
        'gb': gb,
        'lr': lr,
        'calib_lr': calib_lr,
        'auc_valid': auc,
        'valid_log_loss': log_loss(y_va, p_blend_va, labels=[0,1]) if len(np.unique(y_va))>1 else float('nan')
    }


def _predict_prob(models: dict, X: pd.DataFrame) -> float:
    p = 0.5*models['gb'].predict_proba(X)[:,1] + 0.5*models['lr'].predict_proba(X)[:,1]
    # Calibrate
    p_cal = models['calib_lr'].predict_proba(p.reshape(-1,1))[:,1]
    return float(p_cal[-1])


def run_nvda_institutional(symbol: str = 'NVDA',
                           train_start='2024-01-01', train_end='2025-04-30',
                           fwd_start='2025-05-01', fwd_end='2025-08-10',
                           initial_capital: float = 100_000.0,
                           lookback_days: int = 252,
                           retrain_freq: str = 'M',
                           # Tier configuration: either static probs or derived from probability distribution quantiles
                           tier_probs=(0.50,0.56,0.62,0.69), breakout_prob=0.50,
                           tier_mode: str = 'quantile',  # 'static' or 'quantile'
                           tier_quantiles: tuple = (0.55,0.65,0.75,0.85),
                           tier_min_spacing: float = 0.02,
                           hard_exit_prob: float = 0.45, soft_exit_prob: float = 0.50,
                           partial_derisk_prob: float = 0.50,  # de-risk (trim) threshold after min hold
                           partial_derisk_fraction: float = 0.33,
                           min_holding_days: int = 7,
                           atr_initial_mult: float = 1.7, atr_trail_mult: float = 2.2,
                           risk_per_trade_pct: float = 0.04,
                           smoothing_window: int = 2,
                           post_split_warmup_days: int = 5,
                           momentum_20d_threshold: float = 0.08,
                           time_scalein_days: int = 5,
                           # Earlier capital recycling rung added (12%, 30%, 50%)
                           profit_ladder: tuple = (0.12, 0.30, 0.50),
                           profit_trim_fractions: tuple = (0.15, 0.20, 0.25),
                           # Fast scale unlock if early run-up achieved
                           fast_scale_gain_threshold: float = 0.08,
                           # Volatility-adaptive trailing tighten when ATR% compressed
                           enable_vol_adaptive_trail: bool = True,
                           vol_trail_floor_mult: float = 0.9,
                           atr_pct_window: int = 100,
                           low_atr_norm_threshold: float = 0.6,
                           adaptive_trail_mult_after_gain: float = 1.5,
                           adaptive_trail_gain_threshold: float = 0.25,
                           stale_days: int = 60,
                           stale_min_runup: float = 0.10,
                           enable_pullback_reentry: bool = True,
                           finalize_at_end: bool = True,
                           enable_tier_grid_search: bool = False,
                           tier_grid_delta: float = 0.01,
                           tier_grid_span: int = 1,
                           tier_grid_eval_days: int = 60,
                           regime_filter: bool = False,
                           regime_symbol: str = 'SPY',
                           regime_sma: int = 200,
                           regime_reduction: float = 0.5,
                           target_capture_min: float = 0.50,
                           target_capture_reduce: float = 0.70,
                           risk_increment: float = 0.005,
                           risk_decrement: float = 0.0,
                           risk_ceiling: float = 0.06,
                           transaction_cost_bps: float = 5,
                           # Trend suitability gating
                           enable_trend_filter: bool = True,
                           trend_min_score: float = 0.35,
                           # ATR volatility normalization
                           enable_atr_vol_normalization: bool = True,
                           # Early performance guard
                           performance_guard_days: int = 30,
                           performance_guard_min_capture: float = 0.10,
                           performance_guard_max_fraction: float = 0.2,
                           # Adaptive performance guard enhancements
                           performance_guard_min_trades: int = 4,
                           guard_lift_capture: float = 0.25,
                           guard_ramp_days: int = 20,
                           guard_strong_symbols: tuple[str,...] = ('NVDA','MSFT'),
                           # Baseline core fraction and guarded core fraction
                           early_core_fraction: float = 0.60,
                           guard_core_fraction: float = 0.30,
                           # Churn suppression (cooldown) settings
                           churn_cooldown_trades: int = 3,
                           churn_cooldown_days: int = 10,
                           # Rolling expectancy guard
                           enable_expectancy_guard: bool = True,
                           rolling_expectancy_window: int = 8,
                           expectancy_guard_max_fraction: float = 0.15,
                           # Tier smoothing to reduce monthly jumpiness
                           enable_tier_smoothing: bool = True,
                           tier_smoothing_alpha: float = 0.50,
                           # Optional rollback to earlier permissive guard behavior
                           rollback_guard: bool = False,
                           # Symbol strength classification controls
                           symbol_classification: str = 'auto',  # 'auto','strong','default','skip'
                           classification_thresholds: tuple = (0.30, -0.05),  # (strong_cut, skip_cut)
                           classification_map: dict | None = None,
                           chart: bool = True,
                           out_dir: str = 'institutional_results',
                           # --- New enhancement toggles / parameters ---
                           enable_quality_filter: bool = True,
                           quality_prob_z_min: float = -0.25,
                           quality_rsi_period: int = 14,
                           quality_rsi_min: float = 45.0,
                           quality_rsi_max: float = 75.0,
                           quality_volume_multiplier: float = 0.90,
                           momentum_63d_threshold: float = 0.0,
                           enhanced_early_core: bool = True,
                           enhanced_early_core_fraction: float = 0.85,
                           enhanced_early_core_confirm_days: int = 1,
                           readd_pullback_pct: float = 0.08,
                           enable_readd_after_trim: bool = True,
                           ratchet_runup_trigger: float = 0.40,
                           ratchet_min_locked_gain: float = 0.15,
                           ema_prob_fast: int = 3,
                           ema_prob_slow: int = 10,
                           enable_prob_ema_exit: bool = True,
                           vol_cap_percentile: float = 0.90,
                           vol_cap_fraction: float = 0.80,
                           underperf_reval_days: int = 45,
                           underperf_capture_floor: float = 0.50,
                           adjust_trim_ladder: bool = True,
                           new_profit_ladder: tuple = (0.25, 0.45, 0.70),
                           new_profit_trim_fractions: tuple = (0.10, 0.15, 0.20)):
    # Per-symbol subdirectory
    # If caller supplies a path that already contains the symbol (e.g. strong_baseline/nvda_baseline), don't double-prefix
    out_dir_lower = out_dir.lower()
    if not out_dir_lower.endswith('/') and symbol.lower() in Path(out_dir).name.lower():
        pass
    else:
        if not out_dir_lower.startswith(symbol.lower()):
            out_dir = f"{symbol.lower()}_{out_dir}"
    Path(out_dir).mkdir(exist_ok=True)
    # Apply new ladder defaults if requested
    if adjust_trim_ladder:
        profit_ladder = new_profit_ladder
        profit_trim_fractions = new_profit_trim_fractions

    # Split-aware adjusted data
    px = load_equity(symbol, train_start, fwd_end, adjusted=True)
    split_days = list(px.index[px.get('split_factor', pd.Series(dtype=float))>1])
    close = px['Close']
    atr = _compute_atr(px)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma100 = close.rolling(100).mean()
    high20 = close.rolling(20).max()
    atr_pct_series = atr / close
    atr_pct_trend = atr_pct_series - atr_pct_series.rolling(10).mean()
    # Pre-compute ATR% normalization for volatility-adaptive trailing (ratio to rolling 90th percentile)
    atr_pct_90 = atr_pct_series.rolling(atr_pct_window).quantile(0.9)
    atr_pct_norm = atr_pct_series / (atr_pct_90.replace(0, np.nan))
    mom20_series = close.pct_change(20)
    mom63_series = close.pct_change(63)
    # RSI helper
    def _rsi(series: pd.Series, period: int = 14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))
    rsi_series = _rsi(close, quality_rsi_period)
    vol20 = px['Volume'].rolling(20).mean()
    # Rolling percentile ranks helper (raw array)
    def _rolling_last_percentile(s: pd.Series, window: int, min_periods: int = 30):
        def _pct(arr):
            n = len(arr)
            if n < 2:
                return np.nan
            last = arr[-1]
            less = np.sum(arr[:-1] < last)
            equal = np.sum(arr[:-1] == last)
            return (less + 0.5*equal)/(n-1)
        return s.rolling(window, min_periods=min_periods).apply(_pct, raw=True)
    mom20_rank = _rolling_last_percentile(mom20_series, 252)
    atr_pct_rank = _rolling_last_percentile(atr_pct_series, 252)
    # Advanced features built on adjusted close/volume only
    feat_full = build_advanced_features(px[['Close','Volume']])

    # Regime data (optional)
    if regime_filter:
        try:
            spy_df = yf.download(regime_symbol, start=train_start, end=fwd_end, interval='1d', auto_adjust=True, progress=False)
            spy_close = spy_df['Close']
            spy_sma = spy_close.rolling(regime_sma).mean()
        except Exception:
            regime_filter = False
            spy_close = spy_sma = None
    else:
        spy_close = spy_sma = None
    # --- Tier grid search (coarse) over last segment of training window ---
    def _approx_score(tiers):
        # Train single model on full training span
        train_seg = feat_full.loc[train_start:train_end]
        if len(train_seg) < 100:
            return -1e9
        models_tmp = _train_calibrated_classifier(train_seg)
        tail = train_seg.iloc[-tier_grid_eval_days:]
        if len(tail) < 10:
            return -1e9
        probs = []
        for i in range(len(tail)):
            probs.append( _predict_prob(models_tmp, tail[ADV_FEATURE_COLUMNS].iloc[:i+1]) )
        # Detect close column
        close_cols = [c for c in tail.columns if c.lower().startswith('close')]
        price_col = close_cols[0] if close_cols else tail.columns[0]
        prices = tail[price_col]
        # Simulate simplified exposure based on tiers (no exits) -> exposure fraction
        exposures = []
        for p in probs:
            frac = 0.0
            for j,tp in enumerate(tiers):
                if p >= tp:
                    frac = (j+1)/len(tiers)
            exposures.append(frac)
        exposures = np.array(exposures)
        rets = prices.pct_change().fillna(0).values
        strat_ret = (exposures*rets).sum()
        bh_ret = (prices.iloc[-1]/prices.iloc[0]-1)
        capture = strat_ret/(bh_ret+1e-9)
        return capture

    if enable_tier_grid_search:
        base = tier_probs
        best = base
        best_score = -1e9
        # small grid
        for d0 in range(-tier_grid_span, tier_grid_span+1):
            for d1 in range(-tier_grid_span, tier_grid_span+1):
                for d2 in range(-tier_grid_span, tier_grid_span+1):
                    cand = tuple(round(x + tier_grid_delta*dx, 4) for x,dx in zip(base,(d0,d1,d2)))
                    # maintain ordering
                    if not (cand[0] < cand[1] < cand[2]):
                        continue
                    sc = _approx_score(cand)
                    if sc > best_score:
                        best_score = sc
                        best = cand
        tier_probs = best
    forward_dates = px.loc[fwd_start:fwd_end].index

    # ---------- Symbol Strength Classification & Parameter Overrides ----------
    def _compute_symbol_strength(close_series: pd.Series) -> float:
        train_close = close_series.loc[train_start:train_end]
        if len(train_close) < 120:  # need enough history
            return -1.0
        def _h_ret(days: int):
            if len(train_close) <= days:
                return np.nan
            return train_close.iloc[-1] / train_close.iloc[-days] - 1
        r252 = _h_ret(252); r126 = _h_ret(126); r63 = _h_ret(63)
        daily = train_close.pct_change().dropna()
        sharpe_like = (daily.mean()/(daily.std()+1e-9))*np.sqrt(252) if not daily.empty else 0.0
        dd = (train_close/train_close.cummax() - 1).min()
        # Safe scalar helpers
        def _safe(v):
            try: return float(v)
            except Exception: return 0.0
        def _is_nan_scalar(v):
            try: return bool(pd.isna(v)) or bool(np.isnan(float(v)))
            except Exception: return False
        r252s = 0.0 if _is_nan_scalar(r252) else _safe(r252)
        r126s = 0.0 if _is_nan_scalar(r126) else _safe(r126)
        r63s  = 0.0 if _is_nan_scalar(r63)  else _safe(r63)
        if isinstance(dd, pd.Series):
            try: dd = dd.iloc[0]
            except Exception:
                dd = float(dd.values[0]) if hasattr(dd, 'values') and len(dd.values)>0 else 0.0
        try: dd = float(dd)
        except Exception: dd = 0.0
        score = 0.35*r252s + 0.25*r126s + 0.10*r63s + 0.20*sharpe_like + 0.10*dd
        try: return float(score)
        except Exception: return 0.0

    if classification_map and symbol in classification_map:
        classification_used = classification_map[symbol]
    elif symbol_classification != 'auto':
        classification_used = symbol_classification
    else:
        strong_cut, skip_cut = classification_thresholds
        strength_score = _compute_symbol_strength(close)
        if strength_score >= strong_cut:
            classification_used = 'strong'
        elif strength_score < skip_cut:
            classification_used = 'skip'
        else:
            classification_used = 'default'

    # If skip => return minimal summary (no trading) for speed
    if classification_used == 'skip':
        return {
            'symbol': symbol,
            'skipped': True,
            'classification': 'skip',
            'reason': 'below_skip_threshold',
            'final_equity': initial_capital,
            'total_return_pct': 0.0
        }

    # Guard rollback (global loosen) OR strong symbol overrides
    if rollback_guard:
        performance_guard_min_capture = 0.05
        performance_guard_max_fraction = max(performance_guard_max_fraction, 0.30)
        early_core_fraction = max(early_core_fraction, 0.60)
        guard_core_fraction = max(guard_core_fraction, 0.50)
        guard_lift_capture = min(guard_lift_capture, 0.20)
        guard_ramp_days = min(guard_ramp_days, 15)
        performance_guard_min_trades = max(performance_guard_min_trades, 5)
    elif classification_used == 'strong':
        performance_guard_min_capture = min(performance_guard_min_capture, 0.05)
        early_core_fraction = max(early_core_fraction, 0.72)
        guard_core_fraction = max(guard_core_fraction, 0.50)
        performance_guard_max_fraction = max(performance_guard_max_fraction, 0.40)
        guard_lift_capture = min(guard_lift_capture, 0.18)
        guard_ramp_days = min(guard_ramp_days, 10)
        performance_guard_min_trades = max(performance_guard_min_trades, 6)
        enable_expectancy_guard = False  # allow upside growth
        # Slightly delay profit trimming first rung
        if len(profit_ladder) >= 1 and profit_ladder[0] < 0.15:
            profit_ladder = tuple([0.15] + list(profit_ladder[1:]))
        if len(profit_trim_fractions) >=1 and profit_trim_fractions[0] > 0.10:
            profit_trim_fractions = (0.10,) + profit_trim_fractions[1:]

    cash = initial_capital
    shares = 0.0
    in_position = False
    entry_price = None  # first fill anchor
    cost_basis = None   # weighted average cost basis for realized PnL
    stop_price = None
    entry_prob = None
    peak_prob = 0.0
    peak_close_since_entry = None
    consecutive_prob_days = 0  # count days prob above first tier
    ladder_flags = [False]*len(profit_ladder)
    last_exit_date = None
    realized_pnl = 0.0
    unrealized_pnl = 0.0
    dynamic_risk_perc = risk_per_trade_pct

    def record_partial_trade(date, price, shares_exited, reason, prob):
        nonlocal realized_pnl
        if shares_exited <= 0 or cost_basis is None:
            return
        pnl = (price - cost_basis) * shares_exited
        realized_pnl += pnl
        ret_pct = (price/cost_basis - 1)
        completed_trades.append(TradeRecord(
            entry_date=entry_date,
            exit_date=date,
            entry_price=cost_basis,
            exit_price=price,
            shares=shares_exited,
            pnl=pnl,
            return_pct=ret_pct,
            max_run_up=open_trade_stats['max_run_up'],
            max_drawdown=open_trade_stats['max_drawdown'],
            entry_prob=entry_prob if entry_prob is not None else prob,
            peak_prob=peak_prob,
            exit_prob=prob
        ))
    last_retrain_date = None
    entry_date = None
    prob_window = []  # raw probs for smoothing
    prob_history = []  # full history for z-score & EMA
    models = None

    trades = []
    equity_rows = []
    open_trade_stats = {'max_run_up':0.0,'max_drawdown':0.0}
    completed_trades: list[TradeRecord] = []

    current_tier_probs = list(tier_probs)  # mutable current thresholds
    prev_quantile_tier_probs = None  # for smoothing quantile tiers
    performance_guard_triggered = False
    guard_trigger_date = None
    # Churn / expectancy guard state
    consecutive_loss_trades = 0
    churn_cooldown_active_until = None
    recent_trade_returns: list[float] = []
    def _maybe_retrain(current_date):
        nonlocal models, last_retrain_date, current_tier_probs, performance_guard_triggered, prev_quantile_tier_probs
        if models is None or last_retrain_date is None or current_date.to_period('M') != last_retrain_date.to_period('M'):
            hist_end = current_date - pd.Timedelta(days=1)
            hist_start = hist_end - pd.Timedelta(days=lookback_days*1.3)
            hist = feat_full.loc[str(hist_start.date()):str(hist_end.date())]
            if len(hist) > lookback_days//2:
                models = _train_calibrated_classifier(hist)
                last_retrain_date = current_date
                if tier_mode.lower() == 'quantile':
                    try:
                        Xh = hist[ADV_FEATURE_COLUMNS]
                        p = 0.5*models['gb'].predict_proba(Xh)[:,1] + 0.5*models['lr'].predict_proba(Xh)[:,1]
                        p_cal = models['calib_lr'].predict_proba(p.reshape(-1,1))[:,1]
                        qs = [max(0.0,min(1.0,q)) for q in tier_quantiles]
                        new_probs = []
                        for q in qs:
                            new_probs.append(float(np.nanpercentile(p_cal, q*100)))
                        # ensure strictly increasing with minimum spacing
                        for i in range(1,len(new_probs)):
                            if new_probs[i] <= new_probs[i-1] + tier_min_spacing:
                                new_probs[i] = min(0.9999, new_probs[i-1] + tier_min_spacing)
                        # Optional smoothing against previous quantile set
                        if enable_tier_smoothing and prev_quantile_tier_probs is not None and len(prev_quantile_tier_probs)==len(new_probs):
                            alpha = float(np.clip(tier_smoothing_alpha,0.0,1.0))
                            new_probs = [prev_quantile_tier_probs[i]*alpha + new_probs[i]*(1-alpha) for i in range(len(new_probs))]
                            for i in range(1,len(new_probs)):
                                if new_probs[i] <= new_probs[i-1] + tier_min_spacing:
                                    new_probs[i] = min(0.9999, new_probs[i-1] + tier_min_spacing)
                        prev_quantile_tier_probs = list(new_probs)
                        current_tier_probs = new_probs
                    except Exception:
                        current_tier_probs = list(tier_probs)

    for current_date in forward_dates:
        # Retrain monthly
        _maybe_retrain(current_date)
        # Build feature context strictly prior to current_date
        hist_feat = feat_full.loc[:current_date].iloc[:-1]
        if models is not None and len(hist_feat) > 50:
            p_raw = _predict_prob(models, hist_feat[ADV_FEATURE_COLUMNS].iloc[[-1]])
        else:
            p_raw = np.nan
        # Probability smoothing
        prob_window.append(p_raw)
        if len(prob_window) > smoothing_window:
            prob_window.pop(0)
        p = float(np.nanmean(prob_window)) if prob_window else p_raw
        if not np.isnan(p):
            prob_history.append(p)
        # Probability z-score (rolling 60)
        prob_z = 0.0
        if len(prob_history) >= 10:
            ref = prob_history[-60:]
            mu = float(np.nanmean(ref))
            sd = float(np.nanstd(ref) + 1e-9)
            prob_z = (p - mu)/sd

        # Robust scalar extraction (loc may return Series if duplicate index)
        c_val = close.loc[current_date]
        c = float(c_val.iloc[0] if isinstance(c_val, pd.Series) else c_val)
        def _scalar(s, idx):
            v = s.loc[idx]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            return float(v) if pd.notna(v) else np.nan
        a = _scalar(atr, current_date)
        s20 = _scalar(sma20, current_date)
        s50 = _scalar(sma50, current_date)
        s100 = _scalar(sma100, current_date)
        h20 = _scalar(high20, current_date)

    # Relaxed trend gate: only require price > SMA50 (SMA100 no longer mandatory)
        trend_ok = (not np.isnan(s50) and c > s50)
        # Pullback re-entry condition: recent exit, price near SMA20, prob supportive
        reentry_ok = False
        if enable_pullback_reentry and not in_position and last_exit_date is not None and not np.isnan(s20):
            if c <= s20 * 1.01 and p >= tier_probs[0]:
                reentry_ok = True
        breakout = (not np.isnan(h20) and c >= h20 * 0.999)
        # Split warmup: skip fresh entries for N days after a split
        in_split_warmup = any((current_date - sd).days <= post_split_warmup_days for sd in split_days if sd <= current_date)

        # Update open position stats
        if in_position:
            if peak_close_since_entry is None or c > peak_close_since_entry:
                peak_close_since_entry = c
            run_up = (c/entry_price) - 1
            draw_down = (c/peak_close_since_entry) - 1 if peak_close_since_entry else 0
            open_trade_stats['max_run_up'] = max(open_trade_stats['max_run_up'], run_up)
            open_trade_stats['max_drawdown'] = min(open_trade_stats['max_drawdown'], draw_down)
            if p > peak_prob:
                peak_prob = p

    # Determine target tier size (fraction of full size 1.0) based on prob & pyramiding
        target_fraction = 0.0
        # Regime filter dampening
        regime_ok = True
        if regime_filter and spy_close is not None and spy_sma is not None and current_date in spy_close.index:
            sc = spy_close.loc[current_date]
            ss = spy_sma.loc[current_date]
            if isinstance(sc, pd.Series):
                sc = sc.iloc[0]
            if isinstance(ss, pd.Series):
                ss = ss.iloc[0]
            if not (pd.notna(sc) and pd.notna(ss) and sc > ss):
                regime_ok = False

        # Soft regime: always allow evaluation; regime influences scaling later
        if not in_split_warmup and (trend_ok or breakout or reentry_ok):
            if breakout and p >= breakout_prob:
                target_fraction = max(target_fraction, 0.25)
            tier_list_for_eval = current_tier_probs if tier_mode.lower()=='quantile' else tier_probs
            for i, tp in enumerate(tier_list_for_eval):
                if p >= tp:
                    target_fraction = max(target_fraction, (i+1)/len(tier_list_for_eval))
            if reentry_ok:
                target_fraction = max(target_fraction, 0.3)
        # Early core sizing (enhanced conditional momentum confirmation)
        first_tier = (current_tier_probs if tier_mode.lower()=='quantile' else tier_probs)[0]
        momentum_confirm = False
        if enhanced_early_core:
            m20_val = mom20_series.loc[current_date]
            m63_val = mom63_series.loc[current_date] if current_date in mom63_series.index else np.nan
            if isinstance(m20_val, pd.Series): m20_val = m20_val.iloc[0]
            if isinstance(m63_val, pd.Series): m63_val = m63_val.iloc[0]
            momentum_confirm = (pd.notna(m20_val) and m20_val > momentum_20d_threshold and pd.notna(m63_val) and m63_val > momentum_63d_threshold)
        if trend_ok and p >= (first_tier-0.01):
            if enhanced_early_core and momentum_confirm:
                core_frac = enhanced_early_core_fraction
            else:
                core_frac = early_core_fraction
            if performance_guard_triggered:
                core_frac = min(core_frac, guard_core_fraction)
            target_fraction = max(target_fraction, core_frac)
        # Strong symbol momentum override while guard active
        if performance_guard_triggered and classification_used == 'strong':
            mom20_val_guard = mom20_series.loc[current_date]
            if isinstance(mom20_val_guard, pd.Series):
                mom20_val_guard = mom20_val_guard.iloc[0]
            if pd.notna(mom20_val_guard) and mom20_val_guard > 0.15 and p >= first_tier:
                target_fraction = max(target_fraction, 0.5)
        # Fast scale unlock: if early run-up exceeds threshold and prob supportive, ensure at least second tier fraction
        if in_position and entry_price is not None:
            early_run = (c/entry_price) - 1
            if early_run >= fast_scale_gain_threshold and p >= (first_tier-0.01):
                denom = len(current_tier_probs if tier_mode.lower()=='quantile' else tier_probs)
                min_fast_fraction = 2/denom  # at least second tier
                target_fraction = max(target_fraction, min_fast_fraction)
        # Dynamic threshold easing in strong momentum regime
        mom20_val_dyn = mom20_series.loc[current_date]
        if isinstance(mom20_val_dyn, pd.Series):
            mom20_val_dyn = mom20_val_dyn.iloc[0]
        effective_tiers = list(tier_probs)
        if pd.notna(mom20_val_dyn) and mom20_val_dyn >= 0.25:  # strong momentum
            base_list = current_tier_probs if tier_mode.lower()=='quantile' else tier_probs
            effective_tiers = [max(0.50, t - 0.01) for t in base_list]
            for i, tp in enumerate(effective_tiers):
                if p >= tp:
                    target_fraction = max(target_fraction, (i+1)/len(effective_tiers))
        # Vol expansion add: if ATR% rising and price above SMA20 add 0.1
        atr_trend_val = atr_pct_trend.loc[current_date]
        if isinstance(atr_trend_val, pd.Series):
            atr_trend_val = atr_trend_val.iloc[0]
        if not in_split_warmup and pd.notna(atr_trend_val) and atr_trend_val > 0 and (not np.isnan(s20) and c > s20):
            target_fraction = min(1.0, target_fraction + 0.1)
    # Momentum throttle (allow half position if strong 20d momentum & near first tier prob)
        mom20_val = mom20_series.loc[current_date]
        if isinstance(mom20_val, pd.Series):
            mom20_val = mom20_val.iloc[0]
        if not in_split_warmup and pd.notna(mom20_val) and mom20_val >= momentum_20d_threshold and p >= (first_tier-0.01):
            target_fraction = max(target_fraction, 0.7)
        # Time-based scale-in: if already in position and prob sustained above first tier for N days
        if in_position:
            if p >= tier_probs[0]:
                consecutive_prob_days += 1
            else:
                consecutive_prob_days = 0
            if consecutive_prob_days >= time_scalein_days:
                target_fraction = min(1.0, target_fraction + 0.2)
        else:
            consecutive_prob_days = 0
        # Compute desired full position shares based on risk model
        desired_shares = 0.0
        # Regime reduction (if regime not OK, scale down target fraction)
        if not regime_ok:
            target_fraction *= regime_reduction  # scale down instead of blocking

        # Adaptive risk adjustment based on capture ratio so far
        if equity_rows:
            first_close = close.loc[forward_dates[0]]
            if isinstance(first_close, pd.Series):
                first_close = first_close.iloc[0]
            bh_ret_to_date = (c/first_close - 1) if first_close is not None else 0
            strat_ret_to_date = ( (cash + shares*c)/initial_capital - 1 )
            capture_ratio = strat_ret_to_date/(bh_ret_to_date+1e-9) if bh_ret_to_date>0 else 0
            if capture_ratio < target_capture_min and dynamic_risk_perc < risk_ceiling:
                dynamic_risk_perc = min(risk_ceiling, dynamic_risk_perc + risk_increment)

        # Trend suitability filter (after initial target computation)
        if enable_trend_filter:
            mom_rank_val = mom20_rank.loc[current_date]
            if isinstance(mom_rank_val, pd.Series):
                mom_rank_val = mom_rank_val.iloc[0]
            sma_spread = 0.0
            if not np.isnan(s20) and not np.isnan(s50) and s50>0:
                sma_spread = max(0.0, (s20 - s50)/c)
            trend_score = 0.5 * (mom_rank_val if pd.notna(mom_rank_val) else 0.0) + 0.5 * min(1.0, sma_spread/0.05)
            if trend_score < trend_min_score:
                # cap exposure in poor trend regime
                target_fraction = min(target_fraction, 0.2)
        # Early performance guard: after first N forward days if capture below threshold, cap exposure
        if not performance_guard_triggered and equity_rows:
            forward_days_elapsed = len(equity_rows)
            if forward_days_elapsed >= performance_guard_days:
                first_close_guard = close.loc[forward_dates[0]]
                if isinstance(first_close_guard, pd.Series):
                    first_close_guard = first_close_guard.iloc[0]
                bh_ret_guard = (c/first_close_guard - 1) if first_close_guard else 0
                strat_ret_guard = ((cash + shares*c)/initial_capital - 1)
                capture_guard = strat_ret_guard/(bh_ret_guard+1e-9) if bh_ret_guard>0 else strat_ret_guard
                # Skip or relax guard for historically strong symbols
                guard_min_cap = performance_guard_min_capture
                if symbol in guard_strong_symbols:
                    guard_min_cap *= 0.6  # allow more leniency
                # Ensure sufficient trade sample before evaluating
                if len([t for t in trades if t['action']=='SELL']) >= performance_guard_min_trades:
                    if capture_guard < guard_min_cap:
                        performance_guard_triggered = True
                        guard_trigger_date = current_date
        if performance_guard_triggered:
            # Recompute capture to allow dynamic ramp / potential lift
            first_close_guard2 = close.loc[forward_dates[0]]
            if isinstance(first_close_guard2, pd.Series):
                first_close_guard2 = first_close_guard2.iloc[0]
            bh_ret_guard2 = (c/first_close_guard2 - 1) if first_close_guard2 else 0
            strat_ret_guard2 = ((cash + shares*c)/initial_capital - 1)
            capture_guard2 = strat_ret_guard2/(bh_ret_guard2+1e-9) if bh_ret_guard2>0 else strat_ret_guard2
            # Lift guard if recovery strong
            if capture_guard2 >= guard_lift_capture:
                performance_guard_triggered = False
                guard_trigger_date = None
            else:
                # Ramp allowed fraction upward slowly to avoid permanent suppression
                if guard_trigger_date is not None:
                    days_since_guard = (current_date - guard_trigger_date).days
                else:
                    days_since_guard = 0
                ramp_progress = min(1.0, days_since_guard/guard_ramp_days) if guard_ramp_days>0 else 1.0
                # Improvement factor (0 -> 1) relative to lift target
                improvement_factor = min(1.0, capture_guard2 / max(1e-6, guard_lift_capture))
                max_allowed = performance_guard_max_fraction + (early_core_fraction - performance_guard_max_fraction) * 0.5 * ramp_progress * improvement_factor
                target_fraction = min(target_fraction, max_allowed)
        # Rolling expectancy guard (post performance guard)
        if enable_expectancy_guard and recent_trade_returns:
            if len(recent_trade_returns) >= rolling_expectancy_window:
                window_returns = recent_trade_returns[-rolling_expectancy_window:]
                avg_win_recent = np.mean([r for r in window_returns if r > 0]) if any(r>0 for r in window_returns) else 0.0
                avg_loss_recent = np.mean([r for r in window_returns if r < 0]) if any(r<0 for r in window_returns) else 0.0
                win_rate_recent = np.mean([r>0 for r in window_returns])
                expectancy_recent = win_rate_recent*avg_win_recent + (1-win_rate_recent)*avg_loss_recent
                if expectancy_recent < 0:
                    target_fraction = min(target_fraction, expectancy_guard_max_fraction)
        # Churn cooldown flag (block new/add entries while active)
        churn_cooldown_active = churn_cooldown_active_until is not None and current_date <= churn_cooldown_active_until
        # Volatility position cap when ATR% extreme
        atr_pct_val = atr_pct_series.loc[current_date]
        if isinstance(atr_pct_val, pd.Series): atr_pct_val = atr_pct_val.iloc[0]
        atr_pct_cap = atr_pct_series.rolling(252).quantile(vol_cap_percentile if 0<vol_cap_percentile<1 else 0.90)
        cap_ref = atr_pct_cap.loc[current_date] if current_date in atr_pct_cap.index else np.nan
        if pd.notna(atr_pct_val) and pd.notna(cap_ref) and atr_pct_val >= cap_ref:
            target_fraction = min(target_fraction, vol_cap_fraction)

        if target_fraction > 0 and not np.isnan(a) and a > 0:
            # Initial stop distance assumption: atr_initial_mult * a
            eff_atr_initial_mult = atr_initial_mult
            if enable_atr_vol_normalization:
                atr_rank_val = atr_pct_rank.loc[current_date]
                if isinstance(atr_rank_val, pd.Series):
                    atr_rank_val = atr_rank_val.iloc[0]
                if pd.notna(atr_rank_val):
                    # Normalize: higher ATR rank -> slightly tighter (reduce multiple), lower rank -> wider up to bounds
                    norm_factor = np.sqrt(0.5 / max(0.05, atr_rank_val))  # >1 if rank <0.5 (low vol), <1 if high vol
                    norm_factor = float(np.clip(norm_factor, 0.7, 1.3))
                    eff_atr_initial_mult = atr_initial_mult * norm_factor
            stop_dist = eff_atr_initial_mult * a
            if stop_dist > 0:
                # Use current equity (compounding) for risk sizing
                current_equity = cash + shares*c
                risk_per_trade = current_equity * dynamic_risk_perc
                max_risk_shares = risk_per_trade / stop_dist
                # Notional cap = cash + value of existing position
                max_notional_shares = (cash + shares*c)/c
                full_position_shares = min(max_risk_shares, max_notional_shares)
                desired_shares = target_fraction * full_position_shares
        action = None
        # Quality filter gate (prevents new/add entries when conditions poor)
        if enable_quality_filter and target_fraction > shares and not in_position:
            rsi_val = rsi_series.loc[current_date]
            if isinstance(rsi_val, pd.Series): rsi_val = rsi_val.iloc[0]
            vol_ok = True
            vol_row = px['Volume'].loc[current_date]
            if isinstance(vol_row, pd.Series): vol_row = vol_row.iloc[0]
            vol20_row = vol20.loc[current_date]
            if isinstance(vol20_row, pd.Series): vol20_row = vol20_row.iloc[0]
            vol_ok = (pd.notna(vol_row) and pd.notna(vol20_row) and vol_row >= quality_volume_multiplier * vol20_row)
            rsi_ok = (pd.notna(rsi_val) and quality_rsi_min <= rsi_val <= quality_rsi_max)
            prob_z_ok = (prob_z >= quality_prob_z_min)
            if not (vol_ok and rsi_ok and prob_z_ok):
                # gate by reducing target_fraction to zero for this bar
                target_fraction = 0.0
        # Entry / Add logic
        if desired_shares > shares + 1e-6:
            if 'churn_cooldown_active' in locals() and churn_cooldown_active:
                desired_shares = shares  # suppress adds during cooldown
            add_shares = desired_shares - shares
            cost_value = add_shares * c
            tcost = cost_value * (transaction_cost_bps/10000.0)
            total_cost = cost_value + tcost
            if total_cost <= cash and add_shares > 0:
                shares += add_shares
                cash -= total_cost
                if not in_position:
                    in_position = True
                    entry_price = c
                    entry_date = current_date
                    stop_price = c - atr_initial_mult * a if not np.isnan(a) else c*0.9
                    entry_prob = p
                    peak_prob = p
                    peak_close_since_entry = c
                    open_trade_stats = {'max_run_up':0.0,'max_drawdown':0.0}
                    cost_basis = c
                else:
                    # Update weighted cost basis
                    prev = cost_basis if cost_basis is not None else entry_price
                    cost_basis = ((prev * (shares - add_shares)) + c*add_shares)/shares if shares>0 else c
                # Add trade record
                trades.append({'date': current_date, 'action':'BUY', 'price': c, 'shares': add_shares, 'prob': p})
                action = 'BUY'
    # Trailing stop update
        if in_position and not np.isnan(a):
            # Adaptive trailing: tighten after sufficient run-up and volatility compression
            run_up_for_trail = (peak_close_since_entry/entry_price - 1) if (peak_close_since_entry and entry_price) else 0.0
            eff_trail_mult = atr_trail_mult
            if run_up_for_trail >= adaptive_trail_gain_threshold:
                eff_trail_mult = min(atr_trail_mult, adaptive_trail_mult_after_gain)
            if enable_vol_adaptive_trail:
                atr_norm_val = atr_pct_norm.loc[current_date]
                if isinstance(atr_norm_val, pd.Series):
                    atr_norm_val = atr_norm_val.iloc[0]
                if pd.notna(atr_norm_val) and atr_norm_val < low_atr_norm_threshold:
                    # Compress trail multiplier proportional to volatility compression but not below floor
                    eff_trail_mult = max(vol_trail_floor_mult, eff_trail_mult * max(0.4, atr_norm_val))
            if enable_atr_vol_normalization:
                atr_rank_val2 = atr_pct_rank.loc[current_date]
                if isinstance(atr_rank_val2, pd.Series):
                    atr_rank_val2 = atr_rank_val2.iloc[0]
                if pd.notna(atr_rank_val2):
                    norm_factor2 = np.sqrt(0.5 / max(0.05, atr_rank_val2))
                    norm_factor2 = float(np.clip(norm_factor2, 0.7, 1.3))
                    eff_trail_mult *= norm_factor2
            trail_candidate = peak_close_since_entry - eff_trail_mult * a if peak_close_since_entry else None
            if trail_candidate is not None:
                stop_price = max(stop_price, trail_candidate)
            # Ratchet stop: lock portion of gains after large run-up
            if entry_price and run_up_for_trail >= ratchet_runup_trigger:
                lock_level = entry_price * (1 + ratchet_min_locked_gain)
                stop_price = max(stop_price, lock_level)
    # Profit-taking ladder (partial realizes)
        if in_position and entry_price:
            current_runup = (c/entry_price) - 1
            for i, lvl in enumerate(profit_ladder):
                if current_runup >= lvl and not ladder_flags[i]:
                    trim_frac = profit_trim_fractions[i] if i < len(profit_trim_fractions) else profit_trim_fractions[-1]
                    if shares > 0 and trim_frac > 0:
                        trim_shares = shares * trim_frac
                        proceeds = trim_shares * c
                        tcost = proceeds * (transaction_cost_bps/10000.0)
                        cash += (proceeds - tcost)
                        shares -= trim_shares
                        # Realize PnL for trimmed portion
                        if cost_basis is not None:
                            record_partial_trade(current_date, c, trim_shares, f'PROFIT_LVL_{lvl:.0%}', p)
                        trades.append({'date': current_date,'action':'SELL_PART','price': c,'shares': trim_shares,'prob': p,'reason':f'PROFIT_LVL_{lvl:.0%}'})
                        ladder_flags[i] = True
        # Re-add logic after trims: if near peak and probability supportive, attempt to increase target fraction
        if enable_readd_after_trim and in_position and entry_price and any(ladder_flags):
            if peak_close_since_entry and c >= peak_close_since_entry * (1 - readd_pullback_pct):
                if p >= first_tier and target_fraction < 1.0:
                    # Encourage rebuilding toward full size
                    target_fraction = max(target_fraction, min(1.0, target_fraction + 0.2))

        # Exit conditions
        exit_reason = None
        if in_position:
            held_days = (current_date - entry_date).days if entry_date else 0
            # Probability based exits
            # Probability EMA cross exit
            prob_ema_fast_series = pd.Series(prob_history[-(ema_prob_slow+5):]).ewm(span=ema_prob_fast).mean()
            prob_ema_slow_series = pd.Series(prob_history[-(ema_prob_slow+5):]).ewm(span=ema_prob_slow).mean()
            prob_fast = float(prob_ema_fast_series.iloc[-1]) if len(prob_ema_fast_series)>0 else p
            prob_slow = float(prob_ema_slow_series.iloc[-1]) if len(prob_ema_slow_series)>0 else p
            ema_exit = enable_prob_ema_exit and prob_fast < prob_slow and p < first_tier
            if ema_exit:
                exit_reason = 'EMA_PROB'
            elif p < hard_exit_prob:
                exit_reason = 'HARD_PROB'
            elif entry_price and held_days >= min_holding_days and p < soft_exit_prob:
                # Partial de-risk first if configured
                if p >= partial_derisk_prob and partial_derisk_fraction>0 and shares > 0:
                    trim_shares = shares * partial_derisk_fraction
                    proceeds = trim_shares * c
                    tcost = proceeds * (transaction_cost_bps/10000.0)
                    cash += (proceeds - tcost)
                    shares -= trim_shares
                    trades.append({'date': current_date,'action':'SELL_PART','price': c,'shares': trim_shares,'prob': p,'reason':'PARTIAL_SOFT_PROB'})
                    # Recalculate target_fraction continuing; do not full exit
                else:
                    exit_reason = 'SOFT_PROB'
            # Trend breakdown
            if exit_reason is None and (not np.isnan(s100) and c < s100):
                exit_reason = 'TREND'
            # Trailing / initial stop
            if exit_reason is None and stop_price is not None and c <= stop_price:
                exit_reason = 'STOP'
            # Stale position exit
            if exit_reason is None and held_days > stale_days and ((c/entry_price)-1) < stale_min_runup:
                exit_reason = 'STALE'
            # Underperformance re-evaluation exit after threshold days
            if exit_reason is None and entry_date and held_days >= underperf_reval_days:
                # Compare position capture vs symbol since entry
                symbol_ret = (c/entry_price - 1) if entry_price else 0.0
                pos_ret = (c/entry_price - 1)
                capture_local = pos_ret/(symbol_ret+1e-9) if symbol_ret>0 else 0.0
                if symbol_ret > 0 and capture_local < underperf_capture_floor:
                    exit_reason = 'UNDERPERF'
            # Time based (>90 calendar days)
            if exit_reason is None and entry_date and (current_date - entry_date).days > 90:
                exit_reason = 'TIME'
            if exit_reason:
                proceeds = shares * c
                tcost = proceeds * (transaction_cost_bps/10000.0)
                cash += (proceeds - tcost)
                pnl = (c - (cost_basis if cost_basis is not None else entry_price)) * shares
                realized_pnl += pnl
                ret_pct = pnl/(entry_price*shares) if entry_price else 0.0
                trades.append({'date': current_date, 'action':'SELL', 'price': c, 'shares': shares, 'prob': p, 'reason': exit_reason})
                completed_trades.append(TradeRecord(
                    entry_date=next((t['date'] for t in trades if t['action']=='BUY'), current_date),
                    exit_date=current_date,
                    entry_price=entry_price,
                    exit_price=c,
                    shares=shares,
                    pnl=pnl,
                    return_pct=ret_pct,
                    max_run_up=open_trade_stats['max_run_up'],
                    max_drawdown=open_trade_stats['max_drawdown'],
                    entry_prob=entry_prob,
                    peak_prob=peak_prob,
                    exit_prob=p
                ))
                # Update churn / expectancy state
                if pnl < 0:
                    consecutive_loss_trades += 1
                else:
                    consecutive_loss_trades = 0
                recent_trade_returns.append(ret_pct)
                if churn_cooldown_trades > 0 and consecutive_loss_trades >= churn_cooldown_trades:
                    churn_cooldown_active_until = current_date + pd.Timedelta(days=churn_cooldown_days)
                    consecutive_loss_trades = 0
                shares = 0.0
                in_position = False
                entry_price = None
                entry_date = None
                stop_price = None
                entry_prob = None
                peak_prob = 0.0
                peak_close_since_entry = None
                ladder_flags = [False]*len(profit_ladder)
                last_exit_date = current_date
                cost_basis = None
        equity = cash + shares*c
        # Approx exposure fraction vs current equity (avoid div by zero) and store target fraction for diagnostics
        exposure_fraction = 0.0
        if equity > 0 and c > 0:
            exposure_fraction = (shares * c) / equity
        equity_rows.append({
            'date': current_date,
            'equity': equity,
            'cash': cash,
            'shares': shares,
            'prob': p,
            'stop': stop_price,
            'target_fraction': target_fraction,
            'exposure_fraction': exposure_fraction
        })

    # Force final exit to realize unrealized PnL if requested
    if finalize_at_end and in_position and equity_rows:
        last_row = equity_rows[-1]
        c_last = last_row['equity'] - last_row['cash']  # not direct; recompute price from close series
        final_close_val = close.loc[forward_dates[-1]]
        if isinstance(final_close_val, pd.Series):
            final_close_val = final_close_val.iloc[0]
        final_close = float(final_close_val)
        proceeds = shares * final_close
        tcost = proceeds * (transaction_cost_bps/10000.0)
        cash += (proceeds - tcost)
        pnl = (final_close - (cost_basis if cost_basis is not None else entry_price)) * shares
        pnl = float(pnl)
        realized_pnl += pnl
        ret_pct = pnl/(entry_price*shares) if entry_price else 0.0
        trades.append({'date': forward_dates[-1], 'action':'SELL', 'price': final_close, 'shares': shares, 'prob': p, 'reason': 'FORCED_FINAL'})
        completed_trades.append(TradeRecord(
            entry_date=entry_date,
            exit_date=forward_dates[-1],
            entry_price=cost_basis if cost_basis is not None else entry_price,
            exit_price=final_close,
            shares=shares,
            pnl=pnl,
            return_pct=ret_pct,
            max_run_up=open_trade_stats['max_run_up'],
            max_drawdown=open_trade_stats['max_drawdown'],
            entry_prob=entry_prob,
            peak_prob=peak_prob,
            exit_prob=p
        ))
        # Update churn / expectancy state for forced final exit
        if pnl < 0:
            consecutive_loss_trades += 1
        else:
            consecutive_loss_trades = 0
        recent_trade_returns.append(ret_pct)
        if churn_cooldown_trades > 0 and consecutive_loss_trades >= churn_cooldown_trades:
            churn_cooldown_active_until = forward_dates[-1] + pd.Timedelta(days=churn_cooldown_days)
            consecutive_loss_trades = 0
        shares = 0.0
        in_position = False
    equity_df = pd.DataFrame(equity_rows).set_index('date')
    trades_df = pd.DataFrame(trades)

    # Metrics
    if not equity_df.empty:
        final_equity = float(equity_df['equity'].iloc[-1])
        total_ret = final_equity / initial_capital - 1
        daily_ret = equity_df['equity'].pct_change().dropna()
        vol = float(daily_ret.std()*np.sqrt(252)) if not daily_ret.empty else float('nan')
        sharpe = float(daily_ret.mean()/ (daily_ret.std()+1e-9) * np.sqrt(252)) if not daily_ret.empty else float('nan')
        downside = daily_ret[daily_ret<0]
        sortino = float(daily_ret.mean()/ (downside.std()+1e-9) * np.sqrt(252)) if not downside.empty else float('nan')
        dd = equity_df['equity']/equity_df['equity'].cummax() - 1
        max_dd = float(dd.min())
        cagr = (1+total_ret)**(252/len(daily_ret)) - 1 if len(daily_ret)>50 else float('nan')
    else:
        final_equity = initial_capital
        total_ret = 0.0
        vol = sharpe = sortino = max_dd = cagr = float('nan')

    ct_df = pd.DataFrame([t.__dict__ for t in completed_trades]) if completed_trades else pd.DataFrame()
    if not ct_df.empty:
        # Ensure numeric types
        ct_df['pnl'] = pd.to_numeric(ct_df['pnl'], errors='coerce')
        ct_df['return_pct'] = pd.to_numeric(ct_df['return_pct'], errors='coerce')
        win_rate = float((ct_df['pnl'] > 0).mean())
        positive_mask = ct_df['pnl'] > 0
        negative_mask = ct_df['pnl'] < 0
        avg_win = float(ct_df.loc[positive_mask, 'return_pct'].mean()) if positive_mask.any() else 0.0
        avg_loss = float(ct_df.loc[negative_mask, 'return_pct'].mean()) if negative_mask.any() else 0.0
        expectancy = (win_rate*avg_win + (1-win_rate)*avg_loss)
    else:
        win_rate = avg_win = avg_loss = expectancy = float('nan')

    # Buy & hold benchmark
    fwd_slice = close.loc[fwd_start:fwd_end]
    if not fwd_slice.empty:
        try:
            last_price = fwd_slice.iloc[-1]
            first_price = fwd_slice.iloc[0]
            # handle potential Series (should normally be scalar); extract .iloc[0] if needed
            if hasattr(last_price, 'iloc') and not isinstance(last_price, (float,int,np.floating)):
                last_price = float(last_price.iloc[0])
            if hasattr(first_price, 'iloc') and not isinstance(first_price, (float,int,np.floating)):
                first_price = float(first_price.iloc[0])
            bh_ret = float(float(last_price)/float(first_price) - 1)
        except Exception:
            bh_ret = float('nan')
    else:
        bh_ret = float('nan')

    # Unrealized PnL at end (should be zero if forced exit executed)
    if in_position and cost_basis is not None:
        unrealized_pnl = (close.loc[forward_dates[-1]] - cost_basis) * shares
    else:
        unrealized_pnl = 0.0
    # Exposure diagnostics
    if equity_rows:
        er_df = pd.DataFrame(equity_rows)
        avg_exposure_fraction = float(er_df['exposure_fraction'].mean()) if 'exposure_fraction' in er_df else float('nan')
        pct_days_in_market = float((er_df['exposure_fraction'] > 0).mean()) if 'exposure_fraction' in er_df else float('nan')
    else:
        avg_exposure_fraction = pct_days_in_market = float('nan')

    summary = {
        'symbol': symbol,
        'initial_capital': initial_capital,
    'final_equity': float(final_equity),
    'total_return_pct': float(total_ret),
    'buy_hold_return_pct': float(bh_ret),
    'alpha_vs_buy_hold_pct': float(total_ret - bh_ret),
    'vol_annual': float(vol) if not isinstance(vol, str) else vol,
    'sharpe': float(sharpe) if not isinstance(sharpe, str) else sharpe,
    'sortino': float(sortino) if not isinstance(sortino, str) else sortino,
    'max_drawdown': float(max_dd),
    'cagr': float(cagr) if not isinstance(cagr, str) else cagr,
        'num_fills': int(len(trades_df)),
        'completed_trades': int(len(ct_df)),
        'win_rate': win_rate,
        'avg_win_return_pct': avg_win,
        'avg_loss_return_pct': avg_loss,
        'expectancy_return_pct': expectancy,
        'realized_pnl': realized_pnl,
        'unrealized_pnl_end': unrealized_pnl,
    'captured_buy_hold_ratio': float(total_ret/bh_ret) if (bh_ret and not np.isnan(bh_ret)) else float('nan'),
    'valid_capture_ratio': float(total_ret/bh_ret) if (bh_ret > 0 and not np.isnan(bh_ret)) else float('nan'),
        'tier_mode': tier_mode,
        'used_tier_probs': current_tier_probs if tier_mode.lower()=='quantile' else list(tier_probs),
        'enable_trend_filter': enable_trend_filter,
    'splits_count': len(split_days),
        'last_split_date': str(split_days[-1].date()) if split_days else None,
    'performance_guard_triggered': performance_guard_triggered,
    'performance_guard_trigger_date': str(guard_trigger_date.date()) if guard_trigger_date else None,
    'churn_cooldown_active': bool(churn_cooldown_active_until is not None and len(forward_dates) > 0 and churn_cooldown_active_until >= forward_dates[-1]),
    'recent_trades_considered_for_expectancy': len(recent_trade_returns),
    'classification': classification_used,
    'rollback_guard': rollback_guard,
    'alpha_dollars': float(final_equity - initial_capital - (initial_capital * (bh_ret if not np.isnan(bh_ret) else 0.0))),
    'avg_exposure_fraction': avg_exposure_fraction,
    'pct_days_in_market': pct_days_in_market,
    }
    # Debug: coerce any residual Series
    for k,v in list(summary.items()):
        if isinstance(v, pd.Series):
            summary[k] = float(v.iloc[0])

    # Persist
    equity_df.to_csv(Path(out_dir)/'equity_curve.csv')
    trades_df.to_csv(Path(out_dir)/'trades.csv', index=False)
    import json
    with open(Path(out_dir)/'summary.json','w') as f:
        json.dump(summary, f, indent=2)
    # Config metadata for audit
    config_meta = {
        'symbol': symbol,
        'tier_mode': tier_mode,
        'tier_quantiles': tier_quantiles,
        'used_tier_probs': summary.get('used_tier_probs'),
        'tier_min_spacing': '%.3f' % (tier_min_spacing if 'tier_min_spacing' in locals() else 0.0),
        'performance_guard': {
            'days': performance_guard_days,
            'min_capture': performance_guard_min_capture,
            'max_fraction': performance_guard_max_fraction,
            'triggered': performance_guard_triggered
        },
        'trend_filter': {
            'enabled': enable_trend_filter,
            'min_score': trend_min_score
        },
        'atr_vol_normalization': enable_atr_vol_normalization,
        'churn_cooldown': {
            'enabled': churn_cooldown_trades>0,
            'loss_trades_threshold': churn_cooldown_trades,
            'cooldown_days': churn_cooldown_days
        },
        'expectancy_guard': {
            'enabled': enable_expectancy_guard,
            'window': rolling_expectancy_window,
            'max_fraction': expectancy_guard_max_fraction
        },
        'tier_smoothing': {
            'enabled': enable_tier_smoothing,
            'alpha': tier_smoothing_alpha
        },
    'core_fraction': {
            'early_core_fraction': early_core_fraction,
            'guard_core_fraction': guard_core_fraction
    },
    'classification': classification_used,
    'rollback_guard': rollback_guard
    }
    with open(Path(out_dir)/'config_meta.json','w') as f:
        json.dump(config_meta, f, indent=2)

    if chart and not equity_df.empty:
        # Equity
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(equity_df.index, equity_df['equity'], label='Strategy', color='navy')
        if not fwd_slice.empty:
            bh_equity = initial_capital * (fwd_slice / fwd_slice.iloc[0])
            ax.plot(bh_equity.index, bh_equity.values, label='Buy & Hold', color='gray', linestyle='--')
        ax.set_title(f'{symbol} Institutional Strategy vs Buy & Hold')
        ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
        fig.savefig(Path(out_dir)/'equity_curve.png'); plt.close(fig)
        # Price & trades
        fig2, ax1 = plt.subplots(figsize=(11,5))
        ax1.plot(close.loc[fwd_start:fwd_end].index, close.loc[fwd_start:fwd_end], color='blue', label='Close')
        if not trades_df.empty:
            tdf = trades_df.copy(); tdf['date'] = pd.to_datetime(tdf['date']); tdf = tdf.set_index('date')
            buys = tdf[tdf.action=='BUY']; sells = tdf[tdf.action=='SELL']
            if not buys.empty:
                ax1.scatter(buys.index, buys.price, marker='^', color='green', s=70, edgecolors='k', linewidths=0.5, label='BUY')
            if not sells.empty:
                ax1.scatter(sells.index, sells.price, marker='v', color='red', s=70, edgecolors='k', linewidths=0.5, label='SELL')
        ax2 = ax1.twinx()
        ax2.plot(equity_df.index, equity_df['prob'], color='orange', alpha=0.6, label='Prob')
        ax1.set_title(f'{symbol} Institutional Trades & Probability')
        ax1.grid(alpha=0.25)
        lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='upper left')
        fig2.tight_layout(); fig2.savefig(Path(out_dir)/f'{symbol.lower()}_trades.png'); plt.close(fig2)

    return summary

if __name__ == '__main__':
    symbols = ['NVDA','AAPL','MSFT','TSLA','META']
    results = {}
    for sym in symbols:
        print(f'Running institutional strategy for {sym}...')
        res = run_nvda_institutional(symbol=sym)
        results[sym] = res
        print(res)
    print('Summary capture ratios:')
    for sym, res in results.items():
        print(sym, round(res.get('captured_buy_hold_ratio', float('nan')),4))
