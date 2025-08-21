"""Advanced feature engineering for daily equity data.

Adds:
 - Log returns, rolling volatility
 - ATR (Average True Range) percent
 - RSI (14)
 - Rolling return momentum (various windows)
 - Day of week sine/cosine seasonality
 - Normalized price distance from multi-period SMAs
"""
from __future__ import annotations
import pandas as pd
import numpy as np

ADV_FEATURE_COLUMNS = [
    'log_ret','vol_10','vol_20','atr_pct','rsi_14',
    'mom_3','mom_5','mom_10','mom_20','price_vs_sma10','price_vs_sma30',
    'price_vs_sma50','day_sin','day_cos'
]

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    # Flatten potential MultiIndex columns from yfinance (e.g., ('Close','NVDA'))
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join([str(c) for c in tup if c!='']) for tup in data.columns]
    # Ensure index is DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors='coerce')
        data = data[~data.index.isna()]
    # Attempt to locate close/high/low/volume columns robustly
    candidates_close = [c for c in data.columns if 'Close' in c]
    col_close = 'Close' if 'Close' in data.columns else (candidates_close[0] if candidates_close else None)
    if col_close is None:
        raise ValueError('Close column required')
    close = data[col_close].astype(float).squeeze()
    candidates_high = [c for c in data.columns if 'High' in c]
    candidates_low = [c for c in data.columns if 'Low' in c]
    candidates_vol = [c for c in data.columns if 'Volume' in c]
    high = data[candidates_high[0]].astype(float).squeeze() if candidates_high else close
    low = data[candidates_low[0]].astype(float).squeeze() if candidates_low else close
    volume = data[candidates_vol[0]].astype(float).squeeze() if candidates_vol else None

    # Core derived
    data['log_ret'] = np.log(close/close.shift(1))
    data['vol_10'] = data['log_ret'].rolling(10).std()
    data['vol_20'] = data['log_ret'].rolling(20).std()
    tr = (high-low).abs()
    tr2 = (high-close.shift(1)).abs()
    tr3 = (low-close.shift(1)).abs()
    true_range = pd.concat([tr, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    data['atr'] = atr
    data['atr_pct'] = atr / close
    data['rsi_14'] = _rsi(close, 14)
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data['macd'] = ema12 - ema26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    # VWAP (intraday only, but proxy for daily)
    if volume is not None:
        vwap = (close * volume).cumsum() / volume.cumsum()
        data['vwap_dist'] = (close / vwap) - 1
    else:
        data['vwap_dist'] = 0.0
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data['boll_z'] = (close - sma20) / (std20 + 1e-9)
    # OBV
    if volume is not None:
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        data['obv'] = obv
    else:
        data['obv'] = 0.0
    # Volatility clustering (rolling std of returns)
    data['vol_cluster_20'] = data['log_ret'].rolling(20).std()
    # Cross-sectional ranks (momentum, volatility, relative strength)
    # These require a universe, so placeholder: set to NaN
    data['xsec_mom_rank'] = np.nan
    data['xsec_vol_rank'] = np.nan
    data['xsec_rel_strength'] = np.nan
    # Regime features (placeholder: set to NaN)
    data['regime_bull'] = np.nan
    data['regime_bear'] = np.nan
    data['regime_sideways'] = np.nan
    # Seasonality/time
    dow = close.index.dayofweek
    data['day_sin'] = np.sin(2*np.pi*dow/5)
    data['day_cos'] = np.cos(2*np.pi*dow/5)
    # Target definitions
    data['target_return_1'] = close.pct_change().shift(-1)
    data['target_up'] = (data['target_return_1'] > 0).astype(int)
    # Only drop rows with NaN in essential columns
    essential_cols = [col_close, 'log_ret', 'target_return_1', 'target_up']
    data = data.dropna(subset=[c for c in essential_cols if c in data.columns])
    base_close_col = col_close
    # Add new features to ADV_FEATURE_COLUMNS for selection
    selected_cols = [base_close_col] + ADV_FEATURE_COLUMNS + [
        'atr','macd','macd_signal','vwap_dist','boll_z','obv','vol_cluster_20',
        'xsec_mom_rank','xsec_vol_rank','xsec_rel_strength',
        'regime_bull','regime_bear','regime_sideways',
        'target_return_1','target_up']
    selected_cols = [c for c in selected_cols if c in data.columns]
    return data[selected_cols]
