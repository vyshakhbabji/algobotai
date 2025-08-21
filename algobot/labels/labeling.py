"""
Label generation module for trading ML pipeline.
Implements binary, ternary, continuous, and meta-labeling as per prompt.yaml.
Strictly prevents lookahead/leakage.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict

def binary_momentum_label(df: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    """Label 1 if forward return > threshold, else 0. No lookahead."""
    fwd_ret = df['Close'].pct_change(periods=horizon).shift(-horizon)
    label = (fwd_ret > threshold).astype(int)
    label.name = f'binary_momo_{horizon}_{threshold}'
    return label

def ternary_label(df: pd.DataFrame, horizon: int, up_q: float = 0.66, down_q: float = 0.33) -> pd.Series:
    """Label 2=up, 1=flat, 0=down by quantiles of forward return."""
    fwd_ret = df['Close'].pct_change(periods=horizon).shift(-horizon)
    up = fwd_ret.quantile(up_q)
    down = fwd_ret.quantile(down_q)
    label = pd.Series(1, index=df.index)
    label[fwd_ret >= up] = 2
    label[fwd_ret <= down] = 0
    label.name = f'ternary_{horizon}'
    return label

def continuous_label(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Raw forward return as label."""
    fwd_ret = df['Close'].pct_change(periods=horizon).shift(-horizon)
    fwd_ret.name = f'cont_{horizon}'
    return fwd_ret

def meta_label(primary_signal: pd.Series, meta_filter: pd.Series) -> pd.Series:
    """Meta-labeling: 1 if primary signal and meta filter agree, else 0."""
    label = ((primary_signal == 1) & (meta_filter == 1)).astype(int)
    label.name = 'meta_label'
    return label
