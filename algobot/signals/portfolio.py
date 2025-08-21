"""
Signal and portfolio construction module for ML-driven trading pipeline.
Implements volatility scaling, Kelly capping, position/sector caps, ATR-based stops, trailing stops, profit bands as per prompt.yaml.
"""
import numpy as np
import pandas as pd

def volatility_scaling(signal: pd.Series, target_vol: float, realized_vol: pd.Series) -> pd.Series:
    scale = target_vol / (realized_vol + 1e-9)
    return signal * scale

def kelly_fraction(prob: pd.Series, edge: float = 0.5, cap: float = 0.25) -> pd.Series:
    kelly = (prob - (1-prob)/edge)
    kelly = np.clip(kelly, 0, cap)
    return kelly

def apply_position_caps(weights: pd.Series, max_name: float, max_sector: float, sector_map: dict = None) -> pd.Series:
    weights = np.clip(weights, 0, max_name)
    if sector_map:
        sector_weights = weights.groupby(sector_map).sum()
        for sector, w in sector_weights.items():
            if w > max_sector:
                idx = [i for i, s in sector_map.items() if s == sector]
                weights[idx] *= max_sector / w
    return weights

def atr_stop_loss(entry: float, side: str, atr: float, multiple: float = 2.0) -> float:
    if side == 'long':
        return entry - multiple * atr
    else:
        return entry + multiple * atr

def trailing_stop(entry: float, current: float, atr: float, multiple: float = 1.5, side: str = 'long') -> float:
    if side == 'long':
        return max(entry, current - multiple * atr)
    else:
        return min(entry, current + multiple * atr)

def profit_band(entry: float, atr: float, multiple: float = 2.0, side: str = 'long') -> float:
    if side == 'long':
        return entry + multiple * atr
    else:
        return entry - multiple * atr
