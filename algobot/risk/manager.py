"""Core risk management extraction with ATR & simple VaR placeholder."""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
try:
    # Optional import for centralized config
    from algobot.config import GLOBAL_CONFIG as _CFG
except Exception:
    _CFG = None


@dataclass
class TradeLimits:
    max_position_pct: float = 15.0  # per-name cap (percentage points)
    max_portfolio_risk_pct: float = 0.60  # total gross exposure cap (fraction of equity)
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.15
    concentration_limit: float = 40.0  # top-name share of portfolio exposure (%), heuristic


class RiskManagerCore:
    def __init__(self, limits: TradeLimits | None = None):
        if limits is not None:
            self.limits = limits
        elif _CFG is not None:
            # Use centralized config as single source of truth
            self.limits = TradeLimits(
                max_position_pct=float(_CFG.risk.max_position_pct) * 100.0,
                max_portfolio_risk_pct=float(_CFG.risk.max_portfolio_risk_pct),
                stop_loss_pct=float(_CFG.risk.stop_loss_pct),
                take_profit_pct=float(_CFG.risk.take_profit_pct),
            )
        else:
            self.limits = TradeLimits()
        self._recent_returns = []  # list of recent daily returns for VaR

    def validate(self, symbol: str, position_pct: float, current_positions: Dict[str, Dict]) -> Dict:
        """Validate a new position addition against limits.

        Args:
            symbol: Ticker symbol.
            position_pct: Proposed position percentage (0-100 scale).
            current_positions: Mapping of symbol -> {'percentage': float, ...}.
        """
        total_pct = sum(p.get('percentage', 0) for p in current_positions.values())
        new_total = total_pct + position_pct
        approved = True
        reasons = []
        if position_pct > self.limits.max_position_pct:
            approved = False
            reasons.append("position_exceeds_limit")
        if new_total > self.limits.max_portfolio_risk_pct * 100:
            approved = False
            reasons.append("portfolio_risk_exceeded")
        # Concentration check (approximate: if new symbol would exceed concentration limit)
        max_existing = 0.0
        for s, p in current_positions.items():
            max_existing = max(max_existing, float(p.get('percentage', 0)))
        max_after = max(max_existing, position_pct)
        if max_after > self.limits.concentration_limit:
            approved = False
            reasons.append("concentration_exceeded")
        return {"approved": approved, "reasons": reasons}

    def dynamic_limits(self, market_vol: Optional[float] = None, portfolio_concentration: Optional[float] = None) -> TradeLimits:
        """Return a volatility- and concentration-adjusted view of limits.

        market_vol: e.g., SPY 20d ATR% or VIX scaled to 0-1. If >0.25, tighten limits.
        portfolio_concentration: approximate HHI or top-name share in [0,1]. If >0.4 tighten single cap.
        """
        limits = TradeLimits(**vars(self.limits))
        if market_vol is not None:
            if market_vol > 0.25:
                limits.max_position_pct *= 0.7
                limits.max_portfolio_risk_pct *= 0.8
            elif market_vol < 0.15:
                limits.max_position_pct *= 1.1
                limits.max_portfolio_risk_pct *= 1.05
        if portfolio_concentration is not None and portfolio_concentration > 0.4:
            limits.max_position_pct *= 0.8
        return limits

    def stop_loss_price(self, entry: float, side: str) -> float:
        return entry * (1 - self.limits.stop_loss_pct) if side == 'BUY' else entry * (1 + self.limits.stop_loss_pct)

    def take_profit_price(self, entry: float, side: str) -> float:
        return entry * (1 + self.limits.take_profit_pct) if side == 'BUY' else entry * (1 - self.limits.take_profit_pct)

    # --- Advanced Helpers -------------------------------------------------
    @staticmethod
    def atr(prices: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Compute Average True Range from DataFrame with High, Low, Close."""
        cols = {'High','Low','Close'}
        if not cols.issubset(prices.columns):
            return None
        high = prices['High']
        low = prices['Low']
        close = prices['Close']
        prev_close = close.shift(1)
        tr = np.maximum(high-low, np.maximum(abs(high-prev_close), abs(low-prev_close)))
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if not np.isnan(atr) else None

    def atr_stop_loss(self, entry: float, side: str, atr_value: float, multiple: float = 2.0) -> float:
        if atr_value is None:
            return self.stop_loss_price(entry, side)
        if side == 'BUY':
            return entry - multiple * atr_value
        else:
            return entry + multiple * atr_value

    def update_var_history(self, daily_return: float, max_len: int = 250):
        self._recent_returns.append(daily_return)
        if len(self._recent_returns) > max_len:
            self._recent_returns.pop(0)

    def value_at_risk(self, confidence: float = 0.95) -> Optional[float]:
        if len(self._recent_returns) < 30:
            return None
        sorted_r = sorted(self._recent_returns)
        idx = int((1 - confidence) * len(sorted_r))
        return sorted_r[idx]

__all__ = ["RiskManagerCore", "TradeLimits"]
