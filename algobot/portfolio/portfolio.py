"""Portfolio module: tracks positions, cash, P&L, exposure.

This provides a clean abstraction separating trading logic (signals & execution)
from state management and performance accounting.

Key concepts:
  - Position: lot-level tracking (single entry assumed; extendable)
  - Portfolio: holds positions + cash, computes metrics on demand
  - Metrics: unrealized/realized P&L, total equity, allocation %, drawdown placeholder

Future extensions:
  - Multiple lots per symbol (FIFO / average cost)
  - Corporate actions adjustments
  - Risk factor exposures (sector, beta, etc.)
  - Performance attribution
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone


@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float
    entry_time: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    realized_pnl: float = 0.0  # realized profit after partial/closed

    def market_value(self, price: float) -> float:
        return self.shares * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.shares

    def as_dict(self, price: Optional[float] = None) -> Dict[str, Any]:
        d = asdict(self)
        if price is not None:
            d.update({
                'market_value': self.market_value(price),
                'unrealized_pnl': self.unrealized_pnl(price),
                'unrealized_pnl_pct': (price / self.entry_price - 1) if self.entry_price else None
            })
        return d


class Portfolio:
    def __init__(self, initial_cash: float = 100_000.0):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: Dict[str, Position] = {}
        self.realized_pnl_total = 0.0
        self._equity_history: List[Dict[str, Any]] = []

    # --- Position Management -------------------------------------------------
    def open_position(self, symbol: str, shares: float, price: float,
                      stop_loss: Optional[float] = None,
                      take_profit: Optional[float] = None) -> Position:
        if symbol in self.positions:
            raise ValueError(f"Position already exists for {symbol}; use adjust or implement multi-lot support.")
        cost = shares * price
        if cost > self.cash + 1e-9:
            raise ValueError("Insufficient cash to open position")
        self.cash -= cost
        pos = Position(
            symbol=symbol,
            shares=shares,
            entry_price=price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        self.positions[symbol] = pos
        return pos

    def close_position(self, symbol: str, price: float) -> Dict[str, Any]:
        if symbol not in self.positions:
            raise ValueError(f"No position for {symbol}")
        pos = self.positions[symbol]
        proceeds = pos.shares * price
        pnl = (price - pos.entry_price) * pos.shares
        self.cash += proceeds
        self.realized_pnl_total += pnl
        del self.positions[symbol]
        return {
            'symbol': symbol,
            'shares': pos.shares,
            'entry_price': pos.entry_price,
            'exit_price': price,
            'pnl': pnl,
            'pnl_pct': (price / pos.entry_price - 1) if pos.entry_price else None
        }

    def update_stops(self, symbol: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None):
        if symbol not in self.positions:
            raise ValueError(f"No position for {symbol}")
        if stop_loss is not None:
            self.positions[symbol].stop_loss = stop_loss
        if take_profit is not None:
            self.positions[symbol].take_profit = take_profit

    # --- Valuation & Metrics -------------------------------------------------
    def mark_to_market(self, prices: Dict[str, float]) -> Dict[str, Any]:
        total_position_value = 0.0
        unrealized_total = 0.0
        per_symbol = []
        for sym, pos in self.positions.items():
            price = prices.get(sym)
            if price is None:
                continue
            mv = pos.market_value(price)
            upnl = pos.unrealized_pnl(price)
            total_position_value += mv
            unrealized_total += upnl
            per_symbol.append({
                'symbol': sym,
                'shares': pos.shares,
                'price': price,
                'market_value': mv,
                'unrealized_pnl': upnl,
                'unrealized_pnl_pct': (price / pos.entry_price - 1) if pos.entry_price else None,
                'weight_pct': None
            })
        equity = self.cash + total_position_value
        for row in per_symbol:
            row['weight_pct'] = (row['market_value'] / equity * 100.0) if equity > 0 else 0.0
        snapshot = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cash': self.cash,
            'equity': equity,
            'positions_value': total_position_value,
            'positions_count': len(per_symbol),
            'unrealized_pnl_total': unrealized_total,
            'realized_pnl_total': self.realized_pnl_total,
            'total_return_pct': (equity + self.realized_pnl_total - self.initial_cash) / self.initial_cash,
            'positions': per_symbol
        }
        self._equity_history.append({k: snapshot[k] for k in ('timestamp','equity')})
        return snapshot

    def equity_history(self) -> List[Dict[str, Any]]:
        return self._equity_history

    def current_allocation_pct(self) -> float:
        latest = self._equity_history[-1] if self._equity_history else None
        if not latest:
            return 0.0
        return 100.0 * (1 - self.cash / latest['equity']) if latest['equity'] else 0.0

    def symbol_exposure(self) -> Dict[str, float]:
        exposures = {}
        total = sum(p.shares * p.entry_price for p in self.positions.values())
        if total <= 0:
            return exposures
        for sym, pos in self.positions.items():
            exposures[sym] = (pos.shares * pos.entry_price) / total
        return exposures

    def max_drawdown(self) -> Optional[float]:
        if len(self._equity_history) < 2:
            return None
        eq = [h['equity'] for h in self._equity_history]
        peak = eq[0]
        max_dd = 0.0
        for v in eq:
            if v > peak:
                peak = v
            dd = (v / peak) - 1
            if dd < max_dd:
                max_dd = dd
        return max_dd


__all__ = ["Portfolio", "Position"]
