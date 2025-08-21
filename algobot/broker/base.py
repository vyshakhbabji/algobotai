"""Broker abstraction interfaces.

Defines a minimal interface for live/paper trading adapters.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, List, Dict, Optional, Any


@dataclass
class Order:
    symbol: str
    qty: float
    side: str  # 'buy' or 'sell'
    order_type: str = 'market'
    time_in_force: str = 'day'
    id: Optional[str] = None
    status: Optional[str] = None
    filled_qty: float = 0.0


class BrokerClient(Protocol):
    def get_account(self) -> Dict[str, Any]: ...
    def get_positions(self) -> List[Dict[str, Any]]: ...
    def get_latest_price(self, symbol: str) -> Optional[float]: ...
    def submit_order(self, order: Order) -> Order: ...
    def cancel_order(self, order_id: str) -> bool: ...


__all__ = ["BrokerClient", "Order"]
