"""Execution abstraction (simulation)."""
from datetime import datetime
from typing import Dict, List


class SimulatorExecutor:
    def __init__(self):
        self.trades: List[Dict] = []

    def market_order(self, symbol: str, side: str, shares: int, price: float) -> Dict:
        trade = {
            'ts': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'side': side,
            'shares': shares,
            'price': price,
            'value': shares * price,
            'id': f"SIM_{len(self.trades)+1:06d}"
        }
        self.trades.append(trade)
        return trade
