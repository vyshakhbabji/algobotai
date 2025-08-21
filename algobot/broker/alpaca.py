"""Alpaca broker adapter implementing BrokerClient interface."""
from __future__ import annotations
from typing import List, Dict, Optional, Any
from .base import BrokerClient, Order

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    _ALPACA_OK = True
except Exception:  # noqa
    _ALPACA_OK = False


class AlpacaBroker(BrokerClient):
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        if not _ALPACA_OK:
            raise ImportError("alpaca-py not installed. pip install alpaca-py")
        self.trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    def get_account(self) -> Dict[str, Any]:
        acct = self.trading_client.get_account()
        return {
            'cash': float(acct.cash),
            'buying_power': float(acct.buying_power),
            'equity': float(acct.equity),
            'portfolio_value': float(acct.portfolio_value),
            'status': acct.status
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        out = []
        for p in self.trading_client.get_all_positions():
            out.append({
                'symbol': p.symbol,
                'qty': float(p.qty),
                'market_value': float(p.market_value),
                'avg_entry_price': float(p.avg_entry_price),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc),
                'current_price': float(p.current_price)
            })
        return out

    def get_latest_price(self, symbol: str) -> Optional[float]:
        req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        q = self.data_client.get_stock_latest_quote(req)
        if symbol in q:
            quote = q[symbol]
            return (float(quote.bid_price) + float(quote.ask_price)) / 2 if quote.bid_price and quote.ask_price else None
        return None

    def submit_order(self, order: Order) -> Order:
        side_enum = OrderSide.BUY if order.side.lower() == 'buy' else OrderSide.SELL
        tif_enum = TimeInForce.DAY if order.time_in_force.lower() == 'day' else TimeInForce.GTC
        if order.order_type == 'market':
            req = MarketOrderRequest(symbol=order.symbol, qty=order.qty, side=side_enum, time_in_force=tif_enum)
        else:
            raise NotImplementedError("Only market orders implemented")
        resp = self.trading_client.submit_order(req)
        order.id = resp.id
        order.status = resp.status
        order.filled_qty = float(resp.filled_qty) if resp.filled_qty else 0.0
        return order

    def cancel_order(self, order_id: str) -> bool:
        self.trading_client.cancel_order_by_id(order_id)
        return True


__all__ = ["AlpacaBroker"]
