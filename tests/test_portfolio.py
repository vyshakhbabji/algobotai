"""Tests for Portfolio module."""
from algobot.portfolio.portfolio import Portfolio


def test_open_and_close():
    p = Portfolio(initial_cash=10000)
    p.open_position("AAPL", shares=10, price=100)
    snap = p.mark_to_market({"AAPL": 110})
    assert snap['equity'] > 10000
    closed = p.close_position("AAPL", price=120)
    assert closed['pnl'] == 200
    assert round(p.cash, 2) == 10200.0
