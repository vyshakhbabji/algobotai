"""Tests for risk manager core."""
from algobot.risk.manager import RiskManagerCore, TradeLimits


def test_risk_approval():
    rm = RiskManagerCore(TradeLimits(max_position_pct=20.0, max_portfolio_risk_pct=0.30))
    result = rm.validate("AAPL", position_pct=10.0, current_positions={})
    assert result['approved']


def test_risk_reject_large():
    rm = RiskManagerCore(TradeLimits(max_position_pct=5.0))
    result = rm.validate("AAPL", position_pct=10.0, current_positions={})
    assert not result['approved']
