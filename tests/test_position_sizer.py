"""Tests for position sizing."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from algobot.sizing.position_sizer import kelly_size
from realistic_live_trading_system import RealisticLiveTradingSystem


def test_kelly_basic():
    shares = kelly_size(confidence=0.6, equity=100000, price=100)
    assert 0 <= shares <= 150  # 15% cap -> $15k / 100 = 150 shares


def test_kelly_integration():
    system = RealisticLiveTradingSystem(initial_capital=100000)
    prices_today = {'AAPL': 100}
    signal = {'signal': 'BUY', 'strength': 0.6, 'price': 100}
    decision = system.make_human_like_decision('AAPL', signal, prices_today)
    expected = kelly_size(
        confidence=0.6,
        equity=system.current_capital,
        price=100,
        cap_fraction=system.config['max_position_size'],
    )
    assert decision['shares'] == expected
    assert decision['action'] == 'BUY'
