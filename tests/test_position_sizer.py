"""Tests for position sizing."""
from algobot.sizing.position_sizer import kelly_size


def test_kelly_basic():
    shares = kelly_size(confidence=0.6, equity=100000, price=100)
    assert 0 <= shares <= 150  # 15% cap -> $15k / 100 = 150 shares
