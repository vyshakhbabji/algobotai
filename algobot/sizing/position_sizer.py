"""Position sizing strategies.

Includes a simple fractional Kelly implementation with safe caps.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AccountState:
    equity: float


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.60) -> float:
    """Compute fractional Kelly bet fraction of equity.

    Args:
        win_rate: Estimated probability of a win, in [0,1].
        avg_win: Average win (as a fraction, e.g., 0.05 for +5%).
        avg_loss: Average loss (as a fraction, e.g., 0.02 for -2%).
        fraction: Fraction of full Kelly to apply (e.g., 0.60 = 60% Kelly - AGGRESSIVE!).

    Returns:
        Kelly fraction in [0, 1] (typically small), after applying caps and guards.
    """
    try:
        if avg_win <= 0 or avg_loss <= 0 or not (0 <= win_rate <= 1):
            return 0.0
        # Full Kelly for win/loss formulation: f* = (bp - q)/b, where b = avg_win/avg_loss
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        full_kelly = (b * p - q) / max(b, 1e-9)
        # Apply fractional Kelly and clamp to [0, 1]
        f = max(0.0, min(full_kelly * max(fraction, 0.0), 1.0))
        return f
    except Exception:
        return 0.0


def kelly_size(confidence: float, equity: float, price: float, cap_fraction: float = 0.50,
               win_rate: Optional[float] = None, avg_win: float = 0.08, avg_loss: float = 0.025,
               frac: float = 0.60) -> int:
    """Return integer shares using (fractional) Kelly sizing with a per-position cap.

    Backward compatible: if win_rate is None, maps confidence -> win_rate.

    Args:
        confidence: Proxy for win probability in [0,1] when win_rate not provided.
        equity: Account equity in dollars.
        price: Current asset price.
        cap_fraction: Hard cap on position as a fraction of equity (e.g., 0.50 = 50% - AGGRESSIVE!).
        win_rate: If provided, use as win probability; otherwise use confidence.
        avg_win: Assumed average win (fraction of price) - INCREASED TO 8%.
        avg_loss: Assumed average loss (fraction of price) - INCREASED TO 2.5%.
        frac: Fraction of full Kelly to apply (e.g., 0.60 = 60% - MUCH MORE AGGRESSIVE!).

    Returns:
        Integer number of shares to buy given the cap and Kelly fraction.
    """
    wr = confidence if win_rate is None else win_rate
    f = kelly_fraction(wr, avg_win, avg_loss, fraction=frac)
    # Cap the Kelly sizing by cap_fraction to avoid oversized bets
    f_capped = min(max(f, 0.0), max(cap_fraction, 0.0))
    value = equity * f_capped
    return int(value / price) if price > 0 else 0
