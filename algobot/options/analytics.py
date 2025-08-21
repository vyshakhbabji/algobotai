"""Options analytics utilities (initial skeleton).

Includes:
- Simplified Black-Scholes IV approximation (Newton iterations) for calls.
- Strategy payoff simulation for basic multi-leg structures.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict

from math import log, sqrt, exp
from statistics import NormalDist


_normal = NormalDist()

def _bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S-K,0)
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*_normal.cdf(d1) - K*exp(-r*T)*_normal.cdf(d2)


def implied_vol_call(S, K, T, r, price, initial=0.3, tol=1e-5, max_iter=50):
    sigma = initial
    for _ in range(max_iter):
        c = _bs_call_price(S,K,T,r,sigma)
        if abs(c-price) < tol:
            return sigma
        # vega
        if T <= 0:
            break
        d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        vega = S * math.sqrt(T) * math.exp(-0.5*d1*d1) / math.sqrt(2*math.pi)
        if vega < 1e-8:
            break
        sigma -= (c-price)/vega
        if sigma <= 0:
            sigma = tol
    return sigma

@dataclass
class Leg:
    kind: str  # 'call' or 'put'
    direction: str  # 'long' or 'short'
    strike: float
    premium: float  # paid (positive for long)


def payoff(legs: List[Leg], S: float) -> float:
    total = 0.0
    for leg in legs:
        if leg.kind == 'call':
            intrinsic = max(S - leg.strike, 0)
        else:
            intrinsic = max(leg.strike - S, 0)
        pnl = intrinsic - leg.premium
        if leg.direction == 'short':
            pnl = -pnl
        total += pnl
    return total

__all__ = ["implied_vol_call","Leg","payoff"]
