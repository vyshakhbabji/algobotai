#!/usr/bin/env python3
"""
Paper Trade Runner (Alpaca)

Runs a one-shot paper-trading rebalance based on current signals using
the RealPortfolioTrader-style signal logic and submits market orders via Alpaca.

Usage examples:
  python -m algobot.live.paper_trade_runner --account 45000 --dry-run
  python -m algobot.live.paper_trade_runner --account 100000 --execute

Environment or config:
  - Prefer env vars: ALPACA_API_KEY, ALPACA_SECRET_KEY
  - Or provide alpaca_config.json at workspace root with fields:
      {"alpaca": {"api_key": "...", "secret_key": "...", "paper_trading": true}}
"""
from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

from algobot.config import GLOBAL_CONFIG
from algobot.broker.base import Order

# Broker is optional until execute; allow dry-run without alpaca-py
try:
    from algobot.broker.alpaca import AlpacaBroker
    _HAS_ALPACA = True
except Exception:
    AlpacaBroker = None  # type: ignore
    _HAS_ALPACA = False


@dataclass
class LivePosition:
    symbol: str
    qty: float
    avg_entry: float
    market_price: float
    value: float


def load_alpaca_creds() -> Optional[Dict[str, str]]:
    api_key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if api_key and secret:
        return {"api_key": api_key, "secret_key": secret, "paper": True}
    # fallback to json file
    for fname in ("alpaca_config.json", "config/alpaca_config.json"):
        if os.path.exists(fname):
            try:
                with open(fname, "r") as f:
                    data = json.load(f)
                alp = data.get("alpaca") or data
                api_key = alp.get("api_key")
                secret = alp.get("secret_key")
                paper = bool(alp.get("paper_trading", True))
                if api_key and secret:
                    return {"api_key": api_key, "secret_key": secret, "paper": paper}
            except Exception:
                pass
    return None


def download_recent(symbols: List[str], lookback_days: int = 120) -> Dict[str, pd.DataFrame]:
    end = datetime.now().date()
    start = (pd.Timestamp(end) - pd.Timedelta(days=lookback_days)).date()
    out: Dict[str, pd.DataFrame] = {}
    try:
        df = yf.download(symbols, start=start, end=end, group_by='ticker', threads=True, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            for sym in symbols:
                try:
                    if sym in df.columns.get_level_values(0):
                        sub = df[sym].dropna(how='all')
                    else:
                        sub = df.xs(sym, axis=1, level=1).dropna(how='all')
                    if not sub.empty and 'Close' in sub.columns:
                        out[sym] = _indicators(sub)
                except Exception:
                    continue
        else:
            # single symbol case
            if not df.empty and 'Close' in df.columns:
                out[symbols[0]] = _indicators(df)
    except Exception:
        # fallback per-symbol
        for s in symbols:
            try:
                dfi = yf.download(s, start=start, end=end, progress=False)
                if not dfi.empty:
                    out[s] = _indicators(dfi)
            except Exception:
                continue
    # keep only with warmup
    out = {k: v for k, v in out.items() if len(v) >= 40}
    return out


def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['MA5'] = d['Close'].rolling(5).mean()
    d['MA10'] = d['Close'].rolling(10).mean()
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    d['RSI'] = 100 - (100 / (1 + rs))
    return d


def _signal(d: pd.DataFrame, i: int) -> dict:
    price = float(d['Close'].iloc[i])
    close = d['Close']
    ma5 = float(d['MA5'].iloc[i]) if not pd.isna(d['MA5'].iloc[i]) else price
    ma10 = float(d['MA10'].iloc[i]) if not pd.isna(d['MA10'].iloc[i]) else price
    r5 = close.iloc[i-5:i]
    r10 = close.iloc[i-10:i]
    r20 = close.iloc[i-20:i]
    trend_5d = (price - float(r5.mean()))/max(float(r5.mean()), 1e-9)
    trend_10d = (price - float(r10.mean()))/max(float(r10.mean()), 1e-9)
    trend_20d = (price - float(r20.mean()))/max(float(r20.mean()), 1e-9)
    vol10 = float(np.std(r10))/max(float(np.mean(r10)), 1e-9)
    momentum_consistency = np.mean([trend_5d>0, trend_10d>0, trend_20d>0])

    buy_strength = 0.0
    sell_strength = 0.0
    if trend_5d > 0.025 and trend_10d > 0.025:
        buy_strength += min(1.0, (trend_5d + trend_10d)/0.1) * 0.3
    if price > ma5 > ma10:
        buy_strength += min(1.0, (price - ma10)/max(ma10,1e-9)/0.05) * 0.2
    if trend_5d > 0.0125 and float(d['RSI'].iloc[i]) < 20:
        buy_strength += (20 - float(d['RSI'].iloc[i]))/20 * 0.15
    buy_strength += momentum_consistency * 0.2

    if trend_5d < -0.02 and trend_10d < -0.045:
        sell_strength += min(1.0, abs(trend_5d + trend_10d)/0.1) * 0.4
    if price < ma5 < ma10:
        sell_strength += min(1.0, (ma10 - price)/max(ma10,1e-9)/0.05) * 0.3
    if float(d['RSI'].iloc[i]) > 65 and trend_5d < -0.01:
        sell_strength += (float(d['RSI'].iloc[i]) - 65)/35 * 0.2
    if vol10 > 0.07:
        sell_strength += min(1.0, vol10/0.2) * 0.1

    signal = 'HOLD'
    strength = 0.0
    if buy_strength > 0.3 and buy_strength > sell_strength:  # Lowered from 0.4 to 0.3 for more buy signals
        signal = 'BUY'; strength = min(1.0, buy_strength)
    elif sell_strength > 0.3 and sell_strength > buy_strength:
        signal = 'SELL'; strength = min(1.0, sell_strength)
    return {
        'signal': signal,
        'strength': float(strength),
        'buy_strength': float(buy_strength),
        'sell_strength': float(sell_strength),
        'momentum_consistency': float(momentum_consistency),
        'volatility': float(vol10),
        'price': float(price),
    }


def main():
    parser = argparse.ArgumentParser(description="Alpaca paper-trade runner")
    parser.add_argument('--account', type=float, default=100000.0, help='Account size for sizing logic')
    parser.add_argument('--execute', action='store_true', help='If set, place orders on Alpaca')
    parser.add_argument('--dry-run', action='store_true', help='Print intended actions only')
    parser.add_argument('--max-buys', type=int, default=3, help='Max new buys to open this run')
    parser.add_argument('--max-sells', type=int, default=99, help='Max sells to execute this run')
    parser.add_argument('--buy-only', action='store_true', help='Only place buy orders')
    parser.add_argument('--sell-only', action='store_true', help='Only place sell/trim orders')
    parser.add_argument('--force', action='store_true', help='Bypass market-hours check')
    parser.add_argument('--market-hours-only', dest='market_hours_only', action='store_true', help='If set, run only during US market hours')
    parser.add_argument('--universe', type=str, nargs='*', help='Optional override ticker list')
    args = parser.parse_args()

    # Market hours safety (US/Eastern 09:30-16:00, Mon-Fri)
    if args.market_hours_only and not args.force:
        now_et = datetime.now(tz=ZoneInfo('US/Eastern'))
        is_weekday = now_et.weekday() < 5
        in_hours = time(9, 30) <= now_et.time() <= time(16, 0)
        if not (is_weekday and in_hours):
            print(f"Market-hours-only active. Now ET: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')} (not in trading window). Use --force to override.")
            return 0

    # Universe
    symbols: List[str]
    if args.universe:
        symbols = [s.upper() for s in args.universe]
    else:
        symbols = list(GLOBAL_CONFIG.universe.core_universe)[:GLOBAL_CONFIG.universe.max_universe]

    # Download recent data and compute today signals
    data = download_recent(symbols)
    if not data:
        print("No data available for universe; aborting.")
        return 2
    # Common latest date
    common = None
    for df in data.values():
        idx = set(df.index)
        common = idx if common is None else common.intersection(idx)
    dates = sorted(list(common))
    if len(dates) < 30:
        print("Insufficient overlap dates; aborting.")
        return 2
    today = dates[-1]

    # Build signals
    signals: Dict[str, dict] = {}
    for sym, df in data.items():
        i = df.index.get_loc(today)
        if i >= 30:
            signals[sym] = _signal(df, i)

    # Broker/client state
    creds = load_alpaca_creds()
    # Broker instance (set only when executing with valid credentials)
    broker = None
    if args.execute:
        if creds and _HAS_ALPACA:
            broker = AlpacaBroker(api_key=creds['api_key'], secret_key=creds['secret_key'], paper=creds.get('paper', True))
            acct = broker.get_account()
            equity = float(acct.get('equity', args.account))
            cash = float(acct.get('cash', 0.0))
            buying_power = float(acct.get('buying_power', cash))
        else:
            print("No Alpaca broker available; cannot execute. Set ALPACA_API_KEY/SECRET or alpaca_config.json")
            return 2
    else:
        equity = args.account
        buying_power = args.account * float(GLOBAL_CONFIG.risk.max_portfolio_risk_pct)

    # Current positions map
    positions: Dict[str, LivePosition] = {}
    if broker:
        for p in broker.get_positions():
            sym = p['symbol']
            qty = float(p['qty'])
            price = float(p.get('current_price') or 0.0)
            value = float(p.get('market_value') or (qty * price))
            positions[sym] = LivePosition(symbol=sym, qty=qty, avg_entry=float(p.get('avg_entry_price', price)), market_price=price, value=value)

    # Decide sells
    sells: List[Order] = []
    for sym, pos in positions.items():
        sig = signals.get(sym)
        if not sig:
            continue
        price = sig['price']
        # stop loss vs avg entry
        if price < pos.avg_entry * (1 - float(GLOBAL_CONFIG.risk.stop_loss_pct)):
            sells.append(Order(symbol=sym, qty=pos.qty, side='sell'))
            continue
        if sig['signal'] == 'SELL' and sig['strength'] >= 0.5:
            sells.append(Order(symbol=sym, qty=pos.qty, side='sell'))
        elif sig['sell_strength'] >= float(getattr(GLOBAL_CONFIG.execution, 'hard_trim_sell_strength', 0.55)):
            trim_qty = max(0.0, pos.qty * float(getattr(GLOBAL_CONFIG.execution, 'hard_trim_fraction', 0.5)))
            if trim_qty > 0:
                sells.append(Order(symbol=sym, qty=trim_qty, side='sell'))

    # Apply sell-only/buy-only filters and caps
    if args.sell_only:
        pass
    if args.buy_only:
        sells = []
    # Cap sells
    if len(sells) > args.max_sells:
        sells = sells[:args.max_sells]

    # Decide buys
    max_positions = int(getattr(GLOBAL_CONFIG.execution, 'max_positions_default', 10))
    held = set(positions.keys())
    open_slots = max(0, max_positions - len(held))
    can_buy_n = min(open_slots, args.max_buys)
    # rank by signal strength * momentum consistency
    cands = []
    for sym, sig in signals.items():
        if sym in held:
            continue
        if sig['signal'] == 'BUY' and sig['strength'] >= 0.4:  # Lowered from 0.6 to 0.4 for more aggressive trading
            score = sig['strength'] * max(0.0, sig['momentum_consistency'])
            cands.append((score, sym, sig))
    cands.sort(reverse=True)

    buys: List[Order] = []
    portfolio_positions_map = {s: {'current_value': positions[s].value} for s in positions}
    # sizing params
    min_position_size = 0.05
    max_position_size = float(GLOBAL_CONFIG.risk.max_position_pct)
    risk_cap = float(GLOBAL_CONFIG.risk.max_portfolio_risk_pct)

    def calculate_size(sig: dict) -> float:
        s = float(sig.get('strength', 0.0))
        base = min_position_size + s * (max_position_size - min_position_size)
        vol_factor = max(0.5, 1 - float(sig.get('volatility', 0.0)))
        cons_factor = 0.7 + float(sig.get('momentum_consistency', 0.0)) * 0.3
        size = base * vol_factor * cons_factor
        return max(min_position_size, min(max_position_size, size))

    current_portfolio_value = equity
    current_exposure_value = sum(v['current_value'] for v in portfolio_positions_map.values()) if positions else 0.0
    current_exposure_pct = current_exposure_value / max(current_portfolio_value, 1e-9)
    remaining_exposure_value = max(0.0, risk_cap - current_exposure_pct) * current_portfolio_value
    budget = min(buying_power, remaining_exposure_value)

    for _, sym, sig in cands[:can_buy_n]:
        if budget <= 100:
            break
        w = calculate_size(sig)
        target_amt = current_portfolio_value * w
        buy_amt = min(target_amt, budget)
        price = sig['price']
        qty = int(buy_amt // price)
        if qty <= 0:
            continue
        buys.append(Order(symbol=sym, qty=float(qty), side='buy'))
        spend = qty * price
        budget -= spend

    # Summary
    print("===== PAPER TRADE PLAN =====")
    print(f"Date: {pd.Timestamp(today).date()}  Universe: {len(symbols)}  Held: {len(positions)}  Cash budget: ${budget:,.2f}")
    for o in sells:
        print(f"SELL {o.symbol} x{o.qty}")
    for o in buys if not (args.sell_only) else []:
        print(f"BUY  {o.symbol} x{o.qty}")

    if args.dry_run and not args.execute:
        print("-- dry-run only, no orders submitted --")
        # Log plan
        _log_trade_plan(today, sells, buys if not args.sell_only else [], executed=False, meta={
            'account_base': args.account,
            'universe_size': len(symbols),
            'max_buys': args.max_buys,
            'max_sells': args.max_sells,
            'buy_only': args.buy_only,
            'sell_only': args.sell_only,
            'market_hours_only': args.market_hours_only,
        })
        return 0

    if not broker:
        print("No Alpaca broker available; cannot execute. Set ALPACA_API_KEY/SECRET or alpaca_config.json")
        return 2

    # Execute orders
    placed = []
    execution_list = list(sells) + (list(buys) if not args.sell_only else [])
    for o in execution_list:
        try:
            res = broker.submit_order(o)
            placed.append((o.symbol, o.side, o.qty, getattr(res, 'id', None)))
        except Exception as e:
            print(f"Order failed for {o.symbol}: {e}")
    print(f"Submitted {len(placed)} orders via Alpaca (paper).")
    _log_trade_plan(today, sells, buys if not args.sell_only else [], executed=True, meta={
        'placed': placed,
        'account_equity': equity,
        'buying_power': buying_power,
        'universe_size': len(symbols),
    })
    return 0


def _log_trade_plan(today, sells: List[Order], buys: List[Order], executed: bool, meta: dict):
    try:
        os.makedirs('trade_logs', exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        rec = {
            'timestamp': ts,
            'trade_date': str(pd.Timestamp(today).date()),
            'executed': executed,
            'sells': [{'symbol': o.symbol, 'qty': o.qty, 'side': o.side} for o in sells],
            'buys': [{'symbol': o.symbol, 'qty': o.qty, 'side': o.side} for o in buys],
            'meta': meta,
        }
        with open(f'trade_logs/paper_trade_{ts}.json', 'w') as f:
            json.dump(rec, f, indent=2)
    except Exception:
        pass


if __name__ == '__main__':
    raise SystemExit(main())
