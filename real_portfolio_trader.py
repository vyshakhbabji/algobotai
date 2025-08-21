#!/usr/bin/env python3
"""
REAL PORTFOLIO TRADER - Dynamic Allocation
Weekly rebalance with optional midweek deviation actions.
"""
import json
import os
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

from algobot.config import GLOBAL_CONFIG
from algobot.risk.manager import RiskManagerCore, TradeLimits


class RealPortfolioTrader:
    def __init__(self, starting_capital: float = 100000.0):
        self.starting_capital = float(starting_capital)
        # Default universe (overridden by caller)
        self.stocks = list(GLOBAL_CONFIG.universe.core_universe)[:GLOBAL_CONFIG.universe.max_universe]
        # Optimizer-derived technical config
        self.best_config = {
            'trend_5d_buy_threshold': 0.025,
            'trend_5d_sell_threshold': -0.02,
            'trend_10d_buy_threshold': 0.025,
            'trend_10d_sell_threshold': -0.045,
            'rsi_overbought': 65,
            'rsi_oversold': 20,
            'volatility_threshold': 0.07,
            'volume_ratio_threshold': 1.6,
        }
        # Risk and execution from global config
        self.min_position_size = 0.05
        self.max_position_size = float(GLOBAL_CONFIG.risk.max_position_pct)
        self.max_positions = int(getattr(GLOBAL_CONFIG.execution, 'max_positions_default', 10))
        self.cash_reserve = 1.0 - float(GLOBAL_CONFIG.risk.max_portfolio_risk_pct)
        self.rebalance_weekdays = tuple(getattr(GLOBAL_CONFIG.execution, 'rebalance_weekdays', (0,)))
        self.min_holding_days = int(getattr(GLOBAL_CONFIG.execution, 'min_holding_days', 5))
        self.allow_midweek_hard_exits = bool(getattr(GLOBAL_CONFIG.execution, 'allow_midweek_hard_exits', True))
        self.transaction_cost_bps = float(getattr(GLOBAL_CONFIG.execution, 'transaction_cost_bps', 5.0))
        self.hard_add_enabled = bool(getattr(GLOBAL_CONFIG.execution, 'hard_add_enabled', True))
        self.hard_add_thr = float(getattr(GLOBAL_CONFIG.execution, 'hard_add_buy_strength', 0.75))
        self.hard_add_chunk = float(getattr(GLOBAL_CONFIG.execution, 'hard_add_chunk_weight', 0.04))
        self.hard_trim_enabled = bool(getattr(GLOBAL_CONFIG.execution, 'hard_trim_enabled', True))
        self.hard_trim_thr = float(getattr(GLOBAL_CONFIG.execution, 'hard_trim_sell_strength', 0.55))
        self.hard_trim_frac = float(getattr(GLOBAL_CONFIG.execution, 'hard_trim_fraction', 0.5))
        # Trailing stop percent from peak to protect gains (e.g., 10%)
        self.trailing_stop_pct = 0.10
        # Dates
        self.test_end_date = datetime.now().date()
        self.test_start_date = self.test_end_date - timedelta(days=90)
        # Risk manager
        self.risk_manager = RiskManagerCore(TradeLimits(
            max_position_pct=self.max_position_size * 100,
            max_portfolio_risk_pct=float(GLOBAL_CONFIG.risk.max_portfolio_risk_pct),
            stop_loss_pct=float(GLOBAL_CONFIG.risk.stop_loss_pct),
            take_profit_pct=float(GLOBAL_CONFIG.risk.take_profit_pct),
        ))
        print("💰 REAL PORTFOLIO TRADER - Dynamic Allocation")
        print(f"🧪 Testing: {self.test_start_date} to {self.test_end_date}")
        print(f"🎯 Position cap: {self.max_position_size:.0%}, Max positions: {self.max_positions}")
        print(f"🔄 Rebalance: {self.rebalance_weekdays}, Min-hold: {self.min_holding_days}d, Cost: {self.transaction_cost_bps}bps")

    @staticmethod
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

    def _download_data(self) -> dict:
        """Load from cache when available; otherwise batch download remaining.
        Returns dict[symbol]->DataFrame with indicators.
        """
        all_data: dict[str, pd.DataFrame] = {}
        cache_dir = os.path.join(os.path.dirname(__file__), 'data_cache')
        os.makedirs(cache_dir, exist_ok=True)

        def cache_path(sym: str) -> str:
            return os.path.join(cache_dir, f"{sym}_{self.test_start_date}_{self.test_end_date}.csv")

        # 1) Try cache first
        missing: list[str] = []
        for s in self.stocks:
            fp = cache_path(s)
            if os.path.exists(fp):
                try:
                    df = pd.read_csv(fp, index_col=0, parse_dates=True)
                    if not df.empty and 'Close' in df.columns:
                        all_data[s] = self._indicators(df)
                        continue
                except Exception:
                    pass
            missing.append(s)

        # 2) Batch download any missing
        if missing:
            try:
                df = yf.download(missing, start=self.test_start_date, end=self.test_end_date,
                                 group_by='ticker', threads=True, progress=False, auto_adjust=False)
                if isinstance(df.columns, pd.MultiIndex):
                    level0 = list(map(str, df.columns.get_level_values(0).unique()))
                    level1 = list(map(str, df.columns.get_level_values(1).unique()))
                    tickers_on_lvl0 = any(t in level0 for t in missing)
                    tickers_on_lvl1 = any(t in level1 for t in missing)
                    for sym in missing:
                        try:
                            if tickers_on_lvl0 and sym in df.columns.get_level_values(0):
                                sub = df[sym].dropna(how='all')
                            elif tickers_on_lvl1 and sym in df.columns.get_level_values(1):
                                sub = df.xs(sym, axis=1, level=1).dropna(how='all')
                            else:
                                continue
                            if not sub.empty and {'Close'}.issubset(sub.columns):
                                # Save cache
                                sub.to_csv(cache_path(sym))
                                all_data[sym] = self._indicators(sub)
                        except Exception:
                            continue
                else:
                    # Unexpected shape -> fallback to per-symbol
                    raise ValueError('Non-multiindex from batch; fallback to per-symbol')
            except Exception:
                # 3) Fallback per-symbol for any still missing
                for s in missing:
                    try:
                        dfi = yf.download(s, start=self.test_start_date, end=self.test_end_date, progress=False)
                        if not dfi.empty:
                            dfi.to_csv(cache_path(s))
                            all_data[s] = self._indicators(dfi)
                    except Exception:
                        continue
        # Filter tickers with enough history (>= 40 rows for warmup)
        min_len = 40
        all_data = {k: v for k, v in all_data.items() if len(v) >= min_len}
        return all_data

    def _signal(self, d: pd.DataFrame, cfg: dict, i: int) -> dict:
        price = float(d['Close'].iloc[i])
        close = d['Close']
        ma5 = float(d['MA5'].iloc[i]) if not pd.isna(d['MA5'].iloc[i]) else price
        ma10 = float(d['MA10'].iloc[i]) if not pd.isna(d['MA10'].iloc[i]) else price
        rsi = float(d['RSI'].iloc[i]) if not pd.isna(d['RSI'].iloc[i]) else 50.0
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
        if trend_5d > cfg['trend_5d_buy_threshold'] and trend_10d > cfg['trend_10d_buy_threshold']:
            buy_strength += min(1.0, (trend_5d + trend_10d)/0.1) * 0.3
        if price > ma5 > ma10:
            buy_strength += min(1.0, (price - ma10)/max(ma10,1e-9)/0.05) * 0.2
        if rsi < cfg['rsi_oversold'] and trend_5d > cfg['trend_5d_buy_threshold']/2:
            buy_strength += ((cfg['rsi_oversold'] - rsi)/max(cfg['rsi_oversold'],1e-9)) * 0.15
        buy_strength += momentum_consistency * 0.2

        if trend_5d < cfg['trend_5d_sell_threshold'] and trend_10d < cfg['trend_10d_sell_threshold']:
            sell_strength += min(1.0, abs(trend_5d + trend_10d)/0.1) * 0.4
        if price < ma5 < ma10:
            sell_strength += min(1.0, (ma10 - price)/max(ma10,1e-9)/0.05) * 0.3
        if rsi > cfg['rsi_overbought'] and trend_5d < cfg['trend_5d_sell_threshold']/2:
            sell_strength += ((rsi - cfg['rsi_overbought'])/max(100-cfg['rsi_overbought'],1e-9)) * 0.2
        if vol10 > cfg['volatility_threshold']:
            sell_strength += min(1.0, vol10/0.2) * 0.1

        signal = 'HOLD'
        strength = 0.0
        if buy_strength > 0.4 and buy_strength > sell_strength:
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

    def calculate_position_size(self, signal_data: dict, equity: float, positions: dict) -> float:
        if signal_data.get('signal') != 'BUY':
            return 0.0
        s = float(signal_data.get('strength', 0.0))
        base = self.min_position_size + s * (self.max_position_size - self.min_position_size)
        vol_factor = max(0.5, 1 - float(signal_data.get('volatility', 0.0)))
        cons_factor = 0.7 + float(signal_data.get('momentum_consistency', 0.0)) * 0.3
        size = base * vol_factor * cons_factor
        size = max(self.min_position_size, min(self.max_position_size, size))
        current_pct_map = {sym: {'percentage': (positions[sym]['current_value']/max(equity,1e-9)*100.0)
                                  if positions.get(sym,{}).get('current_value',0)>0 else 0.0}
                           for sym in positions}
        result = self.risk_manager.validate('NEW', size*100.0, current_pct_map)
        if not result.get('approved', True):
            curr_total_pct = sum(v['percentage'] for v in current_pct_map.values())
            remaining = max(0.0, self.risk_manager.limits.max_portfolio_risk_pct*100.0 - curr_total_pct)
            cap_pct = min(self.risk_manager.limits.max_position_pct, remaining)
            return max(self.min_position_size, min(cap_pct/100.0, self.max_position_size))
        return size

    @staticmethod
    def _max_drawdown(values: pd.Series) -> tuple[float, float]:
        if values.empty:
            return 0.0, 0.0
        cummax = values.cummax()
        drawdowns = (values - cummax) / cummax
        min_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0
        amount = float((values - cummax).min()) if not values.empty else 0.0
        return min_dd * 100.0, amount

    def run_real_portfolio_simulation(self) -> dict:
        print("\n🚀 RUNNING REAL PORTFOLIO SIMULATION")
        all_data = self._download_data()
        if not all_data:
            return {}
        # Momentum-based filtering: keep top 20 by 30d return
        if len(all_data) > 20:
            scores = []
            for sym, df in all_data.items():
                try:
                    if len(df) >= 40:
                        r = float(df['Close'].iloc[-1]) / float(df['Close'].iloc[-30]) - 1.0
                        scores.append((r, sym))
                except Exception:
                    continue
            scores.sort(reverse=True)
            keep_syms = set(sym for _, sym in scores[:20])
            all_data = {k: v for k, v in all_data.items() if k in keep_syms}
        if not all_data:
            return {}
        # Dates: strict intersection to reduce loop work and ensure consistency
        common_dates = None
        for df in all_data.values():
            s = set(df.index)
            common_dates = s if common_dates is None else common_dates.intersection(s)
        dates = sorted(list(common_dates))
        if len(dates) < 40:
            return {}
        dates = dates[30:]  # warmup
        port = {
            'cash': self.starting_capital,
            'positions': {},
            'total_value': self.starting_capital,
            'daily_values': [],
            'trades': []
        }
        for idx, dt in enumerate(dates):
            if idx % 20 == 0:
                print(f"📊 {pd.Timestamp(dt).date()} ({idx}/{len(dates)})")
            # Update portfolio value
            pv = port['cash']
            prices_cache = {}
            for sym, pos in list(port['positions'].items()):
                df = all_data.get(sym)
                if df is not None and dt in df.index:
                    price = float(df.loc[dt, 'Close'])
                    # Track peak price for trailing stop
                    prev_peak = pos.get('peak_price', pos.get('avg_cost', price))
                    pos['peak_price'] = max(prev_peak, price)
                    pos['current_value'] = pos['shares'] * price
                    pv += pos['current_value']
            port['total_value'] = pv
            # Signals
            signals = {}
            for sym, df in all_data.items():
                if dt in df.index:
                    i = df.index.get_loc(dt)
                    if i >= 30:
                        sig = self._signal(df, self.best_config, i)
                        signals[sym] = sig
            is_rebalance_day = pd.Timestamp(dt).weekday() in self.rebalance_weekdays
            # Sells
            to_close = []
            for sym, pos in port['positions'].items():
                sig = signals.get(sym)
                if not sig:
                    continue
                price = sig['price']
                held_days = (pd.Timestamp(dt) - pd.Timestamp(pos.get('entry_date', dt))).days
                should_sell = False
                reason = ''
                # Trailing stop from peak
                peak = pos.get('peak_price', pos.get('avg_cost', price))
                if price <= peak * (1 - self.trailing_stop_pct):
                    should_sell = True; reason = f'Trailing stop {self.trailing_stop_pct*100:.0f}%'
                elif sig['signal'] == 'SELL' and sig['strength'] > 0.4 and (self.allow_midweek_hard_exits or is_rebalance_day) and held_days >= self.min_holding_days:
                    should_sell = True; reason = f"Strong sell {sig['strength']:.2f}"
                elif sig['sell_strength'] > 0.5 and (self.allow_midweek_hard_exits or is_rebalance_day) and held_days >= self.min_holding_days:
                    should_sell = True; reason = f"High sell pressure {sig['sell_strength']:.2f}"
                elif price < pos['avg_cost'] * 0.92:
                    should_sell = True; reason = 'Stop loss'
                # Removed fixed profit-taking; let trailing stop handle exits
                if should_sell:
                    to_close.append((sym, reason))
            for sym, reason in to_close:
                if sym not in port['positions']:
                    continue
                pos = port['positions'][sym]
                price = float(all_data[sym].loc[dt, 'Close'])
                value = pos['shares'] * price
                cost = value * (self.transaction_cost_bps/10000.0)
                port['cash'] += (value - cost)
                profit = value - (pos['shares'] * pos['avg_cost'])
                port['trades'].append({'date': dt, 'symbol': sym, 'action': 'SELL', 'shares': pos['shares'], 'price': price, 'value': value, 'profit': profit, 'reason': reason})
                del port['positions'][sym]
            # Buys
            candidates = []
            for sym, sig in signals.items():
                if sig['signal'] == 'BUY' and sym not in port['positions'] and is_rebalance_day:
                    size = self.calculate_position_size(sig, pv, port['positions'])
                    if size > 0:
                        score = sig['strength'] * max(0.0, sig['momentum_consistency'])
                        candidates.append((score, sym, sig, size))
            candidates.sort(reverse=True)
            for score, sym, sig, size in candidates:
                if len(port['positions']) >= self.max_positions:
                    break
                avail_cash = port['cash'] * (1 - self.cash_reserve)
                buy_amt = min(avail_cash, pv * size)
                if buy_amt >= self.starting_capital * self.min_position_size:
                    price = sig['price']
                    shares = buy_amt / price
                    cost = buy_amt * (self.transaction_cost_bps/10000.0)
                    port['cash'] -= (buy_amt + cost)
                    port['positions'][sym] = {'shares': shares, 'avg_cost': price, 'current_value': buy_amt, 'entry_date': pd.Timestamp(dt), 'peak_price': price}
                    port['trades'].append({'date': dt, 'symbol': sym, 'action': 'BUY', 'shares': shares, 'price': price, 'value': buy_amt, 'signal_strength': sig['strength']})
            # Snapshot
            port['daily_values'].append({'date': dt, 'total_value': port['total_value'], 'cash': port['cash'], 'positions_value': port['total_value'] - port['cash'], 'num_positions': len(port['positions'])})
            # Midweek deviation
            if not is_rebalance_day:
                if self.hard_add_enabled:
                    for sym, sig in signals.items():
                        if sig['signal'] == 'BUY' and sig['strength'] >= self.hard_add_thr and sym in port['positions']:
                            pos = port['positions'][sym]
                            held_days = (pd.Timestamp(dt) - pd.Timestamp(pos.get('entry_date', dt))).days
                            if held_days < self.min_holding_days:
                                continue
                            price = sig['price']
                            curr_val = pos['shares'] * price
                            curr_w = curr_val / max(pv, 1)
                            target_w = min(self.max_position_size, curr_w + self.hard_add_chunk)
                            curr_port_w = (port['total_value'] - port['cash']) / max(pv, 1)
                            remain = max(0.0, float(GLOBAL_CONFIG.risk.max_portfolio_risk_pct) - curr_port_w)
                            target_w = min(target_w, curr_w + remain)
                            dw = max(0.0, target_w - curr_w)
                            if dw > 0.001 and port['cash'] > 0:
                                amount = min(port['cash'], dw * pv)
                                if amount > 200:
                                    shares = amount / price
                                    cost = amount * (self.transaction_cost_bps/10000.0)
                                    port['cash'] -= (amount + cost)
                                    pos['shares'] += shares
                                    pos['avg_cost'] = (pos['avg_cost'] * curr_val + amount) / max(curr_val + amount, 1e-9)
                                    pos['current_value'] = pos['shares'] * price
                                    port['trades'].append({'date': dt, 'symbol': sym, 'action': 'ADD', 'shares': shares, 'price': price, 'value': amount, 'reason': f'strength={sig["strength"]:.2f}'})
                if self.hard_trim_enabled:
                    for sym, pos in list(port['positions'].items()):
                        sig = signals.get(sym)
                        if not sig:
                            continue
                        held_days = (pd.Timestamp(dt) - pd.Timestamp(pos.get('entry_date', dt))).days
                        if sig['sell_strength'] >= self.hard_trim_thr and held_days >= self.min_holding_days:
                            price = sig['price']
                            curr_val = pos['shares'] * price
                            amount = curr_val * max(0.05, min(0.95, self.hard_trim_frac))
                            shares = amount / price
                            proceeds = amount
                            cost = proceeds * (self.transaction_cost_bps/10000.0)
                            pos['shares'] -= shares
                            port['cash'] += (proceeds - cost)
                            pos['current_value'] = pos['shares'] * price
                            port['trades'].append({'date': dt, 'symbol': sym, 'action': 'TRIM', 'shares': shares, 'price': price, 'value': amount, 'reason': f'sell_strength={sig["sell_strength"]:.2f}'})
                            if pos['shares'] <= 1e-6:
                                del port['positions'][sym]
        # Final metrics
        final_value = port['total_value']
        total_return = (final_value - self.starting_capital)/self.starting_capital * 100.0
        dv = pd.DataFrame(port['daily_values'])
        monthly_pnl = []
        max_dd_pct = 0.0
        max_dd_amt = 0.0
        if not dv.empty:
            dv['date'] = pd.to_datetime(dv['date'])
            dv = dv.set_index('date')
            prev = self.starting_capital
            for dt, val in dv['total_value'].resample('M').last().items():
                monthly_pnl.append({'month': dt.strftime('%Y-%m'), 'pnl': float(val - prev), 'end_value': float(val)})
                prev = float(val)
            dd_pct, dd_amt = self._max_drawdown(dv['total_value'])
            max_dd_pct = float(dd_pct)
            max_dd_amt = float(dd_amt)
        return {
            'starting_capital': self.starting_capital,
            'final_value': float(final_value),
            'total_return': float(total_return),
            'num_trades': int(len(port['trades'])),
            'trades': port['trades'],
            'daily_values': port['daily_values'],
            'monthly_pnl': monthly_pnl,
            'max_drawdown_pct': max_dd_pct,
            'max_drawdown_amount': max_dd_amt,
            'final_positions': port['positions'],
            'final_cash': float(port['cash']),
        }

    def analyze_results(self, res: dict) -> dict:
        print("\n== RESULTS ==")
        print(f"Start: ${res['starting_capital']:,.2f}  Final: ${res['final_value']:,.2f}  Return: {res['total_return']:.2f}%  Trades: {res['num_trades']}")
        if 'max_drawdown_pct' in res:
            print(f"Max Drawdown: {res['max_drawdown_pct']:.2f}% (${res['max_drawdown_amount']:,.0f})")
        for m in res.get('monthly_pnl', []):
            print(f"{m['month']}: PnL ${m['pnl']:.2f}  End ${m['end_value']:.2f}")
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f"real_portfolio_results_{ts}.json", 'w') as f:
            json.dump(res, f, indent=2, default=str)
        return res


if __name__ == '__main__':
    tr = RealPortfolioTrader(100000)
    res = tr.run_real_portfolio_simulation()
    tr.analyze_results(res)
