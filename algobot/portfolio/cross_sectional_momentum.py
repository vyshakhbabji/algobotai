"""Cross-sectional momentum/quality portfolio runner (institutional-style).

Purpose:
- Take the strong-symbol scan (scan_summary.json) and construct a diversified, long-only portfolio.
- Weekly rebalance into top-K names with a probability-driven entry/exit and volatility targeting.
- Use the advanced forward simulator per symbol to generate probabilities and execute trades.

Why this helps:
- Cross-sectional momentum with risk controls is a robust, institutionally common approach that often targets 15–30% annual returns in favorable regimes with controlled drawdowns.
- This module wires your existing advanced model into a consistent, portfolio-level framework.

CLI example:
    python -m algobot.portfolio.cross_sectional_momentum \
        --scan two_year_batch/scan_summary.json \
        --topk 20 \
        --start 2024-01-01 --end 2025-08-10 \
        --out cross_sectional_results \
        --prob-buy 0.60 --prob-exit 0.50 --prob-hard-exit 0.45 \
        --vol-target 0.18 --max-weight 0.20

Outputs:
- equity_curve.csv, trades.csv, per-symbol trade charts, and metrics.json in the output directory.
"""
from __future__ import annotations
from pathlib import Path
import json
import argparse
from typing import List, Dict, Any
import pandas as pd

from algobot.forward_test.advanced_forward_sim import advanced_forward


def _load_topk_symbols(scan_path: str, topk: int, strong_only: bool = True) -> List[str]:
    with open(scan_path, 'r') as f:
        scan = json.load(f)
    strong_cut = float(scan.get('strong_cut', 0.30))
    ranked = scan.get('ranked') or []
    if strong_only:
        syms = [s for s, score in ranked if (isinstance(score, (int, float)) and score >= strong_cut)]
    else:
        syms = [s for s, _ in ranked]
    # De-duplicate while keeping order
    seen = set()
    out = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:topk]


def run_cross_sectional(scan_path: str,
                        start: str,
                        end: str,
                        topk: int = 20,
                        out_dir: str = 'cross_sectional_results',
                        prob_buy: float = 0.60,
                        prob_exit: float = 0.50,
                        prob_hard_exit: float = 0.45,
                        smoothing_window: int = 3,
                        vol_target_annual: float = 0.18,
                        min_holding_days: int = 5,
                        max_symbol_weight: float = 0.20,
                        transaction_cost_bps: float = 5,
                        rebalance_weekdays: tuple = (0,),
                        allow_midweek_hard_exits: bool = True,
                        use_regime_filter: bool = True,
                        regime_symbol: str = 'SPY',
                        regime_fast: int = 20,
                        regime_slow: int = 100,
                        gross_target: float = 1.0,
                        allow_leverage: bool = False) -> Dict[str, Any]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    symbols = _load_topk_symbols(scan_path, topk=topk, strong_only=True)
    if not symbols:
        raise SystemExit('No symbols selected from scan. Run two_year_batch_runner first to generate scan_summary.json.')
    print(f"Selected top-{len(symbols)} strong symbols: {symbols}")
    result = advanced_forward(
        symbols=symbols,
        start=start,
        end=end,
        lookback_days=252,
        retrain_interval='M',
        prob_buy=prob_buy,
        prob_exit=prob_exit,
        prob_hard_exit=prob_hard_exit,
        smoothing_window=smoothing_window,
        vol_target_annual=vol_target_annual,
        min_holding_days=min_holding_days,
        max_symbol_weight=max_symbol_weight,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_weekdays=rebalance_weekdays,
        allow_midweek_hard_exits=allow_midweek_hard_exits,
        use_regime_filter=use_regime_filter,
        regime_symbol=regime_symbol,
        regime_fast=regime_fast,
        regime_slow=regime_slow,
    out_dir=out_dir,
    chart=True,
    gross_target=gross_target,
    allow_leverage=allow_leverage,
    )
    # Persist metrics as JSON for quick review
    metrics_path = Path(out_dir) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(result.metrics, f, indent=2)
    print('Portfolio metrics:', json.dumps(result.metrics, indent=2))
    return result.metrics


def main():
    ap = argparse.ArgumentParser(description='Cross-sectional momentum portfolio runner')
    ap.add_argument('--scan', default='two_year_batch/scan_summary.json', help='Path to scan_summary.json produced by two_year_batch_runner')
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--out', default='cross_sectional_results')
    ap.add_argument('--prob-buy', type=float, default=0.60)
    ap.add_argument('--prob-exit', type=float, default=0.50)
    ap.add_argument('--prob-hard-exit', type=float, default=0.45)
    ap.add_argument('--smooth', type=int, default=3, help='Probability smoothing window')
    ap.add_argument('--vol-target', type=float, default=0.18)
    ap.add_argument('--min-hold', type=int, default=5)
    ap.add_argument('--max-weight', type=float, default=0.20)
    ap.add_argument('--cost-bps', type=float, default=5)
    ap.add_argument('--no-regime', action='store_true', help='Disable SPY regime filter')
    ap.add_argument('--gross', type=float, default=1.0, help='Gross exposure target (e.g., 1.0 = 100% invested)')
    ap.add_argument('--allow-leverage', action='store_true', help='Allow cash to go negative to hit gross target')
    args = ap.parse_args()

    run_cross_sectional(
        scan_path=args.scan,
        start=args.start,
        end=args.end,
        topk=args.topk,
        out_dir=args.out,
        prob_buy=args.prob_buy,
        prob_exit=args.prob_exit,
        prob_hard_exit=args.prob_hard_exit,
        smoothing_window=args.smooth,
        vol_target_annual=args.vol_target,
        min_holding_days=args.min_hold,
        max_symbol_weight=args.max_weight,
        transaction_cost_bps=args.cost_bps,
    use_regime_filter=not args.no_regime,
    gross_target=args.gross,
    allow_leverage=args.allow_leverage,
    )


if __name__ == '__main__':
    main()
