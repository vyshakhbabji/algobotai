"""Batch universe runner for institutional strategy across a large symbol list (e.g. S&P 100).

Workflow:
1. Define or load a universe (default: sp100 static list below).
2. Scan universe for strength scores & classifications (adaptive thresholds).
3. Select subset (top K by strength and/or strong class) for full strategy runs.
4. Execute strategy sequentially or in limited concurrency (simple thread pool) with chart disabled.
5. Persist results summary JSON + per-symbol result JSON files (optional).

CLI Examples:
python -m algobot.portfolio.batch_universe_runner --topk 25 --strong-only
python -m algobot.portfolio.batch_universe_runner --topk 40 --strong-only --strong-pct 0.75 --skip-pct 0.25
python -m algobot.portfolio.batch_universe_runner --universe-file my_universe.txt --topk 50

Outputs:
- batch_scan_results.json : raw scan (scores, classifications, thresholds meta)
- batch_runs_summary.json : aggregated performance metrics for run subset
- per-symbol JSON: symbol_lower_result.json (if --save-individual)

Note: Running full model backtests for 100 symbols can be time-consuming (several minutes) due to per-symbol ML retraining.
Consider lowering lookback or simplifying model if runtime becomes excessive.
"""
from __future__ import annotations
import json, time, argparse, sys, os
from pathlib import Path
from typing import List, Dict, Any

from algobot.portfolio.portfolio_scanner import scan_universe
from algobot.forward_test.nvda_institutional_strategy import run_nvda_institutional

# Static S&P 100 style universe (tickers only; factual, non-copyrightable)
SP100 = [
    'AAPL','ABBV','ABT','ACN','ADBE','AIG','AMD','AMGN','AMT','AMZN','APD','AVGO','AXP','BA','BAC','BK','BKNG','BLK','BMY','C','CAT','CHTR','CL','CMCSA','COF','COP','COST','CRM','CSCO','CVS','CVX','DD','DE','DHR','DIS','DOW','DUK','EMR','EXC','F','FDX','GD','GE','GILD','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KHC','KO','LIN','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','META','MMM','MO','MRK','MS','MSFT','NEE','NFLX','NKE','NVDA','ORCL','PEP','PFE','PG','PM','PYPL','QCOM','RTX','SBUX','SLB','SO','SPG','T','TGT','TMO','TMUS','TSLA','TXN','UNH','UNP','UPS','USB','V','VZ','WFC','WMT','XOM'
]


def load_universe(args) -> List[str]:
    if args.universe_file:
        with open(args.universe_file) as f:
            syms = [l.strip().upper() for l in f if l.strip() and not l.startswith('#')]
            return syms
    if args.universe == 'sp100':
        return SP100
    # fallback
    return SP100


def main():
    p = argparse.ArgumentParser(description='Batch institutional strategy runner')
    p.add_argument('--universe', default='sp100', help='Named universe (sp100)')
    p.add_argument('--universe-file', help='Custom universe file (one ticker per line)')
    p.add_argument('--start', default='2024-01-01')
    p.add_argument('--end', default=None)
    p.add_argument('--strong-pct', type=float, default=0.80)
    p.add_argument('--skip-pct', type=float, default=0.20)
    p.add_argument('--topk', type=int, default=25, help='Top K by strength score to run')
    p.add_argument('--strong-only', action='store_true', help='Restrict to symbols classified strong before topK slice')
    p.add_argument('--out-dir', default='batch_results')
    p.add_argument('--save-individual', action='store_true')
    p.add_argument('--min-score', type=float, default=None, help='Optional minimum strength score filter')
    p.add_argument('--max-symbols', type=int, default=None, help='Optional hard cap on number of symbols scanned')
    p.add_argument('--dry-run', action='store_true', help='Only scan & select, skip full strategy runs')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    universe = load_universe(args)
    if args.max_symbols:
        universe = universe[:args.max_symbols]

    print(f"Scanning universe size={len(universe)} strong_pct={args.strong_pct} skip_pct={args.skip_pct} ...")
    scan_payload = scan_universe(universe, start=args.start, end=args.end, thresholds='auto',
                                 strong_pct=args.strong_pct, skip_pct=args.skip_pct,
                                 write_json=False, use_local=False)
    # Save raw scan
    with open(out_dir/'batch_scan_results.json','w') as f:
        json.dump(scan_payload, f, indent=2)

    results = scan_payload['results']
    # Build scored list
    scored = [(s, d.get('strength_score')) for s,d in results.items() if 'strength_score' in d]
    # Optional min score filter
    if args.min_score is not None:
        scored = [t for t in scored if t[1] is not None and t[1] >= args.min_score]

    # Filter strong-only if requested
    if args.strong_only:
        strong_set = {s for s,d in results.items() if d.get('classification') == 'strong'}
        scored = [t for t in scored if t[0] in strong_set]

    # Sort descending by score
    scored.sort(key=lambda x: x[1], reverse=True)

    selected = [s for s,_ in scored[:args.topk]]
    classification_map = scan_payload['classification_map']

    print(f"Selected {len(selected)} symbols for full runs: {selected}")
    if args.dry_run:
        summary = {'selected': selected, 'scored': scored[:args.topk], 'meta': scan_payload['meta']}
        with open(out_dir/'batch_runs_summary.json','w') as f:
            json.dump(summary, f, indent=2)
        print('Dry run complete (no strategy executions).')
        return

    run_outputs: Dict[str, Any] = {}
    t0 = time.time()
    for i,sym in enumerate(selected,1):
        print(f"[{i}/{len(selected)}] Running strategy for {sym} ...")
        try:
            res = run_nvda_institutional(symbol=sym, chart=False, classification_map=classification_map, symbol_classification='auto')
            run_outputs[sym] = {
                'return_pct': res.get('total_return_pct'),
                'capture': res.get('captured_buy_hold_ratio'),
                'classification': res.get('classification'),
                'buy_hold_return_pct': res.get('buy_hold_return_pct'),
                'final_equity': res.get('final_equity')
            }
            if args.save_individual:
                with open(out_dir/f"{sym.lower()}_result.json",'w') as f:
                    json.dump(res, f, indent=2)
        except Exception as e:
            run_outputs[sym] = {'error': str(e)}
        # Optional pacing if hitting API limits could sleep small
        # time.sleep(0.2)
    runtime = round(time.time()-t0,2)

    summary = {
        'selected': selected,
        'scored': scored[:args.topk],
        'meta': scan_payload['meta'],
        'run_outputs': run_outputs,
        'total_runtime_sec': runtime
    }
    with open(out_dir/'batch_runs_summary.json','w') as f:
        json.dump(summary, f, indent=2)
    print(f"Batch complete in {runtime}s. Summary written to {out_dir/'batch_runs_summary.json'}")

if __name__ == '__main__':
    main()
