"""Portfolio strength scanner.

Features:
1. Downloads OHLCV (Close) for a list of symbols (yfinance).
2. Computes strength_score consistent with institutional strategy classification.
3. Classifies each symbol into strong / default / skip.
4. Supports fixed thresholds OR adaptive percentile-based thresholds.
5. Optional concurrent downloads for speed.
6. Writes both a detailed universe_strength.json and a lightweight classification_map.json.

Adaptive Thresholds:
    Set thresholds='auto' (or None) to derive cuts via percentiles:
        strong_pct (default 0.80) => score at this percentile becomes strong_cut.
        skip_pct   (default 0.20) => score at this percentile becomes skip_cut.
    After computing all raw scores, percentile cuts are calculated, then classification applied.

Programmatic Example:
    from algobot.portfolio.portfolio_scanner import scan_universe
    payload = scan_universe(['NVDA','AAPL','MSFT'], thresholds='auto')
    classification_map = {s: d['classification'] for s, d in payload['results'].items() if 'classification' in d}

CLI:
    python -m algobot.portfolio.portfolio_scanner NVDA AAPL MSFT --auto

"""
from __future__ import annotations
import pandas as pd, numpy as np, json, time, os
from pathlib import Path
import yfinance as yf
from typing import Iterable, Dict, Any, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse classification thresholds consistent with strategy defaults
DEFAULT_THRESHOLDS: Tuple[float, float] = (0.30, -0.05)  # (strong_cut, skip_cut)


def _load_local_csv(symbol: str) -> pd.Series | None:
    """Attempt to load local pre-cleaned CSV (clean_data_SYMBOL.csv)."""
    fname_variants = [f"clean_data_{symbol.upper()}.csv", f"clean_data_{symbol.lower()}.csv"]
    for fn in fname_variants:
        if os.path.exists(fn):
            try:
                raw = pd.read_csv(fn)
                # Expect first column to hold date-like strings after skipping metadata rows
                # Remove rows where first column is 'Ticker' or 'Date'
                col0 = raw.columns[0]
                df = raw[~raw[col0].isin(['Ticker','Date'])].copy()
                # Rename first column to Date if needed
                if col0.lower() != 'date':
                    df.rename(columns={col0: 'Date'}, inplace=True)
                # Parse dates
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date']).set_index('Date').sort_index()
                # Pick Close column
                close_col_candidates = [c for c in df.columns if c.lower() == 'close']
                if not close_col_candidates:
                    return None
                close_col = close_col_candidates[0]
                close = pd.to_numeric(df[close_col], errors='coerce').dropna()
                return close
            except Exception:
                continue
    return None

def _download(symbol: str, start: str, end: str, use_local: bool) -> pd.Series:
    if use_local:
        local = _load_local_csv(symbol)
        if local is not None and not local.empty:
            # Trim to requested date span if inside range
            return local.loc[(local.index >= start) & (local.index <= end)] if isinstance(start,str) else local
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    return df['Close']


def _strength_score(close: pd.Series) -> float:
    if len(close) < 120:
        return -1.0
    def _h_ret(days):
        if len(close) <= days: return np.nan
        return close.iloc[-1]/close.iloc[-days]-1
    r252=_h_ret(252); r126=_h_ret(126); r63=_h_ret(63)
    daily=close.pct_change().dropna()
    sharpe=(daily.mean()/(daily.std()+1e-9))*np.sqrt(252) if not daily.empty else 0.0
    dd=(close/close.cummax()-1).min()
    def scalarize(v):
        # Convert potential Series/array to scalar float
        try:
            if isinstance(v, (list, tuple, np.ndarray)):
                if len(v)==0: return np.nan
                return float(v[-1])
            if hasattr(v, 'iloc') and not isinstance(v, (float,int,np.floating)):
                # pandas object (Series/Index)
                if getattr(v, 'ndim', 1) == 0:
                    return float(v)
                if len(v) == 1:
                    try:
                        return float(v.iloc[0])
                    except Exception:
                        return float(v.values[0])
                return float(v.iloc[-1])
            return float(v)
        except Exception:
            return np.nan
    r252s = scalarize(r252)
    r126s = scalarize(r126)
    r63s  = scalarize(r63)
    def safe(v):
        try:
            if v is None: return 0.0
            if isinstance(v, (float,int,np.floating)):
                if np.isnan(v): return 0.0
                return float(v)
            return 0.0 if pd.isna(v) else float(v)
        except Exception:
            return 0.0
    r252s = safe(r252s); r126s = safe(r126s); r63s = safe(r63s)
    score = 0.35*r252s + 0.25*r126s + 0.10*r63s + 0.20*sharpe + 0.10*float(dd)
    return float(score)


def classify(score: float, thresholds: Tuple[float, float]) -> str:
    strong_cut, skip_cut = thresholds
    if score >= strong_cut:
        return 'strong'
    if score < skip_cut:
        return 'skip'
    return 'default'


def _derive_adaptive_thresholds(scores: List[float], strong_pct: float, skip_pct: float) -> Tuple[float, float]:
    arr = np.array([s for s in scores if np.isfinite(s)])
    if len(arr) == 0:
        return DEFAULT_THRESHOLDS
    strong_cut = float(np.nanpercentile(arr, strong_pct * 100))
    skip_cut = float(np.nanpercentile(arr, skip_pct * 100))
    # Ensure ordering (strong_cut should be >= skip_cut; if not, widen spread)
    if strong_cut < skip_cut:
        strong_cut, skip_cut = max(strong_cut, skip_cut), min(strong_cut, skip_cut)
    return strong_cut, skip_cut


def scan_universe(symbols: Iterable[str], start='2024-01-01', end=None,
                  thresholds: Union[Tuple[float, float], str, None] = DEFAULT_THRESHOLDS,
                  strong_pct: float = 0.80, skip_pct: float = 0.20,
                  write_json: bool = True, out_path: str = 'universe_strength.json',
                  classification_map_path: str = 'classification_map.json',
                  max_workers: int = 8,
                  use_local: bool = True) -> Dict[str, Any]:
    """Scan universe and classify symbols.

    thresholds: (strong_cut, skip_cut) OR 'auto'/None for adaptive percentiles.
    strong_pct / skip_pct: percentiles used when thresholds='auto'.
    max_workers: concurrent download threads.
    """
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    symbols = list(symbols)
    t0 = time.time()
    download_results: Dict[str, Any] = {}

    def _task(sym):
        close = _download(sym, start, end, use_local=use_local)
        score = _strength_score(close)
        return sym, close, score

    # Concurrent downloads
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_task, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                sym, close, score = fut.result()
                download_results[sym] = {'close': close, 'score': score}
            except Exception as e:
                download_results[sym] = {'error': str(e)}

    # Adaptive thresholds if requested
    if thresholds in ('auto', None):
        score_list = [v['score'] for v in download_results.values() if 'score' in v]
        derived = _derive_adaptive_thresholds(score_list, strong_pct, skip_pct)
        thresholds_effective = derived
        mode = 'adaptive'
    else:
        thresholds_effective = thresholds  # type: ignore
        mode = 'fixed'

    results = {}
    for sym, data in download_results.items():
        if 'error' in data:
            results[sym] = {'error': data['error']}
            continue
        close = data['close']
        score = data['score']
        try:
            classification = classify(score, thresholds_effective)  # type: ignore
        except Exception:
            classification = 'default'
        results[sym] = {
            'strength_score': float(score) if isinstance(score,(int,float,np.floating)) else score,
            'classification': classification,
            'last_price': float(close.iloc[-1]) if len(close) else None,
            'history_days': int(len(close))
        }

    # Build classification map for direct use in strategy
    class_map = {s: r['classification'] for s, r in results.items() if 'classification' in r}

    results_meta = {
        'count': len(symbols),
        'runtime_sec': round(time.time() - t0, 2),
        'thresholds_mode': mode,
        'thresholds': tuple(thresholds_effective),  # type: ignore
        'strong_pct': strong_pct,
        'skip_pct': skip_pct
    }
    payload = {'results': results, 'classification_map': class_map, 'meta': results_meta}
    if write_json:
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        with open(classification_map_path, 'w') as f:
            json.dump(class_map, f, indent=2)
    return payload

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Portfolio strength scanner')
    parser.add_argument('symbols', nargs='*', help='Symbols to scan')
    parser.add_argument('--start', default='2024-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--auto', action='store_true', help='Use adaptive percentile thresholds')
    parser.add_argument('--strong-pct', type=float, default=0.80)
    parser.add_argument('--skip-pct', type=float, default=0.20)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', default='universe_strength.json')
    parser.add_argument('--class-map', default='classification_map.json')
    args = parser.parse_args()
    default_syms = ['NVDA','AAPL','MSFT','TSLA','META']
    syms = args.symbols or default_syms
    thresholds = 'auto' if args.auto else DEFAULT_THRESHOLDS
    res = scan_universe(syms, start=args.start, end=args.end, thresholds=thresholds,
                        strong_pct=args.strong_pct, skip_pct=args.skip_pct,
                        out_path=args.out, classification_map_path=args.class_map,
                        max_workers=args.workers)
    print(json.dumps(res['meta'], indent=2))
    # Print quick distribution summary
    scores = [r['strength_score'] for r in res['results'].values() if 'strength_score' in r]
    if scores:
        print('Score stats: min={:.3f} p20={:.3f} median={:.3f} p80={:.3f} max={:.3f}'.format(
            np.min(scores), np.percentile(scores,20), np.median(scores), np.percentile(scores,80), np.max(scores)))
    print('Class counts:', {c: list(res['classification_map'].values()).count(c) for c in set(res['classification_map'].values())})
