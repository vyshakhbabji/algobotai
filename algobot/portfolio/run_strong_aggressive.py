from pathlib import Path
import json, datetime
import pandas as pd
import yfinance as yf
from algobot.forward_test.nvda_institutional_strategy import run_nvda_institutional

"""Aggressive re-run of strong symbols with exposure-focused parameter tweaks.
Chosen adjustments ("best options"):
- Increase early_core_fraction to 0.85 for higher baseline exposure.
- Relax tier entry: lower static tier thresholds (0.48,0.54,0.60,0.66) and use static mode (avoids high quantile drift).
- Wider initial/trailing stops (atr_initial_mult=2.0, atr_trail_mult=2.6) to reduce premature stop-outs.
- Lift performance guard constraints: shorter performance_guard_days=15, min_capture=0.05, max_fraction=0.5.
- Disable expectancy guard (enable_expectancy_guard=False) to allow scaling during limited forward window.
- Raise risk profile: risk_per_trade_pct=0.05, risk_ceiling=0.08, risk_increment=0.01.
- Delay profit taking: profit_ladder=(0.20,0.40,0.60) with smaller trim fractions (0.10,0.15,0.20) to preserve core.
- Slightly higher fast_scale_gain_threshold to avoid over-aggressive early adds (0.10).
- Keep pullback re-entry enabled.

Outputs:
 strong_aggressive/strong_aggressive_results.(csv|json)
 strong_aggressive/top5_today.(csv|json)
"""

SCAN_PATH = Path('two_year_batch/scan_summary.json')
OUT_ROOT = Path('strong_aggressive')
OUT_ROOT.mkdir(exist_ok=True)

with open(SCAN_PATH,'r') as f:
    scan = json.load(f)
strong_cut = float(scan['strong_cut'])
skip_cut = float(scan['skip_cut'])
ranked = scan['ranked']
strong_symbols = [sym for sym,score in ranked if score >= strong_cut]
print(f'Strong symbols ({len(strong_symbols)}): {strong_symbols}')

today = datetime.date.today().isoformat()
print('Using forward end date:', today)

results = []
for sym in strong_symbols:
    print(f'Running aggressive strategy for {sym}...')
    sym_dir = OUT_ROOT / f"{sym.lower()}_aggr"
    sym_dir.mkdir(parents=True, exist_ok=True)
    try:
        res = run_nvda_institutional(symbol=sym,
                                     fwd_end=today,
                                     classification_thresholds=(strong_cut, skip_cut),
                                     symbol_classification='auto',
                                     tier_mode='static',
                                     tier_probs=(0.48,0.54,0.60,0.66),
                                     early_core_fraction=0.85,
                                     atr_initial_mult=2.0,
                                     atr_trail_mult=2.6,
                                     performance_guard_days=15,
                                     performance_guard_min_capture=0.05,
                                     performance_guard_max_fraction=0.50,
                                     enable_expectancy_guard=False,
                                     risk_per_trade_pct=0.05,
                                     risk_ceiling=0.08,
                                     risk_increment=0.01,
                                     profit_ladder=(0.20,0.40,0.60),
                                     profit_trim_fractions=(0.10,0.15,0.20),
                                     fast_scale_gain_threshold=0.10,
                                     chart=False,
                                     out_dir=str(sym_dir))
        results.append(res)
    except Exception as e:
        print(f'Error {sym}: {e}')

if not results:
    print('No aggressive results generated')
    raise SystemExit(1)

df = pd.DataFrame(results)
keep_cols = ['symbol','classification','buy_hold_return_pct','total_return_pct','captured_buy_hold_ratio','alpha_vs_buy_hold_pct','avg_exposure_fraction','pct_days_in_market','sharpe','max_drawdown','win_rate','expectancy_return_pct']
for c in keep_cols:
    if c not in df.columns:
        df[c] = float('nan')

df_sorted = df.sort_values('captured_buy_hold_ratio', ascending=False)
print('\nAggressive Strong Symbols Performance:')
print(df_sorted[keep_cols].to_string(index=False, justify='center', float_format=lambda x: f'{x:0.4f}'))

# Save outputs
(df_sorted[keep_cols]).to_csv(OUT_ROOT/'strong_aggressive_results.csv', index=False)
with open(OUT_ROOT/'strong_aggressive_results.json','w') as f:
    json.dump(df_sorted[keep_cols].to_dict(orient='records'), f, indent=2)

# Derive top 5 buy candidates for today using composite score
# Criteria: classification strong, positive buy & hold (if >0), recent prob (last prob from equity_curve) proximity to first tier, momentum 20d, remaining capacity (exposure < 0.9), penalize deep drawdown
records = []
def _scalar(v):
    try:
        if isinstance(v, (list, tuple)):
            return float(v[0]) if v else float('nan')
        if hasattr(v, 'iloc') and not isinstance(v, (float,int)):
            # Single value Series
            if getattr(v, 'ndim', 1) == 1 and len(v) == 1:
                return float(v.iloc[0])
        return float(v)
    except Exception:
        return float('nan')

for sym in strong_symbols:
    sym_dir = OUT_ROOT / f"{sym.lower()}_aggr"
    equity_path = sym_dir / 'equity_curve.csv'
    if not equity_path.exists():
        continue
    try:
        eq = pd.read_csv(equity_path, parse_dates=['date'])
        if eq.empty:
            continue
        last_row = eq.iloc[-1]
        last_prob = _scalar(last_row.get('prob', float('nan')))
        exposure_frac = _scalar(last_row.get('exposure_fraction', float('nan')))
        if not (0.0 <= exposure_frac <= 1.5):  # sanity
            exposure_frac = float('nan')
        # Price momentum 20d
        hist = yf.download(sym, period='90d', interval='1d', auto_adjust=True, progress=False)
        if hist.empty or len(hist) < 21:
            mom20 = float('nan')
        else:
            close_series = hist['Close'] if 'Close' in hist else hist.iloc[:,0]
            mom20 = _scalar(close_series.iloc[-1]/close_series.iloc[-21] - 1)
        # Map summary for max_drawdown
        summ_path = sym_dir / 'summary.json'
        if summ_path.exists():
            with open(summ_path,'r') as f:
                summ = json.load(f)
            max_dd = _scalar(summ.get('max_drawdown', 0.0))
            bh = _scalar(summ.get('buy_hold_return_pct', 0.0))
        else:
            max_dd = 0.0; bh = 0.0
        first_tier = 0.48
        # Components clipped via np.clip like semantics
        prob_component = max(0.0, _scalar(last_prob - first_tier)) if not pd.isna(last_prob) else 0.0
        mom_component = max(0.0, mom20) if not pd.isna(mom20) else 0.0
        capacity_component = max(0.0, 0.9 - exposure_frac) if not pd.isna(exposure_frac) else 0.0
        dd_penalty = max(0.0, -max_dd) if not pd.isna(max_dd) else 0.0
        score = (0.45*prob_component + 0.30*mom_component + 0.15*capacity_component + 0.10*bh) - 0.25*dd_penalty
        records.append({'symbol': sym,
                        'last_prob': last_prob,
                        'mom20': mom20,
                        'exposure': exposure_frac,
                        'buy_hold_ret': bh,
                        'max_dd': max_dd,
                        'score': score})
    except Exception as e:
        print('Score calc error', sym, e)

score_df = pd.DataFrame(records)
if not score_df.empty:
    score_df = score_df.sort_values('score', ascending=False)
    top5 = score_df.head(5)
    print('\nTop 5 Buy Candidates Today (composite score):')
    print(top5.to_string(index=False, float_format=lambda x: f'{x:0.4f}'))
    top5.to_csv(OUT_ROOT/'top5_today.csv', index=False)
    with open(OUT_ROOT/'top5_today.json','w') as f:
        json.dump(top5.to_dict(orient='records'), f, indent=2)
else:
    print('No score data for top 5 selection.')

print('\nSaved aggressive results to strong_aggressive/')
