from pathlib import Path
import json
import pandas as pd
from algobot.forward_test.nvda_institutional_strategy import run_nvda_institutional

SCAN_PATH = Path('two_year_batch/scan_summary.json')
OUT_DIR = Path('strong_baseline')
OUT_DIR.mkdir(exist_ok=True)

with open(SCAN_PATH,'r') as f:
    scan = json.load(f)
strong_cut = float(scan['strong_cut'])
skip_cut = float(scan['skip_cut'])
ranked = scan['ranked']
strong_symbols = [sym for sym,score in ranked if score >= strong_cut]
print(f'Strong symbols ({len(strong_symbols)}):', strong_symbols)

results = []
for sym in strong_symbols:
    print(f'Running baseline strategy for {sym}...')
    try:
        sym_dir = OUT_DIR / f"{sym.lower()}_baseline"
        sym_dir.mkdir(parents=True, exist_ok=True)
        res = run_nvda_institutional(symbol=sym,
                                     classification_thresholds=(strong_cut, skip_cut),
                                     symbol_classification='auto',
                                     chart=False,
                                     out_dir=str(sym_dir))
        results.append(res)
    except Exception as e:
        print(f'Error {sym}: {e}')

if not results:
    print('No results generated')
    raise SystemExit(1)

df = pd.DataFrame(results)
keep_cols = ['symbol','classification','buy_hold_return_pct','total_return_pct','captured_buy_hold_ratio','avg_exposure_fraction','pct_days_in_market','sharpe','max_drawdown','win_rate','expectancy_return_pct']
for c in keep_cols:
    if c not in df.columns:
        df[c] = float('nan')

df_sorted = df.sort_values('captured_buy_hold_ratio', ascending=False)
print('\nBaseline Strong Symbols Performance:')
print(df_sorted[keep_cols].to_string(index=False, justify='center', float_format=lambda x: f'{x:0.4f}'))

# Save outputs
(df_sorted[keep_cols]).to_csv(OUT_DIR/'strong_baseline_results.csv', index=False)
with open(OUT_DIR/'strong_baseline_results.json','w') as f:
    json.dump(df_sorted[keep_cols].to_dict(orient='records'), f, indent=2)

print('\nSaved results to strong_baseline/strong_baseline_results.csv and .json')
