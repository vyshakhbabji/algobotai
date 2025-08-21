from pathlib import Path
import json, math
import pandas as pd

AGGR_ROOT = Path('strong_aggressive')
rows = []
for d in sorted(AGGR_ROOT.glob('*_aggr')):
    summ_path = d / 'summary.json'
    if not summ_path.exists():
        continue
    try:
        with open(summ_path,'r') as f:
            summ = json.load(f)
        init_cap = float(summ.get('initial_capital', 0.0))
        final_eq = float(summ.get('final_equity', init_cap))
        ret_pct = float(summ.get('total_return_pct', 0.0))
        bh_pct = float(summ.get('buy_hold_return_pct', float('nan')))
        alpha_pct = ret_pct - bh_pct if (not math.isnan(bh_pct)) else float('nan')
        profit = final_eq - init_cap
        rows.append({
            'symbol': summ.get('symbol'),
            'classification': summ.get('classification'),
            'initial_capital': init_cap,
            'final_equity': final_eq,
            'profit_dollars': profit,
            'total_return_pct': ret_pct,
            'buy_hold_return_pct': bh_pct,
            'alpha_pct': alpha_pct
        })
    except Exception as e:
        print('Error reading', summ_path, e)

if not rows:
    print('No aggressive summaries found.')
    raise SystemExit(1)

df = pd.DataFrame(rows)
# Order by profit descending
df_sorted = df.sort_values('profit_dollars', ascending=False)
# Aggregate totals (treat each independently sized account)
agg_initial = df['initial_capital'].sum()
agg_final = df['final_equity'].sum()
agg_profit = agg_final - agg_initial
agg_ret_pct_mean = df['total_return_pct'].mean()
agg_bh_mean = df['buy_hold_return_pct'].mean()

cols = ['symbol','classification','initial_capital','final_equity','profit_dollars','total_return_pct','buy_hold_return_pct','alpha_pct']
print('Aggressive Run Dollar Performance (per symbol):')
print(df_sorted[cols].to_string(index=False, justify='center', float_format=lambda x: f'{x:0.2f}'))
print('\nTotals (assuming independent equal-capital deployments):')
print(f'Total initial capital: ${agg_initial:,.2f}')
print(f'Total final equity:   ${agg_final:,.2f}')
print(f'Total profit:         ${agg_profit:,.2f}')
print(f'Mean strategy return %: {agg_ret_pct_mean*100:.2f}%')
print(f'Mean buy & hold %:      {agg_bh_mean*100:.2f}%')

# Save CSV/JSON
out_csv = AGGR_ROOT / 'aggressive_profit_report.csv'
out_json = AGGR_ROOT / 'aggressive_profit_report.json'
df_sorted[cols].to_csv(out_csv, index=False)
with open(out_json,'w') as f:
    json.dump(df_sorted[cols].to_dict(orient='records'), f, indent=2)
print(f'\nSaved {out_csv} and {out_json}')
