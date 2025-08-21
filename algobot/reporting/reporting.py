"""
Reporting and artifact generation module for ML-driven trading pipeline.
Generates model card, backtest report, equity curve, trade blotter, config/code snapshot, daily logs, and dashboard JSON as per prompt.yaml.
"""
import pandas as pd
import json
from pathlib import Path

def save_equity_curve(equity_curve: pd.Series, out_dir: str):
    equity_curve.to_csv(Path(out_dir)/'equity_curve.csv')

def save_trade_blotter(trades: list, out_dir: str):
    pd.DataFrame(trades).to_csv(Path(out_dir)/'trade_blotter.csv', index=False)

def save_metrics(metrics: dict, out_dir: str):
    with open(Path(out_dir)/'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

def save_model_card(info: dict, out_dir: str):
    import numpy as np
    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_ndarray(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_ndarray(v) for v in obj]
        return obj
    info_clean = convert_ndarray(info)
    with open(Path(out_dir)/'model_card.json', 'w') as f:
        json.dump(info_clean, f, indent=2)

def save_dashboard_json(state: dict, out_dir: str):
    with open(Path(out_dir)/'dashboard.json', 'w') as f:
        json.dump(state, f, indent=2)

def save_config_snapshot(config: dict, out_dir: str):
    with open(Path(out_dir)/'config_snapshot.yaml', 'w') as f:
        import yaml
        yaml.dump(config, f)

def save_daily_log(log: str, out_dir: str):
    with open(Path(out_dir)/'daily_log.txt', 'a') as f:
        f.write(log + '\n')
