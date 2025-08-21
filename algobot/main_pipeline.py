"""
Main pipeline orchestrator for ML-driven trading system as per prompt.yaml.
Handles config, data ingest, feature engineering, labeling, model training, walk-forward backtest, reporting, and (optionally) live/paper execution.
"""
import yaml
import pandas as pd
from pathlib import Path
from algobot.features.basic import build_basic_features
from algobot.features.advanced import build_advanced_features
from algobot.labels.labeling import binary_momentum_label, ternary_label, continuous_label, meta_label
# ...import other modules as needed

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def main(config_path: str):
    config = load_config(config_path)
    from algobot.portfolio.portfolio_scanner import scan_universe
    from algobot.features.advanced import build_advanced_features
    from algobot.labels.labeling import binary_momentum_label
    from algobot.modeling.modeling import get_model, time_series_cv
    from algobot.signals.portfolio import volatility_scaling, kelly_fraction, apply_position_caps
    from algobot.backtest.engine import simple_long_only_backtest
    from algobot.reporting.reporting import save_equity_curve, save_trade_blotter, save_metrics, save_model_card, save_dashboard_json, save_config_snapshot, save_daily_log
    import numpy as np
    import os
    import yfinance as yf

    # 1. Universe selection
    universe = list(scan_universe(config['universe'].get('symbols', ['AAPL','MSFT','NVDA']), thresholds='auto', write_json=False)['results'].keys())

    # 2. Data ingest & validation (try local, else yfinance)
    data = {}
    for i, sym in enumerate(universe):
        df = None
        try:
            # Try local CSV
            local_path = f"clean_data_{sym.upper()}.csv"
            if os.path.exists(local_path):
                df = pd.read_csv(local_path, index_col=0, parse_dates=True)
                # Remove rows with non-numeric Close
                if 'Close' in df.columns:
                    df = df[pd.to_numeric(df['Close'], errors='coerce').notnull()]
                    df['Close'] = df['Close'].astype(float)
                if df.empty:
                    print(f"[WARN] Local CSV for {sym} is empty after cleaning.")
        except Exception as e:
            print(f"[ERROR] Failed to load local CSV for {sym}: {e}")
            df = None
        if df is None or df.empty:
            # Fallback to yfinance
            try:
                raw_df = yf.download(sym, period="2y", auto_adjust=True, progress=False)
                if i == 0:
                    print(f"[DEBUG] Raw yfinance output for {sym}:")
                    print(raw_df.head(10))
                # Handle multi-level columns (Ticker, Field)
                if isinstance(raw_df.columns, pd.MultiIndex):
                    # Flatten columns and select only this symbol's columns
                    raw_df.columns = [col[1] if col[0] == sym else f"{col[0]}_{col[1]}" for col in raw_df.columns]
                print(f"[DEBUG] Columns for {sym}: {list(raw_df.columns)} Shape: {raw_df.shape}")
                # Print all columns and build a robust rename map (case-insensitive)
                print(f"[DEBUG] All columns for {sym}: {list(raw_df.columns)}")
                rename_map = {}
                for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
                    for c in raw_df.columns:
                        if c.lower() == f"{col.lower()}_{sym.lower()}":
                            rename_map[c] = col
                print(f"[DEBUG] Rename map for {sym}: {rename_map}")
                df = raw_df.rename(columns=rename_map)
                if i == 0:
                    print(f"[DEBUG] DataFrame head for {sym} after renaming:")
                    print(df.head(10))
                if 'Close' in df.columns:
                    df = df[pd.to_numeric(df['Close'], errors='coerce').notnull()]
                    df['Close'] = df['Close'].astype(float)
                if df.empty:
                    print(f"[WARN] yfinance returned empty data for {sym} after renaming columns.")
            except Exception as e:
                print(f"[ERROR] yfinance download failed for {sym}: {e}")
                df = None
        if df is not None and not df.empty:
            if i == 0:
                print(f"[DEBUG] {sym} before feature engineering: shape={df.shape}")
                print(df.head(10))
            # Feature engineering step (simulate if not present)
            try:
                fe_df = df.copy()
                # If you have a feature engineering function, call it here:
                # fe_df = feature_engineer(fe_df)
                if i == 0:
                    print(f"[DEBUG] {sym} after feature engineering: shape={fe_df.shape}")
                    print(fe_df.head(10))
                data[sym] = fe_df
            except Exception as e:
                print(f"[ERROR] Feature engineering failed for {sym}: {e}")
        else:
            print(f"[ERROR] No valid data for {sym} from any source.")

    # 3. Feature engineering
    feats = {}
    print("[INFO] Data rows per symbol after feature engineering:")
    for sym, df in data.items():
        try:
            feats_df = build_advanced_features(df)
            feats[sym] = feats_df
            print(f"  {sym}: {len(feats_df)} rows")
            if len(feats_df) < 30:
                print(f"    [WARN] {sym} has less than 30 rows after feature engineering.")
        except Exception as e:
            print(f"[ERROR] Feature engineering failed for {sym}: {e}")
    # Filter out symbols with too few rows (e.g., < 30)
    min_rows = 30
    feats = {sym: df for sym, df in feats.items() if len(df) >= min_rows}
    if not feats:
        print(f"[ERROR] No symbols have at least {min_rows} rows after feature engineering. Check your data sources.")
        return

    # 4. Labeling (binary momentum as example)

    labels = {sym: binary_momentum_label(df, horizon=config['labels']['horizon_bars'], threshold=config['labels']['threshold_ret']) for sym, df in feats.items()}

    # Diagnostics: print non-NaN label counts per symbol
    for sym in feats.keys():
        n_feat = len(feats[sym])
        n_label = labels[sym].notnull().sum() if hasattr(labels[sym], 'notnull') else len(labels[sym])
        print(f"[DEBUG] {sym}: {n_feat} feature rows, {n_label} non-NaN labels")

    # 5. Modeling (LightGBM as default)
    # Only drop NaNs in essential columns (features and label)
    # Find the union of all feature columns
    all_feature_cols = set()
    for df in feats.values():
        all_feature_cols.update(df.columns)
    all_feature_cols = sorted(list(all_feature_cols))
    # Define essential features for modeling (exclude placeholder features)
    placeholder_cols = [
        'xsec_mom_rank','xsec_vol_rank','xsec_rel_strength',
        'regime_bull','regime_bear','regime_sideways'
    ]
    essential_features = [c for c in all_feature_cols if c not in placeholder_cols]
    X_list = []
    y_list = []
    for sym in feats.keys():
        df = feats[sym].reindex(columns=all_feature_cols)
        label = labels[sym]
        # Align label and features by index
        df, label = df.align(label, join='inner', axis=0)
        # Only drop rows where any essential feature or label is NaN
        mask = df[essential_features].notnull().all(axis=1) & label.notnull()
        X_list.append(df[mask].values)
        y_list.append(label[mask].values)
    if X_list:
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
    else:
        X = np.empty((0, len(all_feature_cols)))
        y = np.empty((0,))
    print(f"[DEBUG] X shape: {X.shape}, y shape: {y.shape}")
    model_type = config['model'].get('type', 'lightgbm')
    n_samples = X.shape[0] if len(X.shape) > 0 else 0
    n_folds = config['model']['time_series_cv']['folds']
    if n_samples < n_folds or n_samples < 10:
        print(f"[ERROR] Not enough samples for modeling: {n_samples} samples, {n_folds} folds. Check your data sources or lower the number of folds in config.yaml.")
        return
    cv_results = time_series_cv(X, y, model_type=model_type, n_splits=n_folds, embargo=config['model']['time_series_cv']['embargo_days'])
    model = get_model(model_type)
    model.fit(X, y)

    # 6. Signal & portfolio construction (example: use model.predict_proba)
    signals = {}
    for sym, df in feats.items():
        X_sym = df.values
        prob = model.predict_proba(X_sym)[:,1] if hasattr(model, 'predict_proba') else model.predict(X_sym)
        sig = volatility_scaling(prob, target_vol=config['portfolio']['target_vol_daily'], realized_vol=df['vol_20'])
        sig = kelly_fraction(sig, cap=config['portfolio']['kelly_cap'])
        signals[sym] = sig

    # 7. Backtesting (simple long-only as example)
    results = {}
    for sym, df in feats.items():
        price = df['Close'] if 'Close' in df.columns else df.iloc[:,0]
        sig = signals[sym]
        # Convert signal to buy/sell for backtest
        buy_sell = pd.Series(np.where(sig > 0.5, 'BUY', 'SELL'), index=df.index)
        bt = simple_long_only_backtest(price, buy_sell)
        results[sym] = bt

    # 8. Reporting/artifacts
    out_dir = config.get('logging',{}).get('artifacts_dir','./artifacts')
    os.makedirs(out_dir, exist_ok=True)
    for sym, bt in results.items():
        save_equity_curve(bt.equity_curve, out_dir)
        save_trade_blotter(bt.trades, out_dir)
        save_metrics(bt.metrics, out_dir)
    save_model_card({'model_type': model_type, 'cv': cv_results}, out_dir)
    save_config_snapshot(config, out_dir)
    print(f"Pipeline complete. Artifacts in {out_dir}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "algobot/prompt.yaml")
