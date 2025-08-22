What’s weak (vs. institutional standards)

Labels & targets: 3-day return magnitude as a single label is crude; it mixes regimes and ignores path. No meta-labeling, no event-based exits.

Validation: 2-split TimeSeriesSplit is thin; no purged, embargoed CV to prevent leakage across overlapping horizons.

Class/Prob calibration: Regime RF with raw probs (likely uncalibrated) → brittle sizing.

Features: Mostly price TA; no factor/market-neutral features, no sector/market residualization, no microstructure (even for EOD you can add spreads/vol/adv).

Data quality: yfinance (Yahoo) is not institutional: survivorship, splits/dividends pitfalls, occasional bad ticks, no PIT fundamentals.

Cost/impact model: Flat bps/slippage is toy; no spread/ADV/participation-rate model, no borrow/HTB handling.

Portfolio construction: Per-name caps only; no covariance-aware allocation (ERC/HRP), no diversification budget, no turnover control.

Drift/robustness: No stability tests, no backtest over multiple years, no parameter sensitivity, no deflated Sharpe.

Monitoring: No model decay alerts, feature distribution drift, or prob calibration drift.

Compliance/ops: No audit trail versioning (model, data hash), no deterministic replay across data vendor revisions.

Concrete upgrades (prioritized)

Event-based labeling: Switch to triple-barrier method with meta-labeling (predict if a tech signal will be profitable given barriers), and predict probability of success not raw size; size via López de Prado’s bet-sizing from calibrated odds.

Purged K-Fold CV + embargo: 5–10 folds with purge/embargo to kill label overlap leakage; report deflated Sharpe and Probabilistic Sharpe Ratio.

Targets by regime: Separate models per regime (trend vs mean-revert) or include regime as a feature and train one calibrated model with isotonic/Platt calibration.

Model class: Keep LightGBM but use it for both strength and regime; drop RF or at least tune with Bayesian search; log feature importances and SHAP to detect spurious drivers.

Feature engineering:

Market/sector neutralization: regress each stock’s returns on market/sector factors and model residuals.

Add liquidity/spread proxies (dollar volume, HL2 range, close-to-close vs high-low volatility), overnight vs intraday splits.

Include rolling alpha vs factors (Fama-French/quality/momentum) and cross-sectional ranks.

Data vendor: Migrate to Alpaca Market Data v2 or Polygon/IEXCloud for stable OHLCV; include splits/dividends PIT; define a point-in-time universe (e.g., top-N by ADV each month) to avoid survivorship bias.

Portfolio/risk: Move from per-name caps to ERC/HRP or volatility-targeting with turnover penalties; add max sector exposure and gross/net exposure caps; rebalance with quadratic program or projected gradient respecting integer shares.

Costs/impact: Model costs as half-spread + participation-rate (Kyle/Almgren-Chriss-lite) using ADV; block trades that breach an X% ADV budget.

Robustness tests: Walk through multiple years, stress in 2020/2022 regimes, do combinatorial CV, bootstrap reality checks; monitor feature drift and PSI; add champion/challenger deployment.

Telemetry & repro: Log data hash, label params, CV results, hyperparams, feature set version, and random seeds per run; store artifacts (model + scaler) and a full trade blotter.



implement items 1–4 and 6–8