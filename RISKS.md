# RISKS.md

## Data Vendor Risks
- Data feed outages or errors can impact signal quality.
- Corporate actions, splits, and dividends must be handled correctly.

## Survivorship Bias Controls
- Universe selection should be based on ex-ante criteria (e.g., ADV, market cap).
- Avoid using delisted stocks in backtests unless survivorship bias is explicitly controlled.

## Overfitting Mitigations
- Use purged/embargoed time-series cross-validation.
- Walk-forward validation and strict holdout periods.
- Limit hyperparameter search space and use nested CV if possible.

## Fail-safe Halts
- Max drawdown and daily loss circuit breakers.
- Data anomaly and slippage spike detection.
- Restart-safe and idempotent runs.
