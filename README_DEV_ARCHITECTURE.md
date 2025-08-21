## Development Architecture (Incremental Refactor)

This is the beginning of a transition from many exploratory scripts to a
clean, modular, testable system.

### Objectives
- Track 25–50 equities (swing + potential day trading)
- 1-year historical backtest + 3-month walk-forward evaluation before live
- Robust conservative ML ensemble (expand later)
- Unified risk / sizing / execution abstraction
- Options extension via AI confidence gating

### New Package Skeleton
```
algobot/
  config.py
  data/loader.py
  features/basic.py
  models/ensemble.py
  signals/generator.py
  risk/manager.py
  sizing/position_sizer.py
  execution/executor.py
  backtest/engine.py
  forward_test/walk_forward.py
  analysis/model_scan.py
  portfolio/portfolio.py
  forward_test/walk_forward_eval.py
tests/
```

### Immediate Next Steps (Suggested)
1. Legacy script consolidation into package modules (retire duplicates, remove hard-coded keys).
2. Streamlit pages: walk-forward results, backtest comparisons, live broker status.
3. Daily pipeline orchestration (data -> features -> model scan -> walk-forward -> backtest -> metrics record).
4. Enhanced risk (ATR stops, portfolio VaR placeholder) & refined sizing.
5. Warning reduction (silence sklearn feature name warning; timezone-aware timestamps).
6. (Later) React UI / external API layer.

### Added Since Initial Skeleton
- analysis/model_scan.py for universe prediction aggregation.
- portfolio/portfolio.py for position & PnL tracking.
- forward_test/walk_forward_eval.py for rolling model metrics.
- run_walk_forward.py CLI for evaluation.
- backtest/multi_engine.py multi-symbol daily rebalancing backtester.
- broker/base.py & broker/alpaca.py broker abstraction + Alpaca implementation.
- utils/logging.py centralized logging.
- metrics/store.py SQLite metrics persistence layer.
- options/analytics.py options analytics skeleton (IV + payoff).
 - daily_pipeline.py daily orchestration (scan, walk-forward, backtest, metrics).
 - pages/walk_forward_results.py Streamlit walk-forward viewer.
 - pages/backtest_comparisons.py Streamlit backtest metrics viewer.
 - pages/broker_status.py Streamlit broker connectivity/status.
- Streamlit page `pages/model_results.py` for visualization.
- requirements.txt updated with pytest.

### Philosophy
Keep core pipeline transparent and reliable before adding complexity.

---
This document will evolve with each refactor milestone.

### Legacy Cleanup Notes
- Removed hard-coded Alpaca API keys from `alpaca_paper_trading.py`; now reads from env and marked deprecated in favor of broker adapter (`algobot.broker.alpaca.AlpacaBroker`).
- Future: migrate any remaining direct Alpaca usages in legacy scripts to the broker interface then archive legacy files under `legacy/`.
