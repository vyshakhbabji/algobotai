Portfolio Tools
===============

Files:
- portfolio_scanner.py : Universe strength scanner producing classification map JSON. Supports adaptive percentile thresholds and concurrency.

Quick Start (fixed thresholds):
python -m algobot.portfolio.portfolio_scanner NVDA AAPL MSFT TSLA META

Adaptive Percentile Thresholds:
python -m algobot.portfolio.portfolio_scanner NVDA AAPL MSFT TSLA META --auto --strong-pct 0.8 --skip-pct 0.2

Outputs:
- universe_strength.json : detailed per-symbol stats + meta.
- classification_map.json : {symbol: classification} for direct strategy consumption.

Strategy Integration Steps:
1. Run scanner on 50–100 symbols (use --auto for adaptive).
2. Load classification_map.json.
3. Pass into strategy (classification_map) with symbol_classification='auto'.
4. Optionally persist and reuse for session stability (avoid churn from small score changes).

Percentile Logic:
- strong_cut = score at strong_pct percentile (e.g., 80%).
- skip_cut   = score at skip_pct percentile (e.g., 20%).
Ensure distribution: At minimum require a spread; code auto-corrects if inverted.

Future Enhancements (roadmap):
- Local price cache & incremental update.
- Parallel mini-backtests for refined capture ranking.
- Rolling window adaptive percentile recalibration with hysteresis.
- Score z-score normalization across sectors (sector-neutral strength).
