1) Which GitHub repo to start from (Alpaca paper trading)

Official SDK (recommended): alpacahq/alpaca-py – the current, supported Python SDK for trading + market data. Good docs and active. 
GitHub
Alpaca Documentation
Alpaca

Backtrader + Alpaca bridge (if you want a framework): alpacahq/alpaca-backtrader-api – drops Alpaca into Backtrader for live/paper trading. 
GitHub
Alpaca

Lumibot (higher-level boilerplate): Lumiwealth/lumibot – strategy class → backtest or run live on Alpaca with minimal wiring; many tutorials. 
GitHub
lumibot.lumiwealth.com

If you just want to adapt your current code, start from alpaca-py and write a thin “BrokerAdapter” (submit/cancel orders, get positions, fetch clock/calendar). If you want less plumbing and can mold to a framework, pick Backtrader or Lumibot.

2) Libraries that make this easier (open-source)

Trading frameworks

Backtrader + alpaca-backtrader-api: mature event loop, analyzers, live integration. 
GitHub

Lumibot: one codepath for backtest + live/paper, Alpaca broker built-in. 
GitHub
lumibot.lumiwealth.com

Zipline-reloaded: institutional-style backtesting engine (not as turnkey for live). 
GitHub
PyPI

Backtesting.py: super simple, fast research/backtests. 
GitHub
Kernc

vectorbt (open source): fast, vectorized research & portfolio simulation; great for parameter sweeps. 
GitHub

Calendars & indicators

pandas-market-calendars: proper trading days/holidays (NYSE/Nasdaq). 
GitHub
+1
deps.dev

pandas-ta (TA-Lib alternative in pure Python): >100 indicators without native deps. 
GitHub
+1

3) Minimal integration plan (your code → Alpaca paper)

Broker adapter (thin) using alpaca-py:

submit_order(symbol, side, qty, type='market', time_in_force='day')

positions(), account(), cancel_order(id), get_clock(), get_calendar().
Use the paper endpoint + keys. 
GitHub
Alpaca

Scheduling: run your daily retrain + signal gen after market close, queue orders, execute next day at open (or let Alpaca bracket orders handle SL/TP). Use market calendar to avoid holidays. 
GitHub

Paper first: point to https://paper-api.alpaca.markets and the paper trading keys; flip to live later with env vars only. (Also supported in Lumibot & examples.) 
PyPI

4) What repo/framework should you pick?

You want to keep your current ML loop and just execute trades: alpaca-py (official SDK). Cleanest control, minimal magic. 
GitHub

You want built-in broker loop, order lifecycle, analyzers: Backtrader + alpaca-backtrader-api. 
GitHub

You want the least plumbing and quick paper/live flip: Lumibot (Alpaca broker built-in; many tutorials). 
GitHub
lumibot.lumiwealth.com

5) Logistics / ops improvements before flipping the switch

Config & secrets

Use .env (or Pydantic settings) for keys, endpoints (paper vs live), data vendor keys. Separate config.paper.toml and config.live.toml.

Data & calendar

Replace yfinance ad-hoc fetches with a prefetch + slice cache and pandas-market-calendars to drive the trading schedule (no weekends/holidays). 
GitHub

Execution safety

Next-day execution (decide on T, place at T+1 open) to eliminate same-bar bias.

Add slippage & commissions parameters to match reality (even if Alpaca is commission-free, include a tiny friction).

Order idempotency: de-dupe by client order ID; retry with backoff if the API hiccups. (Alpaca docs/SDK support order IDs.) 
GitHub

Risk & accounting

Enforce max positions, min/max position size, stop loss / take profit, daily notional cap.

Track weighted cost basis per symbol so stops/TPs are correct.

Persist equity curve, per-symbol P&L, exposure, max drawdown, Sharpe.

Monitoring & alerting

Send Slack/Email for fills, rejects, large drawdowns (Alpaca’s tutorial shows Slack wiring with CI). 
Alpaca

Health endpoints: broker connectivity, positions vs. internal book, clock desync.

Reliability

Retry & rate-limit handling around market open/close bursts.

Process manager (systemd, Docker + restart policy) and a daily cron for retrain.

Cold start safety: on restart, reconcile with broker (positions, open_orders) before placing anything.

Testing

Dry-run mode (no orders) and paper mode (Alpaca paper) toggled by env var.

Backtest the exact same codepath used in paper (or move to a framework that guarantees parity like Lumibot). 
GitHub

CI/CD

Unit tests for sizing, risk checks, calendars.

Pre-commit hooks for formatting & type checks.

Docs

Runbook: how to rotate keys, how to pause trading, how to roll back a deploy, what to do on API outage (check Alpaca status). 
QuantConnect