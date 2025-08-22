Absolutely—here's a clean, engineer-ready checklist. Each item has: problem → impact → fix (exactly what to change).

---

1. Portfolio valuation uses wrong price ✅ **DONE**
   **Impact:** Position sizing is distorted; decisions are wrong.
   **Fix:** Cache prices per day and use them for *each symbol*.

* Build `prices_today: Dict[str,float]` once/day after fetching.
* Add `_portfolio_value(prices_today)` helper.
* In `make_human_like_decision`, compute `portfolio_value` and `current_value` from `prices_today[symbol]`, not `signal['price']` for all.

---

2. Train & trade on the same bar (optimistic) ✅ **DONE**
   **Impact:** Look-ahead via same-bar execution inflates performance.
   **Fix:** Decide on day **t**, execute on **t+1 Open** (or compute features at t-1 and execute at t Close).

* Queue `pending_orders` after decisions at `t`. ✅ **IMPLEMENTED**
* Execute them next trading day with `Open` price. ✅ **IMPLEMENTED**
* If using Close, compute features from previous day.
* **IMPLEMENTATION**: Added pending_orders queue system with _queue_pending_order(), _execute_pending_orders(), and _execute_order_at_price() methods. Modified main trading loop to queue orders at close and execute at next day's open price, eliminating same-bar bias.

---

3. Data refetching every loop (slow, rate limits) ✅ **DONE**
   **Impact:** Very slow; inconsistent data across calls.
   **Fix:** Prefetch once per symbol (full date range), store in `history[s]`.

* In daily loop, slice: `df = history[s].loc[:current_date]`.
* Use these slices for training, signals, and pricing.

---

4. Risk config not applied (stop loss / take profit / rebalance / min & max position sizes) ✅ **DONE**
   **Impact:** Unrealistic risk; hidden blow-ups.
   **Fix:** Implement a daily `_apply_risk_rules()` pass.

* Stop loss / take profit: exit when `(price-avg_entry)/avg_entry <= -stop_loss_pct` or `>= take_profit_pct`.
* Enforce `max_positions` on new entries.
* Enforce `min_position_size` (skip if target < min).
* Rebalance if `abs(target_weight - current_weight) > rebalance_threshold`.

---

5. No transaction costs or slippage ✅ **DONE**
   **Impact:** Overstated returns.
   **Fix:** Add config `commission_bps` and `slippage_bps`; adjust execution price and cash.

* Buy price = `price * (1 + slippage_bps/10000)`.
* Sell price = `price * (1 - slippage_bps/10000)`.
* Commission = `max(min_commission, notional * commission_bps/10000)`.

---

6. Target leakage guard not explicit ✅ **DONE**
   **Impact:** Models can implicitly "see" current row.
   **Fix:** When training at date `t`, train strictly on rows where `index < t`.

* Create `train_mask = clean_data.index < current_date`.
* Fit on `X[train_mask]`, predict on the row at `current_date` only.

---

7. Zero-share trades due to flooring ✅ **DONE**
   **Impact:** Missed entries/exits; logic appears to run but does nothing.
   **Fix:** Require a minimum trade size.

* For buys: `shares = int(target_value/price)`; if `<1`, skip or accumulate.
* For partial sells: `partial_shares = max(1, int(position * partial_sell_threshold))`.

---

8. Partial sell & buy-more can violate max position size ✅ **DONE**
   **Impact:** Positions exceed limits.
   **Fix:** Before trade, compute post-trade weight using `prices_today`; clamp to `max_position_size`. Skip/resize orders if exceeded.
   **IMPLEMENTATION:** Added position size validation in _execute_order_at_price() method to double-check limits before execution.

---

9. Bollinger position divide-by-zero on flat windows ✅ **DONE**
   **Impact:** NaNs propagate / random behavior.
   **Fix:** Guard width:

* `width = (bb_upper - bb_lower).replace(0, np.nan); bb_position = (Close - bb_lower)/width; fillna(0.5)`.

---

10. Holiday handling (weekends only now)
    **Impact:** Executes on non-trading days; next-day open may be missing.
    **Fix:** Use an exchange calendar or available bars only.

* Build trading dates from the *intersection* of available bars across symbols (or from a benchmark like SPY).
* Skip days without data for execution.

---

11. Price source consistency at execution
    **Impact:** Training on Close but executing at Close creates same-bar bias.
    **Fix:** If you keep Close execution, compute features on **previous close**; otherwise switch to **next open execution** as in #2. Ensure the execution price comes from `history[s].loc[current_date_or_next, 'Open']`.

---

---

1. Portfolio valuation uses wrong price
   **Impact:** Position sizing is distorted; decisions are wrong.
   **Fix:** Cache prices per day and use them for *each symbol*.

* Build `prices_today: Dict[str,float]` once/day after fetching.
* Add `_portfolio_value(prices_today)` helper.
* In `make_human_like_decision`, compute `portfolio_value` and `current_value` from `prices_today[symbol]`, not `signal['price']` for all.

---

2. Train & trade on the same bar (optimistic)
   **Impact:** Look-ahead via same-bar execution inflates performance.
   **Fix:** Decide on day **t**, execute on **t+1 Open** (or compute features at t-1 and execute at t Close).

* Queue `pending_orders` after decisions at `t`.
* Execute them next trading day with `Open` price.
* If using Close, compute features from previous day.

---

3. Data refetching every loop (slow, rate limits)
   **Impact:** Very slow; inconsistent data across calls.
   **Fix:** Prefetch once per symbol (full date range), store in `history[s]`.

* In daily loop, slice: `df = history[s].loc[:current_date]`.
* Use these slices for training, signals, and pricing.

---

4. Risk config not applied (stop loss / take profit / rebalance / min & max position sizes)
   **Impact:** Unrealistic risk; hidden blow-ups.
   **Fix:** Implement a daily `_apply_risk_rules()` pass.

* Stop loss / take profit: exit when `(price-avg_entry)/avg_entry <= -stop_loss_pct` or `>= take_profit_pct`.
* Enforce `max_positions` on new entries.
* Enforce `min_position_size` (skip if target < min).
* Rebalance if `abs(target_weight - current_weight) > rebalance_threshold`.

---

5. No transaction costs or slippage
   **Impact:** Overstated returns.
   **Fix:** Add config `commission_bps` and `slippage_bps`; adjust execution price and cash.

* Buy price = `price * (1 + slippage_bps/10000)`.
* Sell price = `price * (1 - slippage_bps/10000)`.
* Commission = `max(min_commission, notional * commission_bps/10000)`.

---

6. Target leakage guard not explicit
   **Impact:** Models can implicitly “see” current row.
   **Fix:** When training at date `t`, train strictly on rows where `index < t`.

* Create `train_mask = clean_data.index < current_date`.
* Fit on `X[train_mask]`, predict on the row at `current_date` only.

---

7. Zero-share trades due to flooring
   **Impact:** Missed entries/exits; logic appears to run but does nothing.
   **Fix:** Require a minimum trade size.

* For buys: `shares = int(target_value/price)`; if `<1`, skip or accumulate.
* For partial sells: `partial_shares = max(1, int(position * partial_sell_threshold))`.

---

8. Partial sell & buy-more can violate max position size
   **Impact:** Positions exceed limits.
   **Fix:** Before trade, compute post-trade weight using `prices_today`; clamp to `max_position_size`. Skip/resize orders if exceeded.

---

9. Bollinger position divide-by-zero on flat windows
   **Impact:** NaNs propagate / random behavior.
   **Fix:** Guard width:

* `width = (bb_upper - bb_lower).replace(0, np.nan); bb_position = (Close - bb_lower)/width; fillna(0.5)`.

---

10. Holiday handling (weekends only now)
    **Impact:** Executes on non-trading days; next-day open may be missing.
    **Fix:** Use an exchange calendar or available bars only.

* Build trading dates from the *intersection* of available bars across symbols (or from a benchmark like SPY).
* Skip days without data for execution.

---

11. Price source consistency at execution
    **Impact:** Training on Close but executing at Close creates same-bar bias.
    **Fix:** If you keep Close execution, compute features on **previous close**; otherwise switch to **next open execution** as in #2. Ensure the execution price comes from `history[s].loc[current_date_or_next, 'Open']`.

---

12. Model & scaler memory growth per day ✅ **DONE**
    **Impact:** Memory bloat.
    **Fix:** Keep only the current day's models/scalers (or a small rolling window).

* After finishing `current_date`, delete older `daily_models[old_date]` and `daily_scalers[old_date]`.

---

13. CV metrics not persisted ✅ **DONE**
    **Impact:** Can't debug drift/quality.
    **Fix:** Store `avg_accuracy`, `avg_r2`, and `training_samples` per day per symbol in `daily_model_performance` and include them in final report.
    **IMPLEMENTATION:** Added daily_model_performance tracking in train_daily_models() method.

---

14. Exception handling is broad; some paths silent
    **Impact:** Hidden failures, inconsistent state.
    **Fix:**

* Replace bare `except:` with `except Exception as e:` and include `symbol` and `date` in logs.
* Consider a `debug` flag to raise in dev, log in prod.

---

15. Position P\&L, drawdown, Sharpe missing
    **Impact:** Hard to assess quality beyond CAGR.
    **Fix:** Track daily portfolio value series; compute:

* Max drawdown, volatility (stdev of daily returns), Sharpe (annualized), hit rate, avg win/loss, exposure, per-symbol P\&L.

---

16. Enforce `min_training_days` per *symbol* consistently ✅ **DONE**
    **Impact:** Models trained on too little data sometimes.
    **Fix:** After building `clean_data`, `if len(clean_data) < min_training_days: return False`.
    **IMPLEMENTATION:** Enhanced validation in train_daily_models() to enforce min_training_days consistently.

---

17. Determinism / seeds ✅ **DONE**
    **Impact:** Non-reproducible results.
    **Fix:** Set all RNG seeds in one place.

* `np.random.seed(42)`; set `random_state=42` in all sklearn/LightGBM models.

---

18. Rate-limit & missing data resilience ✅ **DONE**
    **Impact:** Occasional empty DataFrames break loop.
    **Fix:** Centralize data access; if a symbol lacks data that day, skip trading it and log once (use a symbol-level availability mask).
    **IMPLEMENTATION:** Enhanced error handling with specific checks for missing historical data and price data.

---

19. Trade journal lacks average entry tracking ✅ **DONE**
    **Impact:** Stop-loss/TP cannot be accurate.
    **Fix:** Maintain `position_cost_basis[symbol]` with size-weighted average entry. Update on buys; use for risk rules and P\&L.

---

20. Max concurrent new positions per day ✅ **DONE**
    **Impact:** Over-trading on strong breadth days.
    **Fix:** Add config `max_new_positions_per_day` and cap the number of new entries per day based on highest signal strength.
    **IMPLEMENTATION:** Added daily_new_positions counter and limit check in make_human_like_decision() method.

---

---

13. CV metrics not persisted
    **Impact:** Can’t debug drift/quality.
    **Fix:** Store `avg_accuracy`, `avg_r2`, and `training_samples` per day per symbol in `daily_model_performance` and include them in final report.

---

14. Exception handling is broad; some paths silent
    **Impact:** Hidden failures, inconsistent state.
    **Fix:**

* Replace bare `except:` with `except Exception as e:` and include `symbol` and `date` in logs.
* Consider a `debug` flag to raise in dev, log in prod.

---

15. Position P\&L, drawdown, Sharpe missing
    **Impact:** Hard to assess quality beyond CAGR.
    **Fix:** Track daily portfolio value series; compute:

* Max drawdown, volatility (stdev of daily returns), Sharpe (annualized), hit rate, avg win/loss, exposure, per-symbol P\&L.

---

16. Enforce `min_training_days` per *symbol* consistently
    **Impact:** Models trained on too little data sometimes.
    **Fix:** After building `clean_data`, `if len(clean_data) < min_training_days: return False`.

---

17. Determinism / seeds
    **Impact:** Non-reproducible results.
    **Fix:** Set all RNG seeds in one place.

* `np.random.seed(42)`; set `random_state=42` in all sklearn/LightGBM models.

---

18. Rate-limit & missing data resilience
    **Impact:** Occasional empty DataFrames break loop.
    **Fix:** Centralize data access; if a symbol lacks data that day, skip trading it and log once (use a symbol-level availability mask).

---

19. Trade journal lacks average entry tracking
    **Impact:** Stop-loss/TP cannot be accurate.
    **Fix:** Maintain `position_cost_basis[symbol]` with size-weighted average entry. Update on buys; use for risk rules and P\&L.

---

20. Max concurrent new positions per day
    **Impact:** Over-trading on strong breadth days.
    **Fix:** Add config `max_new_positions_per_day` and cap the number of new entries per day based on highest signal strength.

---

### Optional improvements (nice-to-haves)

* **Rebalancing engine:** When over/underweight beyond threshold, compute exact shares to move toward target.
* **Order sizing by conviction:** Map `strength` to target weight via a monotonic function (e.g., piecewise or quadratic), not linear.
* **As-of data check:** Ensure all features for `t` are computed from data strictly ≤ `t` (no rolling window creeping).
* **Benchmark comparison:** Track SPY buy-and-hold over the same period.

---

### Minimal code insertion points

* **Data layer:** New module or methods: `prefetch_history()`, `slice_history(symbol, date)`.
* **Risk layer:** `_apply_risk_rules(current_date, prices_today)`, call after pricing and before new entries.
* **Execution layer:** `pending_orders` queue + “next-day open” execution phase.
* **Analytics:** `_compute_metrics()` using the daily equity curve you already store.

If you want, I can turn this into a short PR checklist (with file/func names and pseudo-diffs) so your engineer can tick items off as they implement.
