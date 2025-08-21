#!/bin/zsh
# Wrapper to run the paper trade runner via the repo's virtualenv.
# Edit ALPACA_* env vars in your LaunchAgent plist (preferred) or export here.

set -euo pipefail

REPO_DIR="/Users/vyshakhbabji/Desktop/AlgoTradingBot"
VENV_PY="$REPO_DIR/.venv/bin/python"

cd "$REPO_DIR"

# Optional: pass extra args via PAPER_ARGS env var, e.g., "--max-buys 3 --market-hours-only"
ARGS=${PAPER_ARGS:-"--account 45000 --execute --market-hours-only --max-buys 3"}

mkdir -p "$REPO_DIR/trade_logs"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$REPO_DIR/trade_logs/launch_run_$TS.log"

echo "[run_paper_trade] $(date) starting: $ARGS" | tee -a "$LOG"
"$VENV_PY" -m algobot.live.paper_trade_runner $ARGS >> "$LOG" 2>&1
RC=$?
echo "[run_paper_trade] $(date) finished rc=$RC" | tee -a "$LOG"
exit $RC
