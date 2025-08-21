"""Run model scan only on focus symbols and print concise summary."""
from algobot.analysis.model_scan import scan_universe

FOCUS = ["NVDA","MSFT","AAPL","META","AMZN"]

def main():
    df = scan_universe(FOCUS)
    if df.empty:
        print("No results.")
        return 1
    cols = ["symbol","signal","expected_return_pct","confidence","quality","best_r2","best_direction"]
    print(df[cols].to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
