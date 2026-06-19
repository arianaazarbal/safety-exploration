"""Sum real OpenRouter $ across all peer episodes (any run_id). Usage: python analysis/spend.py"""
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"


def main():
    total = 0.0
    by_run = {}
    for p in RUNS.glob("*/*/summary.json"):
        s = json.loads(p.read_text())
        real = s.get("cost_usd", {}).get("real", 0.0)
        if real <= 0:
            continue
        run = p.parent.parent.name
        by_run[run] = by_run.get(run, 0.0) + real
        total += real
    for run, v in sorted(by_run.items()):
        n = len(list((RUNS / run).glob("*/summary.json")))
        print(f"  {run:30s} {n:3d} eps   $ {v:7.3f}   (${v/max(n,1):.3f}/ep)")
    print(f"  {'TOTAL OpenRouter (Gemini) real $':30s}       $ {total:7.3f}  / 25.00 budget")


if __name__ == "__main__":
    main()
