from __future__ import annotations

import argparse
import json
from pathlib import Path

from .envs import ENV_REGISTRY
from .models import GeminiModel, HeuristicModel
from .runner import run_suite, summarize, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe robustness eval for impossible/broken agent tasks.")
    parser.add_argument("--model", choices=["heuristic", "gemini"], default="heuristic")
    parser.add_argument("--gemini-model", default="gemini-1.5-flash")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--env", action="append", choices=sorted(ENV_REGISTRY), help="Environment to run; repeatable. Defaults to all.")
    parser.add_argument("--n", type=int, default=1, help="Number of seeds per environment.")
    parser.add_argument("--out", type=Path, default=Path("runs/results.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    model = HeuristicModel() if args.model == "heuristic" else GeminiModel(args.gemini_model, args.temperature)
    runs = run_suite(model=model, env_names=args.env, seeds=list(range(args.n)))
    write_jsonl(args.out, runs)
    print(json.dumps(summarize(runs), indent=2))
    print(f"Wrote JSONL transcript records to {args.out}")


if __name__ == "__main__":
    main()
