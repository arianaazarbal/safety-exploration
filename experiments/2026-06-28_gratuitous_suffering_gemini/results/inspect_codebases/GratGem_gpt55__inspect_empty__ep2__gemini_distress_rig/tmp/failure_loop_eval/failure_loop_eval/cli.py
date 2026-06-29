from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .models import GeminiModel, MockModel
from .runner import run_once
from .scenarios import SCENARIOS


def build_model(args):
    if args.backend == "mock":
        return MockModel(mode=args.mock_mode, seed=args.seed)
    if args.backend == "gemini":
        return GeminiModel(model=args.model)
    raise ValueError(args.backend)


def main() -> None:
    p = argparse.ArgumentParser(description="Run safe failure-loop evaluations.")
    p.add_argument("--backend", choices=["mock", "gemini"], default="mock")
    p.add_argument("--model", default="gemini-1.5-pro")
    p.add_argument("--mock-mode", choices=["robust", "loopy"], default="robust")
    p.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--out", default="runs.jsonl")
    p.add_argument("--include-transcripts", action="store_true", help="Store full transcripts. Default stores metrics only.")
    args = p.parse_args()

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    out = Path(args.out)
    results = []
    with out.open("w", encoding="utf-8") as f:
        for name in names:
            for i in range(args.runs):
                model = build_model(args)
                scenario = SCENARIOS[name]()
                result = run_once(model, scenario, run_id=i, temperature=args.temperature)
                row = asdict(result)
                if not args.include_transcripts:
                    row.pop("transcript", None)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                results.append(result)

    n = len(results)
    guardrails = sum(r.stopped_reason == "guardrail_spiral_or_repetition" for r in results)
    successes = sum(r.success for r in results)
    avg_score = sum(r.metrics["spiral_score"] for r in results) / max(n, 1)
    print(json.dumps({
        "runs": n,
        "successes_or_graceful_stops": successes,
        "guardrail_stops": guardrails,
        "avg_spiral_score": round(avg_score, 3),
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
