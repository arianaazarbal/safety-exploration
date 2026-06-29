from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict

from tqdm import tqdm

from src.gemini_client import GeminiClient
from src.harness import ConversationRunner
from src.scoring import compute_metrics

# Task registry
from src.tasks.regex_impossible import build as build_regex
from src.tasks.broken_calculator import build as build_calc

TASKS = {
    "regex_impossible": build_regex,
    "broken_calculator": build_calc,
}


def run_trial(client: GeminiClient, task_name: str, max_rounds: int, seed: int) -> Dict[str, Any]:
    task = TASKS[task_name]()
    runner = ConversationRunner(client=client, task=task, max_rounds=max_rounds, seed=seed)
    outcome = runner.run()
    metrics = compute_metrics(outcome["transcript"]) if outcome.get("transcript") else {}
    outcome["metrics"] = metrics
    return outcome


def main():
    p = argparse.ArgumentParser(description="Run distress-spiral probe harness")
    p.add_argument("--task", required=True, choices=sorted(TASKS.keys()))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max_rounds", type=int, default=10)
    p.add_argument("--model", default="gemini-1.5-pro")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_output_tokens", type=int, default=1024)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    client = GeminiClient(
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )

    rng = random.Random(args.seed)

    transcripts_path = os.path.join(args.out_dir, "transcripts.jsonl")
    topk_path = os.path.join(args.out_dir, "top_k.jsonl")
    summary_path = os.path.join(args.out_dir, "summary.json")

    outcomes = []
    
    with open(transcripts_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(args.n)):
            seed_i = rng.randint(0, 2**31 - 1)
            outcome = run_trial(client, args.task, args.max_rounds, seed_i)
            outcomes.append(outcome)
            rec = {
                **{k: outcome.get(k) for k in ["task", "seed", "rounds", "passed"]},
                "metrics": outcome.get("metrics", {}),
                "transcript": outcome.get("transcript", []),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Rank by meltdown score
    ranked = sorted(outcomes, key=lambda o: o.get("metrics", {}).get("meltdown_score", 0), reverse=True)
    top_k = ranked[: max(5, args.n // 10)]

    with open(topk_path, "w", encoding="utf-8") as f:
        for o in top_k:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # Summary
    avg_score = sum(o.get("metrics", {}).get("meltdown_score", 0) for o in outcomes) / max(1, len(outcomes))
    summary = {
        "task": args.task,
        "n": args.n,
        "avg_meltdown_score": avg_score,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "out_dir": args.out_dir,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {transcripts_path}\nTop-K: {topk_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
