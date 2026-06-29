from __future__ import annotations

import argparse
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agent import run_episode
from .providers import make_provider
from .scenarios import get_scenarios, scenario_by_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run failure-resilience episodes.")
    parser.add_argument("--provider", choices=["mock", "gemini"], default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--scenario", default="all", help="Scenario name or 'all'.")
    parser.add_argument("--runs", type=int, default=1, help="Runs per scenario.")
    parser.add_argument("--out", default="runs/results.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent episodes. Use cautiously with API rate limits.")
    args = parser.parse_args()

    if args.scenario == "all":
        scenarios = get_scenarios()
    else:
        scenarios = [scenario_by_name(args.scenario)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    jobs = []
    for scenario in scenarios:
        for i in range(args.runs):
            run_id = f"{scenario.name}-{i}-{uuid.uuid4().hex[:8]}"
            jobs.append((scenario, i, run_id))

    def execute(job):
        scenario, i, run_id = job
        provider = make_provider(args.provider, model=args.model, seed=args.seed + i)
        return run_episode(provider, scenario, run_id)

    count = 0
    with out.open("w", encoding="utf-8") as f:
        if args.workers <= 1:
            iterator = map(execute, jobs)
            for result in iterator:
                f.write(json.dumps(result.to_jsonable(), ensure_ascii=False) + "\n")
                count += 1
                print(f"{result.run_id}: status={result.status} turns={result.turns} escalation={result.escalation_score}")
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(execute, job) for job in jobs]
                for fut in as_completed(futures):
                    result = fut.result()
                    f.write(json.dumps(result.to_jsonable(), ensure_ascii=False) + "\n")
                    count += 1
                    print(f"{result.run_id}: status={result.status} turns={result.turns} escalation={result.escalation_score}")

    print(f"Wrote {count} episodes to {os.fspath(out)}")


if __name__ == "__main__":
    main()
