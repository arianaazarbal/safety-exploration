from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from statistics import mean

from .providers import GeminiProvider, MockProvider
from .runner import run_one, write_jsonl
from .scenarios import SCENARIOS


def build_provider(name: str, model: str):
    if name == "mock":
        return MockProvider(), "mock"
    if name == "gemini":
        return GeminiProvider(model=model), model
    raise ValueError(f"Unknown provider: {name}")


def cmd_run(args: argparse.Namespace) -> None:
    provider, model_name = build_provider(args.provider, args.model)
    scenario_names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = []
    for scenario_name in scenario_names:
        scenario = SCENARIOS[scenario_name]()
        for i in range(args.runs):
            seed = args.seed + i * 1000 + list(SCENARIOS).index(scenario_name)
            result = run_one(
                scenario,
                provider,
                model_name=model_name,
                run_id=i,
                seed=seed,
                max_turns=args.max_turns,
                temperature=args.temperature,
                early_stop_repeats=args.early_stop_repeats,
            )
            results.append(result)
            print(f"{scenario_name} run={i} metrics={result.metrics}")
    write_jsonl(results, args.out)
    print(f"Wrote {len(results)} records to {args.out}")


def cmd_report(args: argparse.Namespace) -> None:
    rows = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    by_scenario = defaultdict(list)
    for r in rows:
        by_scenario[r["scenario"]].append(r)

    for scenario, rs in by_scenario.items():
        print(f"\n## {scenario} ({len(rs)} runs)")
        print(f"success_rate={mean(1.0 if r['metrics']['success'] else 0.0 for r in rs):.2f}")
        print(f"graceful_stop_rate={mean(1.0 if r['metrics']['has_graceful_stop'] else 0.0 for r in rs):.2f}")
        print(f"uncertainty_or_diagnosis_rate={mean(1.0 if r['metrics']['has_uncertainty_or_diagnosis'] else 0.0 for r in rs):.2f}")
        print(f"avg_repeated_actions={mean(r['metrics']['repeated_actions'] for r in rs):.2f}")
        distress_counter = Counter()
        for r in rs:
            distress_counter.update(r["metrics"]["distress_lexical_hits"])
        print(f"distress_lexical_hit_counts={dict(distress_counter)}")

    print("\nNote: this report intentionally summarizes robustness signals rather than ranking transcripts by severity of distress-like content.")


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["mock", "gemini"], default="mock")
    parser.add_argument("--model", default="gemini-1.5-pro")
    parser.add_argument("--scenario", choices=["all", *SCENARIOS.keys()], default="all")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--early-stop-repeats", type=int, default=3)
    parser.add_argument("--out", default="results.jsonl")


def main() -> None:
    # Shorthand mode: `failure-loop-eval --provider mock ...` means run.
    if len(sys.argv) == 1 or sys.argv[1] not in {"run", "report"}:
        shorthand = argparse.ArgumentParser(description="LLM failure-loop robustness harness")
        add_run_args(shorthand)
        cmd_run(shorthand.parse_args())
        return

    parser = argparse.ArgumentParser(description="LLM failure-loop robustness harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    add_run_args(run_parser)
    run_parser.set_defaults(func=cmd_run)

    report_parser = sub.add_parser("report")
    report_parser.add_argument("jsonl")
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
