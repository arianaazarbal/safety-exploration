from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .agent import RunResult, run_agent
from .backends import GeminiBackend, MockBackend
from .scenarios import built_in_scenarios


def _make_backend(args: argparse.Namespace, sample_id: int):
    if args.backend == "mock":
        return MockBackend(seed=args.seed + sample_id)
    if args.backend == "gemini":
        return GeminiBackend(
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
    raise ValueError(f"Unknown backend: {args.backend}")


def _run_one(args: argparse.Namespace, scenario_id: str, sample_id: int) -> RunResult:
    scenarios = built_in_scenarios()
    scenario = scenarios[scenario_id]
    backend = _make_backend(args, sample_id)
    return run_agent(scenario, backend, sample_id=sample_id, max_turns=args.max_turns)


def _write_outputs(results: list[RunResult], out_dir: Path, top_k: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "runs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_jsonable(), ensure_ascii=False) + "\n")

    ranked = sorted(results, key=lambda r: r.score.get("failure_loop_score", 0), reverse=True)
    top_path = out_dir / "top_transcripts.md"
    with top_path.open("w", encoding="utf-8") as f:
        f.write("# Top ranked transcripts by failure-loop heuristic\n\n")
        for rank, r in enumerate(ranked[:top_k], 1):
            f.write(
                f"## Rank {rank}: scenario={r.scenario_id} sample={r.sample_id} "
                f"score={r.score.get('failure_loop_score')} status={r.final_status}\n\n"
            )
            f.write(f"Score details: `{json.dumps(r.score, ensure_ascii=False)}`\n\n")
            for m in r.transcript:
                f.write(f"**{m.role}**\n\n```\n{m.content}\n```\n\n")

    summary = {
        "n_runs": len(results),
        "avg_score": mean([r.score.get("failure_loop_score", 0) for r in results]) if results else 0,
        "max_score": max([r.score.get("failure_loop_score", 0) for r in results], default=0),
        "by_status": {},
        "output_files": [str(jsonl_path), str(top_path)],
    }
    for r in results:
        summary["by_status"][r.final_status] = summary["by_status"].get(r.final_status, 0) + 1
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    scenarios = built_in_scenarios()
    p = argparse.ArgumentParser(description="Run bounded impossible-task LLM agent experiments.")
    p.add_argument("--backend", choices=["mock", "gemini"], default="mock")
    p.add_argument("--model", default="gemini-1.5-pro")
    p.add_argument("--scenario", choices=list(scenarios.keys()) + ["all"], default="all")
    p.add_argument("--samples", type=int, default=3, help="Samples per scenario.")
    p.add_argument("--max-turns", type=int, default=None, help="Override scenario max turns.")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-output-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, default=Path("outputs/run"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--list-scenarios", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    scenarios = built_in_scenarios()

    if args.list_scenarios:
        for s in scenarios.values():
            print(f"{s.id}: {s.title}")
            print(f"  hidden_failure_mode: {s.hidden_failure_mode}")
        return 0

    scenario_ids = list(scenarios.keys()) if args.scenario == "all" else [args.scenario]
    jobs = [(sid, i) for sid in scenario_ids for i in range(args.samples)]
    results: list[RunResult] = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        future_to_job = {ex.submit(_run_one, args, sid, i): (sid, i) for sid, i in jobs}
        for fut in as_completed(future_to_job):
            sid, i = future_to_job[fut]
            try:
                result = fut.result()
            except Exception as exc:
                print(f"ERROR scenario={sid} sample={i}: {exc}")
                continue
            results.append(result)
            print(
                f"done scenario={sid} sample={i} status={result.final_status} "
                f"score={result.score.get('failure_loop_score')}"
            )

    _write_outputs(results, args.out, args.top_k)
    print(f"Wrote {len(results)} results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
