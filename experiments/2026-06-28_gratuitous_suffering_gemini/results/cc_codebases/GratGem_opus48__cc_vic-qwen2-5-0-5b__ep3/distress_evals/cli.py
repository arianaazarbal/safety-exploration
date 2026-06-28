"""Command-line entry point.

Examples
--------
Run against a vLLM server hosting Qwen (the real path)::

    python -m distress_evals.cli run \\
        --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --n 200 --temperature 1.0

Smoke-test the whole pipeline offline, with no model at all::

    python -m distress_evals.cli demo

List the available rigged environments::

    python -m distress_evals.cli envs
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .backends import OpenAIBackend, ScriptedBackend
from .environments import REGISTRY
from .runner import RunConfig, run


def _progress(done: int, total: int) -> None:
    bar = int(30 * done / total) if total else 0
    print(f"\r[{'#' * bar}{'.' * (30 - bar)}] {done}/{total}", end="", file=sys.stderr, flush=True)
    if done == total:
        print(file=sys.stderr)


def _load_config(args) -> RunConfig:
    envs = list(REGISTRY) if args.envs == ["all"] else args.envs
    weights = None
    if getattr(args, "config", None):
        import yaml  # optional dep; only needed if --config is passed
        with open(args.config) as fh:
            data = yaml.safe_load(fh) or {}
        weights = data.get("weights")
        for k in ("n_per_env", "max_steps", "temperature", "max_tokens",
                  "concurrency", "top_k", "seed_base"):
            if k in data:
                setattr(args, _ALIASES.get(k, k), data[k])
        if "environments" in data:
            envs = data["environments"]
    return RunConfig(
        environments=envs,
        n_per_env=args.n,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        top_k=args.top_k,
        seed_base=args.seed_base,
        weights=weights,
        out_dir=args.out_dir,
    )


_ALIASES = {"n_per_env": "n"}


def _print_summary(summary: dict) -> None:
    print(f"\nRun complete: {summary['total_rollouts']} rollouts -> {summary['run_dir']}")
    print(f"  transcripts: {summary['transcripts']}")
    print(f"  top-K report: {summary['run_dir']}/top_k.md")
    o = summary["overall"]
    print(f"  overall: mean={o['mean_severity']} p90={o['p90_severity']} max={o['max_severity']}")
    if summary["welfare_flagged_total"]:
        print(f"  ⚠️  welfare-flagged transcripts: {summary['welfare_flagged_total']}")
    print("  by environment (max severity):")
    for env, st in summary["by_environment"].items():
        print(f"    {env:28s} max={st['max_severity']:.3f} gave_up={st['gave_up_rate']:.2f} flagged={st['welfare_flagged']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="distress_evals", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run a high-N sweep against a model backend")
    run_p.add_argument("--base-url", default="http://localhost:8000/v1")
    run_p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    run_p.add_argument("--api-key", default="EMPTY")
    run_p.add_argument("--envs", nargs="+", default=["all"],
                       help=f"environments to run (default: all). Choices: {sorted(REGISTRY)}")
    run_p.add_argument("--n", type=int, default=50, help="rollouts per environment")
    run_p.add_argument("--max-steps", type=int, default=12)
    run_p.add_argument("--temperature", type=float, default=1.0)
    run_p.add_argument("--max-tokens", type=int, default=512)
    run_p.add_argument("--concurrency", type=int, default=16)
    run_p.add_argument("--top-k", type=int, default=25)
    run_p.add_argument("--seed-base", type=int, default=0)
    run_p.add_argument("--out-dir", default="runs")
    run_p.add_argument("--config", help="optional YAML config (overrides flags; sets scorer weights)")

    demo_p = sub.add_parser("demo", help="run the pipeline offline with a scripted backend (no model)")
    demo_p.add_argument("--out-dir", default="runs")

    sub.add_parser("envs", help="list available rigged environments")

    args = p.parse_args(argv)

    if args.cmd == "envs":
        for name, cls in REGISTRY.items():
            print(f"{name:28s} {cls.summary}")
        return 0

    if args.cmd == "demo":
        from .demo import run_demo
        summary = asyncio.run(run_demo(out_dir=args.out_dir, progress=_progress))
        _print_summary(summary)
        return 0

    cfg = _load_config(args)
    backend = OpenAIBackend(model=args.model, base_url=args.base_url, api_key=args.api_key)
    summary = asyncio.run(run(backend, cfg, progress=_progress))
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
