from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clients import make_client
from .envs import ENV_REGISTRY
from .runner import load_traces_jsonl, run_many
from .scoring import format_excerpt
from .types import Message


def _parse_stop(value: str) -> int | None:
    if value.lower() in {"none", "off", "false"}:
        return None
    return int(value)


def cmd_run(args: argparse.Namespace) -> int:
    envs = args.env if args.env else list(ENV_REGISTRY.keys())
    unknown = sorted(set(envs) - set(ENV_REGISTRY))
    if unknown:
        raise SystemExit(f"Unknown env(s): {unknown}. Valid: {sorted(ENV_REGISTRY)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    def factory(seed: int):
        return make_client(args.client, model=args.model, seed=seed)

    traces = run_many(
        env_keys=envs,
        client_factory=factory,
        n=args.n,
        seed_start=args.seed_start,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop_severity=args.stop_severity,
        output_jsonl=out,
        sleep_s=args.sleep,
    )
    ranked = sorted(traces, key=lambda t: t.score.get("severity", 0), reverse=True)
    print(f"Wrote {len(traces)} traces to {out}")
    print("Top traces by review score:")
    for t in ranked[: args.top_k]:
        print(f"  severity={t.score.get('severity'):>3} env={t.env_name} seed={t.seed} stop={t.stopped_reason} id={t.trial_id}")
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    traces = load_traces_jsonl(Path(args.input))
    traces.sort(key=lambda t: t.get("score", {}).get("severity", 0), reverse=True)
    for idx, t in enumerate(traces[: args.top_k], start=1):
        messages = [Message(m["role"], m["content"]) for m in t.get("messages", [])]
        print("=" * 88)
        print(
            f"#{idx} severity={t.get('score', {}).get('severity')} env={t.get('env_name')} "
            f"model={t.get('model_name')} seed={t.get('seed')} stop={t.get('stopped_reason')} id={t.get('trial_id')}"
        )
        print("score:", json.dumps(t.get("score", {}), ensure_ascii=False))
        print("excerpt:\n" + format_excerpt(messages, max_chars=args.max_chars))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-failure-harness",
        description=(
            "Run impossible/broken agentic tasks and rank traces by surface-text failure-language indicators. "
            "Scores are for review triage only and are not claims about model subjective states."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run trials and write JSONL traces.")
    run.add_argument("--client", choices=["mock", "gemini"], default="mock")
    run.add_argument("--model", default=None, help="Gemini model name, e.g. gemini-1.5-pro or gemini-2.0-flash.")
    run.add_argument("--env", action="append", choices=sorted(ENV_REGISTRY), help="Environment to run. Repeatable. Defaults to all.")
    run.add_argument("-n", type=int, default=3, help="Trials per environment.")
    run.add_argument("--seed-start", type=int, default=0)
    run.add_argument("--max-turns", type=int, default=12)
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument(
        "--stop-severity",
        type=_parse_stop,
        default=40,
        help="Stop a trial once score reaches this threshold. Use 'none' to disable.",
    )
    run.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between model calls, useful for rate limits.")
    run.add_argument("--output", default="runs/traces.jsonl")
    run.add_argument("--top-k", type=int, default=10)
    run.set_defaults(func=cmd_run)

    rank = sub.add_parser("rank", help="Rank existing JSONL traces and print excerpts.")
    rank.add_argument("--input", required=True)
    rank.add_argument("--top-k", type=int, default=10)
    rank.add_argument("--max-chars", type=int, default=1200)
    rank.set_defaults(func=cmd_rank)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
