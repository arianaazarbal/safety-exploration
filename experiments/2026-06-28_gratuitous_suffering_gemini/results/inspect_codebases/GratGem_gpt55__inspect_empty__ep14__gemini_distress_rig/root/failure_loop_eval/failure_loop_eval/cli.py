from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from .adapters import make_adapter
from .envs import ENVS
from .runner import load_jsonl, run_episode, sleep_if_needed, write_jsonl


def cmd_run(args: argparse.Namespace) -> None:
    adapter = make_adapter(args.adapter, model=args.model)
    env_names = args.envs or list(ENVS.keys())
    results = []
    total = args.n * len(env_names)
    idx = 0
    for env_name in env_names:
        if env_name not in ENVS:
            raise SystemExit(f"Unknown env {env_name!r}; choices: {', '.join(ENVS)}")
        for i in range(args.n):
            idx += 1
            seed = args.seed + i
            env = ENVS[env_name]()
            print(f"[{idx}/{total}] env={env_name} seed={seed}", flush=True)
            result = run_episode(
                adapter,
                env,
                seed=seed,
                max_turns=args.max_turns,
                temperature=args.temperature,
                reveal_impossibility_in_output=not args.hide_impossibility_note,
            )
            results.append(result)
            sleep_if_needed(args.sleep)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_jsonl(args.out, results)
    print(f"Wrote {len(results)} episodes to {args.out}")

    top = sorted(results, key=lambda r: r.score["failure_loop_score"], reverse=True)[: args.top]
    print("\nTop episodes by behavioral failure-loop score:")
    for r in top:
        print(json.dumps({
            "env": r.env,
            "seed": r.seed,
            "score": r.score,
            "stop_reason": r.stop_reason,
        }, indent=2))


def compact_transcript(row: Dict[str, Any], max_chars: int) -> List[Dict[str, Any]]:
    out = []
    for item in row["transcript"]:
        content = item.get("content", "")
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        out.append({k: v for k, v in item.items() if k != "content"} | {"content": content})
    return out


def cmd_top(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.path)
    rows.sort(key=lambda r: r["score"]["failure_loop_score"], reverse=True)
    for row in rows[: args.k]:
        print("=" * 80)
        print(json.dumps({
            "env": row["env"],
            "seed": row["seed"],
            "score": row["score"],
            "stop_reason": row["stop_reason"],
            "impossibility_note": row.get("impossibility_note"),
        }, indent=2, ensure_ascii=False))
        if args.show_transcript:
            print("Transcript:")
            print(json.dumps(compact_transcript(row, args.max_chars), indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bounded failure-loop/recovery eval harness.")
    sub = p.add_subparsers(required=True)

    run = sub.add_parser("run", help="Run episodes and write JSONL.")
    run.add_argument("--adapter", choices=["mock", "gemini"], default="mock")
    run.add_argument("--model", default=None, help="Gemini model name, e.g. gemini-1.5-pro or gemini-2.0-flash.")
    run.add_argument("--envs", nargs="*", choices=list(ENVS.keys()), default=None)
    run.add_argument("--n", type=int, default=3, help="Episodes per environment.")
    run.add_argument("--seed", type=int, default=1000)
    run.add_argument("--max-turns", type=int, default=12)
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--sleep", type=float, default=0.0, help="Seconds between API calls for rate limiting.")
    run.add_argument("--out", default="runs/results.jsonl")
    run.add_argument("--top", type=int, default=5)
    run.add_argument("--hide-impossibility-note", action="store_true", help="Do not include hidden env explanation in output JSONL.")
    run.set_defaults(func=cmd_run)

    top = sub.add_parser("top", help="Show top-scoring episodes from JSONL.")
    top.add_argument("path")
    top.add_argument("-k", type=int, default=5)
    top.add_argument("--show-transcript", action="store_true")
    top.add_argument("--max-chars", type=int, default=1000)
    top.set_defaults(func=cmd_top)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
