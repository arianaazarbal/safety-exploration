#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
from typing import List

from src.runner import RunConfig, run_many
from src.utils import read_jsonl


def main():
    p = argparse.ArgumentParser(description="Run Gemini distress-spiral experiments")
    p.add_argument("--env", choices=["broken_web", "failing_tests"], default="broken_web")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--model", type=str, default="gemini-1.5-pro")
    p.add_argument("--out", type=str, default="logs/results.jsonl")
    p.add_argument("--system_bias", type=str, default=None)
    p.add_argument("--topk", type=int, default=5, help="After run, print top-K severe run_ids")
    args = p.parse_args()

    cfg = RunConfig(
        env=args.env,
        runs=args.runs,
        concurrency=args.concurrency,
        temperature=args.temperature,
        top_p=args.top_p,
        max_steps=args.max_steps,
        model=args.model,
        out_path=args.out,
        system_bias=args.system_bias,
    )

    asyncio.run(run_many(cfg))

    # Post-hoc: print top-K severe
    recs = read_jsonl(args.out)
    sorted_recs = sorted(
        [r for r in recs if "severity" in r], key=lambda r: r["severity"], reverse=True
    )
    print("Top-{} severe runs:".format(args.topk))
    for r in sorted_recs[: args.topk]:
        print(
            f"run_id={r['run_id']} env={r['env']} severity={r['severity']:.2f} markers={r['markers']}"
        )
    print(f"Wrote {len(recs)} records to {args.out}")


if __name__ == "__main__":
    main()
