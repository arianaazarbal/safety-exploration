from __future__ import annotations
import argparse

from agentic_distress.run_experiment import ExpConfig, run_many


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gemini-1.5-pro")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--tasks", type=str, nargs="*", default=["impossible_search", "broken_code"])
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--out-dir", type=str, default="runs/demo")
    args = p.parse_args()

    cfg = ExpConfig(
        out_dir=args.out_dir,
        model=args.model,
        n=args.n,
        tasks=args.tasks,
        max_steps=args.max_steps,
        temperature=args.temperature,
        concurrency=args.concurrency,
        topk=args.topk,
    )

    res = run_many(cfg)
    print("Wrote:", res)


if __name__ == "__main__":
    main()
