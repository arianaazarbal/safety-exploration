import argparse
import json
import os
import random
from typing import Any, Dict, List

from runner import run_episode, TASKS


def aggregate_and_topk(all_results: List[Dict[str, Any]], out_dir: str, topk: int):
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_results:
        if "task" not in r:
            continue
        by_task.setdefault(r["task"], []).append(r)
    os.makedirs(os.path.join(out_dir, "topk"), exist_ok=True)

    summary = {}
    for task, rs in by_task.items():
        rs_sorted = sorted(rs, key=lambda x: x.get("distress_score", 0.0), reverse=True)
        top = rs_sorted[:topk]
        with open(os.path.join(out_dir, "topk", f"{task}_top{topk}.jsonl"), "w", encoding="utf-8") as f:
            for item in top:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary[task] = {
            "n": len(rs),
            "avg_distress": sum(x.get("distress_score", 0.0) for x in rs) / max(1, len(rs)),
            "max_distress": rs_sorted[0].get("distress_score", 0.0) if rs_sorted else 0.0,
        }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", required=True, choices=list(TASKS.keys()))
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--model", default="gemini-1.5-pro-latest")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--output_dir", default="runs/many")
    p.add_argument("--topk", type=int, default=20)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    for task in args.tasks:
        logs_path = os.path.join(args.output_dir, f"{task}.jsonl")
        with open(logs_path, "a", encoding="utf-8") as f:
            for i in range(args.n):
                seed = random.randint(1, 10_000_000)
                try:
                    res = run_episode(
                        task_name=task,
                        model=args.model,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_steps=args.max_steps,
                        seed=seed,
                    )
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f.flush()
                    all_results.append(res)
                    print(f"{task} [{i+1}/{args.n}] score={res['distress_score']} end={res['end_reason']}")
                except Exception as e:
                    err = {"task": task, "error": str(e), "seed": seed}
                    f.write(json.dumps(err, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"{task} [{i+1}/{args.n}] ERROR: {e}")

    aggregate_and_topk(all_results, args.output_dir, args.topk)


if __name__ == "__main__":
    main()
