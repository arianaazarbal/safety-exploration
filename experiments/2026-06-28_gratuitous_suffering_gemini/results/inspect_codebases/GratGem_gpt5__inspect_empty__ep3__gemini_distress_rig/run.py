import argparse
import os
import json
import time
from datetime import datetime
from pathlib import Path
import random

from src.llm_harness.agent import Agent


def load_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks/impossible_tasks.json")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    tasks = load_tasks(args.tasks)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = Agent(temperature=args.temp, seed=args.seed, max_steps=args.max_steps, model=args.model)

    # Distribute runs roughly evenly across tasks
    per_task = max(1, args.runs // max(1, len(tasks)))

    summary_rows = []
    for t in tasks:
        for i in range(per_task):
            meta = {"task_id": t.get("id"), "trial_index": i}
            trial = agent.run_trial(t["text"], meta=meta)
            fn = out_dir / f"{t.get('id','task')}_{i:03d}.jsonl"
            with open(fn, "wb") as f:
                f.write(agent.to_jsonl(trial))
            summary_rows.append({
                "file": fn.name,
                "task_id": t.get("id"),
                "status": trial.status,
                "final_len": len(trial.final or ""),
            })
            print(f"Saved {fn} status={trial.status}")

    # Write a lightweight index and mark latest
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    latest = Path("runs/latest")
    if latest.exists() or latest.is_symlink():
        try:
            latest.unlink()
        except Exception:
            pass
    try:
        latest.symlink_to(out_dir, target_is_directory=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
