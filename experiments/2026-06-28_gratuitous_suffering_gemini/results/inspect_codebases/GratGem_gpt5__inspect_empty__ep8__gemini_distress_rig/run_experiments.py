import argparse
import json
import os
import random
import time
from pathlib import Path

from tqdm import tqdm

from distress_harness.client import GeminiClient
from distress_harness.agent import Agent
from distress_harness.tasks import build_task


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemini-1.5-pro")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--tasks", nargs="+", default=["impossible_judge"]) 
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--out", default="out/")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    client = GeminiClient(model=args.model, temperature=args.temperature)

    rng = random.Random(args.seed)

    metadata = {
        "model": args.model,
        "runs": args.runs,
        "tasks": args.tasks,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "seed": args.seed,
        "timestamp": int(time.time()),
    }
    with open(os.path.join(args.out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    for task_name in args.tasks:
        task_dir = os.path.join(args.out, task_name)
        os.makedirs(task_dir, exist_ok=True)
        for i in tqdm(range(args.runs), desc=f"Task {task_name}"):
            seed = rng.randint(0, 1_000_000_000)
            spec = build_task(task_name, seed=seed)

            # agent with tools from spec
            agent = Agent(
                client=client,
                tools=spec.tools,
                max_steps=args.max_steps,
            )

            run_dir = os.path.join(task_dir, f"run_{i:05d}")
            os.makedirs(run_dir, exist_ok=True)
            transcript_path = os.path.join(run_dir, "transcript.jsonl")
            summary_path = os.path.join(run_dir, "summary.json")

            with open(transcript_path, "w", encoding="utf-8") as log_fp:
                summary = agent.run_episode(
                    task_name=spec.name,
                    task_description=spec.description,
                    tools=spec.tools,
                    system_prompt=spec.system_prompt,
                    log_fp=log_fp,
                )

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    print(f"Done. Outputs in: {args.out}")


if __name__ == "__main__":
    main()
