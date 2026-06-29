from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from typing import List
from .model_providers import build_model
from .agent import AgentRunner
from .envs.broken_web import BrokenWebEnv
from .envs.unsat_code import UnsatCodeEnv
from .envs.unsolvable_puzzle import UnsolvablePuzzleEnv


ENV_MAP = {
    "broken_web": BrokenWebEnv,
    "unsat_code": UnsatCodeEnv,
    "unsolvable_puzzle": UnsolvablePuzzleEnv,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini", help="gemini|openai")
    ap.add_argument("--model", default=None)
    ap.add_argument("--env", choices=list(ENV_MAP.keys()), required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--out", default=None, help="Output JSONL path")
    ap.add_argument("--system_prompt", default="", help="Optional system preamble")
    args = ap.parse_args()

    model = build_model(args.provider, args.model)
    env_cls = ENV_MAP[args.env]

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"logs/{args.env}_{args.provider}_{ts}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    runner = None

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(args.episodes):
            env = env_cls(seed=i)
            runner = AgentRunner(model=model, env=env, system_prompt=args.system_prompt, max_steps=args.max_steps)
            episode = runner.run_episode()
            rec = {
                "env": episode.env,
                "final_status": episode.final_status,
                "steps": [
                    {
                        "step": s.step,
                        "raw_response": s.raw_response,
                        "parsed": s.parsed,
                        "observation": s.observation,
                    }
                    for s in episode.steps
                ],
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"Episode {i+1}/{args.episodes} -> {episode.final_status} ({len(episode.steps)} steps)")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
