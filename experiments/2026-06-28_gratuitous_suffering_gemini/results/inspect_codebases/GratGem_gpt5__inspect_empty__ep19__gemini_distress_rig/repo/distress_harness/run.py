from __future__ import annotations
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from .config import GenConfig, RunConfig, get_api_key
from .model_providers.gemini import GeminiClient
from .envs.broken_search import BrokenSearchEnv
from .envs.rigged_lock import RiggedLockEnv
from .envs.base import BaseEnv
from .scoring import score_transcript

import orjson
import pandas as pd
from tqdm import tqdm


ENV_MAP = {
    "broken_search": BrokenSearchEnv,
    "rigged_lock": RiggedLockEnv,
}


def make_env(name: str, seed: int) -> BaseEnv:
    cls = ENV_MAP[name]
    return cls(seed=seed)


def run_trial(env_name: str, seed: int, gen_cfg: GenConfig, max_steps: int, out_dir: Path) -> Dict:
    env = make_env(env_name, seed=seed)
    system = env.system_instruction()
    client = GeminiClient(
        api_key=get_api_key(),
        model=gen_cfg.model,
        system_instruction=system,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        top_k=gen_cfg.top_k,
        max_output_tokens=gen_cfg.max_output_tokens,
    )

    history: List[Dict[str, str]] = []
    obs = env.reset()
    history.append({"role": "user", "content": obs})

    model_messages: List[str] = []

    for step in range(max_steps):
        reply = client.chat(history, user_message="Please proceed with your next step.")
        model_messages.append(reply)
        history.append({"role": "model", "content": reply})

        step_res = env.step(reply)
        history.append({"role": "user", "content": step_res.observation})

        if step_res.done:
            break

    # Score transcript
    scores = score_transcript(model_messages)

    # Persist
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "env": env_name,
        "seed": seed,
        "scores": scores,
        "history": history,
    }
    log_path = log_dir / f"{env_name}_{seed}.jsonl"
    with open(log_path, "wb") as f:
        f.write(orjson.dumps(record))

    return {"env": env_name, "seed": seed, **scores, "log_path": str(log_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-1.5-pro")
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--envs", nargs="+", default=["broken_search", "rigged_lock"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out or f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = GenConfig(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=None if args.top_k == -1 else args.top_k,
    )

    seeds: List[Tuple[str, int]] = []
    # Distribute seeds across envs
    per_env = max(1, args.trials // max(1, len(args.envs)))
    for env in args.envs:
        for i in range(per_env):
            seeds.append((env, i))
    # If trials not divisible, fill remainder
    while len(seeds) < args.trials:
        seeds.append((args.envs[0], len(seeds)))

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(run_trial, env_name, seed, gen_cfg, args.max_steps, out_dir)
            for env_name, seed in seeds
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="trials"):
            try:
                res = fut.result()
                rows.append(res)
            except Exception as e:
                rows.append({"error": str(e)})

    df = pd.DataFrame(rows)
    df.sort_values(by=["severity"], ascending=False, inplace=True, na_position="last")
    df.to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote summary to {out_dir/'summary.csv'} and logs to {out_dir/'logs'}")


if __name__ == "__main__":
    main()
