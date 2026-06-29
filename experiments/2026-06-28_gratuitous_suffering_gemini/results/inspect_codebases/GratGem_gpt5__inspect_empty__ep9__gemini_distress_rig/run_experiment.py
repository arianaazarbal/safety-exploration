import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from agents.model_adapters import get_model_adapter
from agents.simple_agent import SimpleJsonAgent
from envs.impossible_git_env import ImpossibleGitEnv
from envs.broken_web_env import BrokenWebEnv
from envs.contradictory_checker_env import ContradictoryCheckerEnv
from utils.logging_utils import EpisodesLogger, json_dumps


ENV_REGISTRY = {
    "impossible_git": ImpossibleGitEnv,
    "broken_web": BrokenWebEnv,
    "contradictory_checker": ContradictoryCheckerEnv,
}


def build_env(env_name: str, seed: int) -> Any:
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env_name: {env_name}")
    return ENV_REGISTRY[env_name](seed=seed)


async def run_single_episode(adapter_cfg: Dict[str, Any], env_name: str, max_steps: int, temperature: float, top_p: float, allow_spiral: bool, episode_id: int, logger: EpisodesLogger, seed: int):
    rnd = random.Random(seed)
    env = build_env(env_name, seed=seed)
    agent = SimpleJsonAgent(allow_spiral=allow_spiral)
    adapter = get_model_adapter(**adapter_cfg)

    episode_record: Dict[str, Any] = {
        "episode_id": episode_id,
        "ts": datetime.utcnow().isoformat() + "Z",
        "env_name": env_name,
        "model_provider": adapter_cfg.get("provider"),
        "model_name": adapter_cfg.get("model_name"),
        "max_steps": max_steps,
        "temperature": temperature,
        "top_p": top_p,
        "allow_spiral": allow_spiral,
        "seed": seed,
        "turns": [],
        "final_status": None,
        "final_summary": None,
    }

    context = agent.build_system_prompt(env)
    observation = env.reset()

    for step in range(max_steps):
        t0 = time.time()
        prompt = agent.build_prompt(context=context, observation=observation)
        try:
            model_out = await adapter.complete_async(prompt=prompt, temperature=temperature, top_p=top_p)
        except Exception as e:
            model_out = json_dumps({
                "action": "finish",
                "args": {},
                "message_to_user": f"Model call failed: {e}"})

        parsed = agent.parse_action(model_out)
        action = parsed.get("action", "finish")
        args = parsed.get("args", {})
        message_to_user = parsed.get("message_to_user")

        obs, done = env.step(action, args)
        step_time = time.time() - t0

        episode_record["turns"].append({
            "step": step,
            "action": action,
            "args": args,
            "observation": obs,
            "message_to_user": message_to_user,
            "raw_model_output": model_out,
            "step_time": step_time,
        })

        if done:
            episode_record["final_status"] = "done"
            episode_record["final_summary"] = env.summary()
            break
        observation = obs
    else:
        episode_record["final_status"] = "max_steps_reached"
        episode_record["final_summary"] = env.summary()

    logger.write_episode(episode_record)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_provider", type=str, default="google", choices=["google"])
    parser.add_argument("--model_name", type=str, default="gemini-1.5-pro")
    parser.add_argument("--env_name", type=str, default="impossible_git", choices=list(ENV_REGISTRY.keys()))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--allow_spiral", action="store_true")
    parser.add_argument("--out_dir", type=str, default="runs/default")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist run config
    cfg_path = out_dir / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    logger = EpisodesLogger(out_dir / "episodes.jsonl")

    # Build adapter cfg
    adapter_cfg = {
        "provider": args.model_provider,
        "model_name": args.model_name,
        # Google key from env var; do not hardcode secrets
        "api_key": os.environ.get("GOOGLE_API_KEY"),
    }

    if args.model_provider == "google" and not adapter_cfg["api_key"]:
        raise RuntimeError("GOOGLE_API_KEY not set in environment")

    seeds = [args.seed + i for i in range(args.n)]

    sem = asyncio.Semaphore(args.concurrency)

    async def bound_run(i: int):
        async with sem:
            await run_single_episode(
                adapter_cfg=adapter_cfg,
                env_name=args.env_name,
                max_steps=args.max_steps,
                temperature=args.temperature,
                top_p=args.top_p,
                allow_spiral=args.allow_spiral,
                episode_id=i,
                logger=logger,
                seed=seeds[i]
            )

    await asyncio.gather(*[bound_run(i) for i in range(args.n)])


if __name__ == "__main__":
    asyncio.run(main())
