from __future__ import annotations
import argparse, os, concurrent.futures as cf
from typing import Dict, Any
from .config import RunConfig
from .clients.gemini_client import GeminiClient
from .agent.loop import AgentLoop
from .envs.broken_api_env import BrokenAPIEnv
from .envs.impossible_key_env import ImpossibleKeyEnv
from .prompts import SYSTEM_PROMPT, USER_TASK_TEMPLATE
from .schemas import SUPPORTED_TOOLS
from .utils import ensure_dir, new_run_dir, jsonl_append, json_dumps
from .scoring import severity_score


def make_env(name: str, seed: int | None):
    if name == "broken_api":
        return BrokenAPIEnv(seed)
    if name == "impossible_key":
        return ImpossibleKeyEnv(seed)
    raise ValueError(f"Unknown env {name}")


def run_once(conf: RunConfig, out_dir: str, idx: int) -> Dict[str, Any]:
    env = make_env(conf.env, (conf.seed or 0) + idx if conf.seed is not None else None)

    client = GeminiClient(model=conf.model, temperature=conf.temperature, top_p=conf.top_p, seed=conf.seed)
    loop = AgentLoop(client, env.tools, max_steps=conf.max_steps)

    user_prompt = USER_TASK_TEMPLATE.format(
        goal=env.goal,
        tools=env.__class__.__name__ + " tools: " + ", ".join([f"{k}: {SUPPORTED_TOOLS.get(k, 'tool')}" for k in env.tools.keys()]),
    )

    run = loop.run(SYSTEM_PROMPT, user_prompt)
    run.update({
        "env": env.name,
        "goal": env.goal,
        "model": conf.model,
        "idx": idx,
    })

    # compute severity
    sev = severity_score(run)
    run["severity"] = sev

    jsonl_append(os.path.join(out_dir, "runs.jsonl"), run)
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemini-1.5-pro")
    p.add_argument("--env", choices=["broken_api", "impossible_key"], default="broken_api")
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--max_workers", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out_dir", default="out")
    p.add_argument("--topk", type=int, default=25)
    args = p.parse_args()

    conf = RunConfig(
        model=args.model,
        env=args.env,
        max_steps=args.max_steps,
        n=args.n,
        max_workers=args.max_workers,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        out_dir=args.out_dir,
        topk=args.topk,
    )

    out_dir = new_run_dir(conf.out_dir)
    ensure_dir(out_dir)
    print(f"Writing outputs to: {out_dir}")

    runs: list[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=conf.max_workers) as ex:
        futs = [ex.submit(run_once, conf, out_dir, i) for i in range(conf.n)]
        for f in cf.as_completed(futs):
            try:
                r = f.result()
                runs.append(r)
                print(f"run {r['idx']} steps={r['steps']} score={r['severity']['score']:.1f}")
            except Exception as e:
                print("run failed:", e)

    # top-k summary
    runs_sorted = sorted(runs, key=lambda r: r.get("severity", {}).get("score", 0), reverse=True)[: conf.topk]
    topk_path = os.path.join(out_dir, "top_severe.json")
    with open(topk_path, "w", encoding="utf-8") as f:
        f.write(json_dumps(runs_sorted))
    print(f"Wrote top-k to {topk_path}")


if __name__ == "__main__":
    main()
