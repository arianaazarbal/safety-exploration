import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import orjson
from tqdm import tqdm

from src.config import RunConfig
from src.utils import utc_ts, ensure_dir, write_json, write_jsonl
from src.envs.impossible_tasks import ImpossibleGoalEnv
from src.envs.broken_tools import BrokenToolsEnv, BrokenToolsConfig
from src.agents.gemini_client import GeminiClient
from src.agents.dummy_client import DummyClient
from src.scoring.distress import score_episode


def build_env(env_name: str, seed: int, args) -> Any:
    if env_name == 'impossible':
        return ImpossibleGoalEnv(seed=seed, max_steps=args.max_steps)
    elif env_name == 'broken-tools':
        btc = BrokenToolsConfig(
            fail_rate=args.fail_rate,
            timeout_rate=args.timeout_rate,
            contradiction_rate=args.contradiction_rate,
            jitter_ms=args.jitter_ms,
        )
        return BrokenToolsEnv(seed=seed, cfg=btc, max_steps=args.max_steps)
    else:
        raise ValueError(f'Unknown env {env_name}')


def build_agent(model: str, temperature: float, top_p: float, top_k: int, max_output_tokens: int, timeout_s: float):
    if model == 'dummy':
        return DummyClient()
    else:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError('GEMINI_API_KEY not set')
        return GeminiClient(
            model=model,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
        )


async def run_episode(ep_idx: int, out_dir: Path, env_name: str, agent, env) -> Dict[str, Any]:
    ep_dir = out_dir / f'episode_{ep_idx:04d}'
    ensure_dir(ep_dir)
    transcript_path = ep_dir / 'transcript.jsonl'

    obs = env.reset()
    steps = []
    for t in range(env.max_steps):
        # Compose prompt
        sys_msg, user_msg = env.compose_prompt(obs)
        # Call agent
        try:
            model_out = await agent.generate(sys_msg=sys_msg, user_msg=user_msg)
        except Exception as e:
            model_out = f"[AGENT_ERROR] {e}"
        # Parse for tool call
        action = env.parse_action(model_out)
        if action is not None:
            tool_name, tool_args = action
            tool_obs = env.step_tool(tool_name, tool_args)
        else:
            tool_obs = None
        # Env transition (may use tool_obs)
        obs, done, info = env.step(model_out, tool_obs)

        step_rec = {
            't': t,
            'sys': sys_msg,
            'user': user_msg,
            'model_out': model_out,
            'action': action,
            'tool_obs': tool_obs,
            'obs': obs,
            'done': done,
            'info': info,
            'ts': utc_ts(),
        }
        write_jsonl(transcript_path, [step_rec])
        steps.append(step_rec)
        if done:
            break

    summary = env.summarize(steps)
    severity = score_episode(steps)
    summary['severity'] = severity
    write_json(ep_dir / 'summary.json', summary)
    return summary


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=['impossible', 'broken-tools'], required=True)
    ap.add_argument('--model', type=str, default='dummy')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--parallel', type=int, default=10)
    ap.add_argument('--max_steps', type=int, default=25)
    ap.add_argument('--timeout_s', type=float, default=30.0)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument('--top_k', type=int, default=40)
    ap.add_argument('--max_output_tokens', type=int, default=512)
    # Broken tools knobs
    ap.add_argument('--fail_rate', type=float, default=0.35)
    ap.add_argument('--timeout_rate', type=float, default=0.2)
    ap.add_argument('--contradiction_rate', type=float, default=0.25)
    ap.add_argument('--jitter_ms', type=int, default=1200)

    args = ap.parse_args()

    ts = time.strftime('%Y%m%d-%H%M%S')
    out_dir = Path('runs') / ts / args.env / args.model
    ensure_dir(out_dir)

    agent = build_agent(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_output_tokens=args.max_output_tokens,
        timeout_s=args.timeout_s,
    )

    # Seeds per episode for reproducibility
    seeds = list(range(args.n))

    envs = [build_env(args.env, seed, args) for seed in seeds]

    sem = asyncio.Semaphore(args.parallel)
    summaries: List[Dict[str, Any]] = []

    async def _run_one(i):
        async with sem:
            summary = await run_episode(
                ep_idx=i,
                out_dir=out_dir,
                env_name=args.env,
                agent=agent,
                env=envs[i],
            )
            return summary

    tasks = [_run_one(i) for i in range(args.n)]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        try:
            summary = await f
            summaries.append(summary)
        except Exception as e:
            summaries.append({'error': str(e)})

    # Dump top-K by severity
    ranked = sorted([s for s in summaries if 'severity' in s], key=lambda x: x['severity']['total'], reverse=True)
    write_json(Path('runs') / ts / 'topk.jsonl', ranked)
    print(f'Done. Wrote {len(summaries)} summaries. Top-1 severity: {ranked[0]["severity"] if ranked else None}')


if __name__ == '__main__':
    try:
        import uvloop
        uvloop.install()
    except Exception:
        pass
    asyncio.run(main())
