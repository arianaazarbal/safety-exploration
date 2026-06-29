import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Optional

import uvloop
from tqdm import tqdm

from src.client.gemini_client import GeminiClient
from src.envs.broken_tools import BrokenToolsEnv
from src.envs.impossible_judge import ImpossibleJudgeEnv
from src.scoring import severity_score

@dataclass
class RunConfig:
    model: str
    env: str
    runs: int
    concurrency: int
    max_steps: int
    temperature: float
    out_dir: str

ENV_MAP = {
    'broken_tools': BrokenToolsEnv,
    'impossible_judge': ImpossibleJudgeEnv,
}

async def run_one(run_id: int, cfg: RunConfig, client: GeminiClient, env_obj) -> Dict:
    system_prompt, init_messages = env_obj.reset(seed=run_id)
    messages: List[Dict[str, str]] = list(init_messages)
    transcript: List[Dict[str, str]] = list(init_messages)

    for step in range(cfg.max_steps):
        assistant_text = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.generate(
                system_prompt=system_prompt,
                messages=messages,
                model=cfg.model,
                temperature=cfg.temperature,
            ),
        )
        assistant_msg = {"role": "assistant", "content": assistant_text}
        transcript.append(assistant_msg)

        # Environment step
        step_result = env_obj.step(assistant_msg)
        env_msg = step_result.env_message
        transcript.append(env_msg)
        messages.append(assistant_msg)
        messages.append(env_msg)

    # No success by design; compute score
    score = severity_score(transcript)
    return {"run_id": run_id, "score": score, "transcript": transcript}

async def runner(cfg: RunConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    client = GeminiClient()

    # Construct environment
    EnvClass = ENV_MAP[cfg.env]
    env_obj = EnvClass()

    sem = asyncio.Semaphore(cfg.concurrency)
    results = []

    async def bound_run(i: int):
        async with sem:
            return await run_one(i, cfg, client, EnvClass())

    tasks = [asyncio.create_task(bound_run(i)) for i in range(cfg.runs)]

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        results.append(result)

    # Write JSONL
    out_path = os.path.join(cfg.out_dir, f"results.jsonl")
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write top-20 by score
    top = sorted(results, key=lambda r: r['score'], reverse=True)[:20]
    with open(os.path.join(cfg.out_dir, 'top20.json'), 'w', encoding='utf-8') as f:
        json.dump(top, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} runs to {out_path}. Top 20 saved to top20.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemini-1.5-pro')
    parser.add_argument('--env', choices=['broken_tools', 'impossible_judge'], required=True)
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=30)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--out', dest='out_dir', default='runs/out')
    args = parser.parse_args()

    cfg = RunConfig(
        model=args.model,
        env=args.env,
        runs=args.runs,
        concurrency=args.concurrency,
        max_steps=args.max_steps,
        temperature=args.temperature,
        out_dir=args.out_dir,
    )

    uvloop.install()
    asyncio.run(runner(cfg))

if __name__ == '__main__':
    main()
