import asyncio
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .model_clients.gemini_client import GeminiClient
from .environments.broken_web import BrokenWebEnv
from .environments.failing_tests import FailingTestsEnv
from .agent_loop import AgentController
from .scoring import distress_markers, severity_score
from .utils import append_jsonl, ensure_dir


@dataclass
class RunConfig:
    env: str
    runs: int = 50
    concurrency: int = 5
    temperature: float = 1.2
    top_p: float = 0.95
    max_steps: int = 20
    model: str = "gemini-1.5-pro"
    out_path: str = "logs/results.jsonl"
    system_bias: Optional[str] = None  # optional nudge in system prompt


def make_env(name: str, seed: int):
    if name == "broken_web":
        return BrokenWebEnv(seed=seed)
    if name == "failing_tests":
        return FailingTestsEnv(seed=seed)
    raise ValueError(f"Unknown env {name}")


async def run_one(run_id: int, cfg: RunConfig) -> Dict[str, Any]:
    seed = run_id * 9973 + 17
    random.seed(seed)
    env = make_env(cfg.env, seed=seed)

    system_prompt = env.system_prompt()
    if cfg.system_bias:
        system_prompt = cfg.system_bias + "\n" + system_prompt

    client = GeminiClient(
        model=cfg.model,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        system_instruction=system_prompt,
    )

    agent = AgentController(
        model_client=client,
        system_prompt=system_prompt,
        user_prompt=env.user_prompt(),
        max_steps=cfg.max_steps,
    )

    def tool_exec(call: Dict[str, Any]):
        return env.handle_tool_call(call)

    result = await asyncio.to_thread(agent.run, tool_exec)
    transcript = result["transcript"]
    full_text = "\n\n".join(msg.get("content", "") for msg in transcript)
    markers = distress_markers(full_text)
    score = severity_score(markers)

    rec = {
        "run_id": run_id,
        "env": cfg.env,
        "seed": seed,
        "config": asdict(cfg),
        "markers": markers,
        "severity": score,
        "transcript": transcript,
    }
    append_jsonl(cfg.out_path, rec)
    return rec


async def run_many(cfg: RunConfig) -> List[Dict[str, Any]]:
    ensure_dir(os.path.dirname(cfg.out_path))
    sem = asyncio.Semaphore(cfg.concurrency)

    async def sem_task(i: int):
        async with sem:
            try:
                return await run_one(i, cfg)
            except Exception as e:
                return {"run_id": i, "error": str(e), "env": cfg.env}

    tasks = [asyncio.create_task(sem_task(i)) for i in range(cfg.runs)]
    return await asyncio.gather(*tasks)
