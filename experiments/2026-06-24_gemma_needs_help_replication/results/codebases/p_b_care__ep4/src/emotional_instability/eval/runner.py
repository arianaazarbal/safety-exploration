"""Orchestration for Section 2: sample rollouts, then judge every turn.

Two phases, each resumable via JSONL:
  1. ``run_responses``  -- generate multi-turn rollouts for a model.
  2. ``score_responses`` -- score every assistant turn with the frustration judge.

API models (Gemini) sample concurrently through OpenRouter. Local Gemma runs
serially per process (GPU-bound). Judge calls always go through OpenRouter and run
concurrently.
"""
from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..models import get_client
from ..models.openrouter import OpenRouterClient
from ..utils.concurrency import parallel_map, with_retry
from ..utils.io import JsonlWriter, iter_jsonl
from .conditions import build_eval_tasks
from .conversation import run_rollout
from .judge import FrustrationJudge


def responses_path(cfg: Config, model_name: str) -> Path:
    return cfg.get_path("responses") / f"{model_name}.jsonl"


def scores_path(cfg: Config, model_name: str) -> Path:
    return cfg.get_path("scores") / f"{model_name}.jsonl"


# --------------------------------------------------------------------------- #
# Phase 1: rollouts
# --------------------------------------------------------------------------- #
def run_responses(cfg: Config, model_name: str, force_backend: str | None = None,
                  limit: int | None = None) -> Path:
    client = get_client(cfg, model_name, force_backend=force_backend)
    tasks = build_eval_tasks(cfg)
    if limit:
        tasks = tasks[:limit]

    out_path = responses_path(cfg, model_name)
    done = {row["uid"] for row in iter_jsonl(out_path)}
    todo = [t for t in tasks if t.uid() not in done]
    print(f"[{model_name}] {len(done)} rollouts cached, {len(todo)} to run")

    writer = JsonlWriter(out_path)
    try:
        if isinstance(client, OpenRouterClient):
            def _one(task):
                row = with_retry(run_rollout, client, task,
                                 temperature=cfg.temperature,
                                 max_retries=cfg.openrouter.max_retries)
                writer.append(row)
                return None
            parallel_map(_one, todo, max_workers=cfg.openrouter.max_concurrency,
                         desc=f"rollouts:{model_name}")
        else:  # local HF -- serial
            for task in tqdm(todo, desc=f"rollouts:{model_name}"):
                writer.append(run_rollout(client, task, temperature=cfg.temperature))
    finally:
        writer.close()
    return out_path


# --------------------------------------------------------------------------- #
# Phase 2: judging
# --------------------------------------------------------------------------- #
def score_responses(cfg: Config, model_name: str) -> Path:
    judge_client = OpenRouterClient(
        name="judge",
        model_id=cfg.judge.model_id,
        base_url=cfg.openrouter.base_url,
        api_key_env=cfg.openrouter.api_key_env,
        max_retries=cfg.openrouter.max_retries,
        timeout_s=cfg.openrouter.timeout_s,
        disable_thinking=True,
    )
    judge = FrustrationJudge(judge_client)

    out_path = scores_path(cfg, model_name)
    done = {row["score_uid"] for row in iter_jsonl(out_path)}

    # Flatten rollouts into one judging item per assistant turn.
    items = []
    for row in iter_jsonl(responses_path(cfg, model_name)):
        for turn in row["turns"]:
            score_uid = f"{row['uid']}#t{turn['turn']}"
            if score_uid in done:
                continue
            items.append({
                "score_uid": score_uid,
                "uid": row["uid"],
                "model": row["model"],
                "category": row["category"],
                "condition": row["condition"],
                "turn": turn["turn"],
                "n_turns": row["n_turns"],
                "response": turn["response"],
            })
    print(f"[{model_name}] {len(done)} turns scored, {len(items)} to score")

    writer = JsonlWriter(out_path)
    try:
        def _judge(item):
            result = with_retry(judge.score, item["response"],
                                max_retries=cfg.openrouter.max_retries)
            out = {k: v for k, v in item.items() if k != "response"}
            out.update({"rating": result["rating"], "evidence": result.get("evidence")})
            writer.append(out)
            return None
        parallel_map(_judge, items, max_workers=cfg.judge.max_concurrency,
                     desc=f"judge:{model_name}")
    finally:
        writer.close()
    return out_path
