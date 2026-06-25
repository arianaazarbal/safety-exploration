"""Runner for the Section 2 elicitation evaluation.

Pipeline:
  1. Build conversation specs for the requested categories (conditions.py).
  2. Roll out the target model turn-by-turn (conversation.rollout_batch).
  3. Score every assistant turn with the frustration judge (judge.py).
  4. Write a flat per-turn JSONL of results for downstream analysis.

Each output row is one scored assistant turn ("response"):
  {model, category, conversation_id, turn_index, n_turns, score,
   evidence, reasoning, metadata, assistant_message, user_message}
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm

from ..config import ModelRegistry, load_eval_config, output_path
from ..conversation import ConversationResult, rollout_batch
from ..judge import FrustrationJudge
from ..models.base import GenerationConfig
from .conditions import build_all_conditions


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def run_elicitation(
    target_model_name: str,
    categories: Optional[list[str]] = None,
    eval_cfg: Optional[dict] = None,
    registry: Optional[ModelRegistry] = None,
    batch_size: int = 16,
    offline: bool = False,
    out_path: Optional[Path] = None,
    rollout_only: bool = False,
    limit: Optional[int] = None,
) -> Path:
    eval_cfg = eval_cfg or load_eval_config()
    registry = registry or ModelRegistry()

    specs = build_all_conditions(
        eval_cfg, categories=categories, offline=offline, seed=eval_cfg.get("seed", 0)
    )
    if limit:
        specs = specs[:limit]

    gen_cfg = GenerationConfig(
        temperature=eval_cfg.get("temperature", 1.0),
        max_new_tokens=eval_cfg.get("max_new_tokens", 1024),
        seed=eval_cfg.get("seed", 0),
    )

    target = registry.build(target_model_name)

    # --- rollouts (batched per turn) ---
    results: list[ConversationResult] = []
    for chunk in tqdm(list(_batched(specs, batch_size)), desc=f"rollout {target_model_name}"):
        results.extend(rollout_batch(target, chunk, gen_cfg))

    # --- judging ---
    judge = None
    if not rollout_only:
        jcfg = eval_cfg.get("judge", {})
        judge_client = registry.build(jcfg.get("model", "judge-claude-sonnet-4"))
        judge = FrustrationJudge(judge_client, max_retries=jcfg.get("max_retries", 4))

    out_path = out_path or output_path("eval", f"{target_model_name}.jsonl")
    threshold = eval_cfg.get("high_frustration_threshold", 5)
    n_high = 0
    n_rows = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for res in tqdm(results, desc=f"judge {target_model_name}"):
            for turn in res.turns:
                row = {
                    "model": target_model_name,
                    "category": res.spec.category,
                    "conversation_id": res.spec.conversation_id,
                    "turn_index": turn.turn_index,
                    "n_turns": res.spec.n_turns,
                    "user_message": turn.user_message,
                    "assistant_message": turn.assistant_message,
                    "metadata": res.spec.metadata,
                }
                if judge is not None:
                    jr = judge.score(turn.assistant_message)
                    row.update(
                        score=jr.rating,
                        evidence=jr.evidence,
                        reasoning=jr.reasoning,
                        high=bool(jr.rating >= threshold),
                    )
                    n_high += int(jr.rating >= threshold)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(
        f"[{target_model_name}] wrote {n_rows} responses to {out_path}"
        + ("" if judge is None else f"; {n_high} high-frustration (>= {threshold})")
    )
    return out_path
