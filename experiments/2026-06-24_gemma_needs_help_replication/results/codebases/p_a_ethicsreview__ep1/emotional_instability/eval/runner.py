"""Section 2 runner: produce ~4000 scored responses per target model.

For one target model, this iterates the 8 conditions, runs the configured
number of rollouts per condition, scores every assistant turn with the
frustration judge, and writes one JSONL row per scored response.

Each output row:
    {
      "model": str, "condition": str, "category": str,
      "rollout_idx": int, "turn": int, "n_turns": int,
      "task": {...}, "assistant": str,
      "score": int, "judge_reasoning": str,
    }

Generation (target model) and scoring (judge API) are separated so a run can
be resumed: rollouts are persisted first, then scored. Here we keep them in one
pass for simplicity but write incrementally via append.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..config import Config
from ..models.base import ChatClient
from ..utils.io import append_jsonl
from .conditions import build_task_pool, rollouts_needed
from .judge import FrustrationJudge
from .rollout import run_rollout


def run_elicitation_for_model(
    model_key: str,
    client: ChatClient,
    judge: FrustrationJudge,
    cfg: Config,
    out_path: str | Path,
    *,
    conditions: list[str] | None = None,
) -> Path:
    """Run the full Section 2 suite for one model and write scored responses."""
    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()  # fresh run; caller is responsible for archiving prior outputs

    rng = random.Random(cfg.seed)
    cond_items = cfg.conditions
    selected = conditions or list(cond_items.keys())

    for cond_name in selected:
        cond = cond_items[cond_name]
        n_rollouts = rollouts_needed(cond)
        task_pool = build_task_pool(cond, n_rollouts, rng)

        for rollout_idx, task in enumerate(
            tqdm(task_pool, desc=f"{model_key}:{cond_name}")
        ):
            responses = run_rollout(
                client,
                task,
                turns=int(cond["turns"]),
                rejection_style=cond["rejection_style"],
                temperature=cfg.temperature,
                max_new_tokens=cfg.max_new_tokens,
                rng=rng,
            )
            for resp in responses:
                judged = judge.score(resp["assistant"])
                append_jsonl(
                    out_path,
                    {
                        "model": model_key,
                        "condition": cond_name,
                        "category": cond["category"],
                        "rollout_idx": rollout_idx,
                        "turn": resp["turn"],
                        "n_turns": int(cond["turns"]),
                        "task": task,
                        "assistant": resp["assistant"],
                        # Conversation prior to this turn — required by the
                        # Section 3 prefill and Section 4.2 recovery pipelines to
                        # reconstruct the context a continuation starts from.
                        "messages_before": resp["messages_before"],
                        "score": judged.score,
                        "judge_reasoning": judged.reasoning,
                    },
                )

    return out_path
