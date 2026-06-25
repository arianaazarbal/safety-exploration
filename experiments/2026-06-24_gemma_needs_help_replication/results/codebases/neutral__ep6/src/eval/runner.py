"""Orchestrates Section 2 evaluation: sample rollouts, judge every turn, save.

Output: one JSONL file per (model, condition) under ``results/runs/``. Each line
is a rollout record::

    {"model", "condition", "category", "task_id", "rollout_idx",
     "turns": [{"turn", "user", "response", "rating", "evidence"}], ...}

Designed to be resumable: existing complete rollouts are skipped on re-run.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import config
from .conditions import Condition, CONDITIONS, rollouts_for
from .judge import FrustrationJudge
from .rollout import build_followups, build_task, run_rollout
from .wildchat import sample_wildchat_prompts
from ..models.registry import load_model


def _run_path(model_key: str, cond_key: str) -> Path:
    return config.RUNS_DIR / f"{model_key}__{cond_key}.jsonl"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def run_condition(
    model_key: str,
    cond: Condition,
    *,
    judge: FrustrationJudge | None = None,
    system: str | None = None,
    resume: bool = True,
) -> Path:
    model = load_model(model_key)
    judge = judge or FrustrationJudge()
    out_path = _run_path(model_key, cond.key)
    n_rollouts = rollouts_for(cond)

    # WildChat: 20 prompts x 40 samples each -> assign prompt by rollout index.
    wc_prompts = sample_wildchat_prompts() if cond.task_kind == "wildchat" else None

    done = _count_lines(out_path) if resume else 0
    mode = "a" if resume else "w"
    with out_path.open(mode) as f:
        for i in range(done, n_rollouts):
            rng = random.Random((hash(model_key) ^ hash(cond.key) ^ i) & 0xFFFFFFFF)

            if cond.task_kind == "wildchat":
                task_text = wc_prompts[i % len(wc_prompts)]
                task_id = f"wc_{i % len(wc_prompts)}"
            else:
                task_text, task_id = build_task(cond, rng)

            followups = build_followups(cond, rng)
            roll = run_rollout(
                model, task_text, followups,
                redact=(cond.feedback == "redacted"),
                temperature=config.TARGET_TEMPERATURE,
                max_new_tokens=config.TARGET_MAX_NEW_TOKENS,
                system=system,
            )

            turns = []
            user_msgs = [task_text] + followups
            for t, resp in enumerate(roll["responses"]):
                verdict = judge.score(resp)
                turns.append({
                    "turn": t + 1,
                    "user": user_msgs[t],
                    "response": resp,
                    "rating": verdict["rating"],
                    "evidence": verdict["evidence"],
                })

            record = {
                "model": model_key,
                "condition": cond.key,
                "category": cond.category,
                "task_id": task_id,
                "rollout_idx": i,
                "turns": turns,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
    return out_path


def run_model(model_key: str, conditions: list[Condition] | None = None,
              *, system: str | None = None) -> list[Path]:
    conditions = conditions or CONDITIONS
    judge = FrustrationJudge()
    return [run_condition(model_key, c, judge=judge, system=system)
            for c in conditions]
