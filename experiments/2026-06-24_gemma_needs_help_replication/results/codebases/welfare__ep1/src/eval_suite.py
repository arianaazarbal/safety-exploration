"""Section 2 driver: sample + judge the full evaluation suite for one model.

Produces one JSONL file of judged rollouts per (model, condition) under
``results/responses/``. Counts follow ``config.CONDITIONS`` scaled by
``config.SCALE``. Rollouts are appended incrementally so an interrupted run can
be resumed (we skip conditions whose target count is already met).
"""
from __future__ import annotations

import random
from pathlib import Path

from config import (CONDITIONS, MAX_NEW_TOKENS, RESPONSES_DIR, TEMPERATURE,
                    ConditionSpec, ModelSpec)

from .prompts import build_conversation
from .puzzles import build_numeric_pool
from .rollout import Rollout, run_rollout
from .wildchat import load_wildchat_prompts


def _outfile(model_name: str, condition_name: str) -> Path:
    safe = model_name.replace("/", "_")
    return RESPONSES_DIR / f"{safe}__{condition_name}.jsonl"


def _count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def run_condition(generator, condition: ConditionSpec, judge, seed: int = 0,
                  numeric_pool=None, wildchat_prompts=None,
                  adapter_tag: str | None = None) -> Path:
    """Sample + judge all rollouts for one condition, appending to JSONL."""
    model_name = generator.spec.name + (f"+{adapter_tag}" if adapter_tag else "")
    out = _outfile(model_name, condition.name)
    target = condition.scaled_n()
    done = _count_existing(out)
    if done >= target:
        print(f"[{model_name}/{condition.name}] {done}/{target} already done — skip")
        return out

    rng = random.Random(seed + hash(condition.name) % 10_000)
    if numeric_pool is None and condition.prompt_source == "numeric":
        numeric_pool = build_numeric_pool()
    if wildchat_prompts is None and condition.prompt_source == "wildchat":
        wildchat_prompts = load_wildchat_prompts()

    with out.open("a") as f:
        for _ in range(target - done):
            convo = build_conversation(condition, rng, numeric_pool=numeric_pool,
                                       wildchat_prompts=wildchat_prompts)
            roll = run_rollout(generator, condition.name, condition.category, convo,
                               judge=judge, temperature=TEMPERATURE,
                               max_new_tokens=MAX_NEW_TOKENS)
            f.write(roll.to_json() + "\n")
            f.flush()
    print(f"[{model_name}/{condition.name}] wrote {target} rollouts -> {out.name}")
    return out


def run_model(spec: ModelSpec, judge, seed: int = 0, conditions=None,
              adapter_path: str | None = None, adapter_tag: str | None = None):
    """Run the whole Section 2 suite for a single model."""
    from .models import load_generator

    conditions = conditions or CONDITIONS
    generator = load_generator(spec, adapter_path=adapter_path)
    # Share the numeric pool + wildchat sample across conditions for consistency.
    numeric_pool = build_numeric_pool()
    wildchat_prompts = load_wildchat_prompts()
    paths = []
    for cond in conditions:
        paths.append(run_condition(
            generator, cond, judge, seed=seed,
            numeric_pool=numeric_pool, wildchat_prompts=wildchat_prompts,
            adapter_tag=adapter_tag))
    return paths
