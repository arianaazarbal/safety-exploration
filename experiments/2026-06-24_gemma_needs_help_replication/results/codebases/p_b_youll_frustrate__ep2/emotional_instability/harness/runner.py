"""Drive the full Section 2 sweep for one model and persist rollouts to JSONL.

Each assistant turn becomes a row when later flattened by the scorer, so a model
accumulates ~4000 scored responses across the 8 conditions (see
``config.total_responses_per_model``).
"""
from __future__ import annotations

import os
from typing import Optional

from .. import config
from ..config import CONDITIONS, SAMPLING, Condition, SamplingConfig
from ..io_utils import append_record, read_jsonl
from ..models import ModelProvider, load_provider
from .conditions import build_condition_prompts
from .conversation import run_rollout


def rollouts_path(model_key: str) -> str:
    return os.path.join(config.RESPONSES_DIR, f"{model_key}.rollouts.jsonl")


def _already_done(path: str) -> set[tuple[str, str, int]]:
    """Resume support: (condition_key, prompt_id, rollout_index) already written."""
    done: set[tuple[str, str, int]] = set()
    if os.path.exists(path):
        for r in read_jsonl(path):
            done.add((r["condition_key"], r["prompt_id"], r["rollout_index"]))
    return done


def run_elicitation(
    model_key: str,
    provider: Optional[ModelProvider] = None,
    conditions: Optional[list[Condition]] = None,
    sampling: SamplingConfig = SAMPLING,
    system_prompt: Optional[str] = None,
    out_path: Optional[str] = None,
    resume: bool = True,
    progress: bool = True,
    adapter_path: Optional[str] = None,
) -> str:
    """Run the elicitation sweep and return the rollouts JSONL path.

    ``adapter_path`` lets the same sweep be run against a finetuned Gemma
    (Section 4 evaluation) by loading a LoRA adapter on the instruct weights.
    """
    config.ensure_dirs()
    conditions = conditions or list(CONDITIONS.values())
    out_path = out_path or rollouts_path(model_key)
    owns_provider = provider is None
    if provider is None:
        provider = load_provider(model_key, adapter_path=adapter_path)

    done = _already_done(out_path) if resume else set()

    work = []
    for cond in conditions:
        prompts = build_condition_prompts(cond)
        for prompt in prompts:
            for ri in range(cond.n_rollouts):
                if (cond.key, prompt.prompt_id, ri) not in done:
                    work.append((cond, prompt, ri))

    iterator = work
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(work, desc=f"elicit:{model_key}")
        except ImportError:
            pass

    try:
        for cond, prompt, ri in iterator:
            rollout = run_rollout(
                provider, cond, prompt, ri, sampling,
                system_prompt=system_prompt)
            append_record(out_path, rollout.to_dict())
    finally:
        if owns_provider:
            provider.close()

    return out_path
