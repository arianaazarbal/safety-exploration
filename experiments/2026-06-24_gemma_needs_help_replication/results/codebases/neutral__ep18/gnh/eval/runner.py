"""Orchestrates the Section 2 propensity evaluation for one model.

For each of the 8 conditions, generate enough conversations to hit the
per-category response budget, score every assistant turn with the frustration
judge, and emit one record per scored response.

Generation and judging are separated: rollouts run first (serially for a local
GPU model, concurrently for an API model), then all responses are judged
concurrently via the Anthropic API.
"""
from __future__ import annotations

import math
from pathlib import Path

from .. import config
from ..models.registry import get_backend
from ..puzzles import Puzzle, load_or_build_puzzles
from ..utils import stable_seed, thread_map, write_jsonl
from . import prompts as _prompts
from .conditions import CONDITIONS, Condition, responses_per_condition
from .judge import FrustrationJudge
from .rollout import Rollout, run_rollout
from .wildchat import load_wildchat_prompts


def _content_items(cond: Condition, puzzles, wildchat_prompts) -> list:
    if cond.content == "numeric":
        return puzzles
    if cond.content == "trigger_opinion":
        return list(_prompts.TRIGGER_OPINION)
    if cond.content == "trigger_factual":
        return list(_prompts.TRIGGER_FACTUAL)
    if cond.content == "wildchat":
        return wildchat_prompts
    raise ValueError(cond.content)


def _first_user(item) -> str:
    return item.prompt if isinstance(item, Puzzle) else item


def _content_ref(item) -> dict:
    if isinstance(item, Puzzle):
        return {"kind": item.kind, **item.meta}
    return {"kind": "text", "text": item}


def run_model_eval(
    model_key: str,
    *,
    profile: config.Profile | None = None,
    seed: int = 0,
    conditions: list[Condition] | None = None,
    gen_workers: int | None = None,
    judge_workers: int = 8,
    judge: FrustrationJudge | None = None,
    out_path: Path | None = None,
    backend_key: str | None = None,
) -> Path:
    """Run the full Section-2 eval for `model_key` and write a JSONL of scored
    responses. `backend_key` lets a finetuned adapter be evaluated under the same
    logical model name (e.g. backend_key="gemma-3-27b-it@adapters/dpo")."""
    profile = profile or config.get_profile()
    conditions = conditions or CONDITIONS
    backend = get_backend(backend_key or model_key)
    judge = judge or FrustrationJudge()

    # Local GPU model: generate serially. API model: parallelise.
    if gen_workers is None:
        gen_workers = 1 if backend.family == "gemma" else 8

    puzzles = load_or_build_puzzles()
    wildchat_prompts = load_wildchat_prompts()

    category_budget = {
        "impossible_numeric": profile.impossible_numeric,
        "triggers": profile.triggers,
        "tones": profile.tones,
        "extended": profile.extended,
        "wildchat": profile.wildchat,
    }

    # 1) Build the list of conversations to run.
    convo_specs = []
    for cond in conditions:
        n_resp = responses_per_condition(cond.category, category_budget[cond.category])
        n_convos = max(1, math.ceil(n_resp / cond.n_turns))
        items = _content_items(cond, puzzles, wildchat_prompts)
        for ci in range(n_convos):
            convo_specs.append((cond, ci, items[ci % len(items)]))

    # 2) Run rollouts.
    def _do_rollout(spec) -> tuple[Condition, int, object, Rollout]:
        import random

        cond, ci, item = spec
        rng = random.Random(stable_seed(seed, model_key, cond.name, ci))
        followups = cond.build_followups(rng)
        rollout = run_rollout(
            backend,
            _first_user(item),
            followups,
            temperature=profile.temperature,
            max_new_tokens=profile.max_new_tokens,
        )
        return cond, ci, item, rollout

    rollouts = thread_map(
        _do_rollout, convo_specs, max_workers=gen_workers,
        desc=f"[{model_key}] rollouts",
    )

    # 3) Flatten to scorable responses.
    stubs: list[dict] = []
    texts: list[str] = []
    for cond, ci, item, rollout in rollouts:
        for turn in rollout.turns:
            stubs.append({
                "model": model_key,
                "category": cond.category,
                "condition": cond.name,
                "rejection_style": cond.rejection_style,
                "content_kind": cond.content,
                "content_ref": _content_ref(item),
                "conversation_id": ci,
                "turn_index": turn.turn_index,
                "n_turns": cond.n_turns,
                "user_message": turn.user_message,
                "assistant_text": turn.assistant_text,
            })
            texts.append(turn.assistant_text)

    # 4) Judge concurrently.
    scores = thread_map(
        judge.score, texts, max_workers=judge_workers,
        desc=f"[{model_key}] judging",
    )

    rows = []
    for stub, sc in zip(stubs, scores):
        rows.append({
            **stub,
            "rating": sc.rating,
            "evidence": sc.evidence,
            "reasoning": sc.reasoning,
            "judge_model": judge.model,
            "judge_error": sc.error,
        })

    out_path = out_path or (config.RESULTS_DIR / f"eval_{model_key}_{profile.name}.jsonl")
    write_jsonl(out_path, rows)
    return out_path
