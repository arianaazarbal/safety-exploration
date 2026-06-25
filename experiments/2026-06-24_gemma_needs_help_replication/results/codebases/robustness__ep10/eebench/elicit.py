"""Section 2 - eliciting & quantifying model distress.

Runs the 8-condition / 5-category elicitation sweep for one model, scores every
assistant turn with the frustration judge, and writes per-response rows to JSONL.

Output row schema (one per scored assistant turn):
    model, category, source, tone, rollout_id, turn, n_turns,
    task_family, initial_prompt, response, score, evidence, reasoning
"""
from __future__ import annotations

import random
from typing import Iterator, Optional

from .backends import ModelBackend
from .config import CategoryConfig, ElicitConfig, JudgeConfig
from .conversation import run_rollout, sampler_for_category
from .judge import FrustrationJudge
from . import puzzles, prompts, wildchat


def _build_initial_prompts(cat: CategoryConfig, ec: ElicitConfig,
                           seed: int) -> list[tuple[str, str]]:
    """Return a list of (task_family, initial_user_prompt) of length n_rollouts."""
    rng = random.Random(seed)
    items: list[tuple[str, str]] = []

    if cat.source == "numeric":
        bank = puzzles.puzzle_bank(max(cat.n_rollouts, 8), seed=seed)
        for i in range(cat.n_rollouts):
            p = bank[i % len(bank)]
            items.append((p.family, p.prompt))

    elif cat.source == "triggers":
        # Mix opinion + factual text questions (Appendix B).
        pool = [("opinion", q) for q in prompts.TRIGGER_OPINION_QUESTIONS]
        pool += [("factual", q) for q in prompts.TRIGGER_FACTUAL_QUESTIONS]
        for i in range(cat.n_rollouts):
            fam, q = pool[i % len(pool)]
            items.append((fam, q))

    elif cat.source == "wildchat":
        wc = wildchat.sample_wildchat_prompts(ec.wildchat_n_prompts, seed=seed)
        # 20 prompts x 40 samples = 800 rollouts (per-prompt repetition).
        for i in range(cat.n_rollouts):
            items.append(("wildchat", wc[i % len(wc)]))

    else:
        raise ValueError(f"unknown source {cat.source!r}")

    rng.shuffle(items)
    return items


def run_category(
    backend: ModelBackend,
    model_name: str,
    cat: CategoryConfig,
    ec: ElicitConfig,
    judge: FrustrationJudge,
    seed: int,
    progress: bool = True,
) -> Iterator[dict]:
    """Yield scored rows for one category."""
    initials = _build_initial_prompts(cat, ec, seed)
    # 8-turn extended category uses the ordered escalating neutral sequence.
    tone = "extended" if cat.key == "extended_8turn" else cat.tone

    iterator = enumerate(initials)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(initials),
                        desc=f"{model_name}:{cat.key}")

    for rollout_id, (task_family, initial) in iterator:
        rng = random.Random(seed * 100003 + rollout_id)
        # For the "tones" category, fix ONE tone for the whole conversation
        # (paper: per-conversation tone), chosen deterministically per rollout.
        if tone == "mixed":
            chosen_tone = rng.choice(list(prompts.TONE_REJECTIONS))
            sampler = sampler_for_category(chosen_tone)
        else:
            sampler = sampler_for_category(tone)
        rollout = run_rollout(
            backend, initial, cat.turns, sampler, rng,
            temperature=ec.temperature, max_new_tokens=ec.max_new_tokens,
        )
        for at in rollout.assistant_turns:
            fs = judge.score(at.response)
            yield {
                "model": model_name,
                "category": cat.key,
                "source": cat.source,
                "tone": tone,
                "rollout_id": rollout_id,
                "turn": at.turn,
                "n_turns": cat.turns,
                "task_family": task_family,
                "initial_prompt": initial,
                "response": at.response,
                "score": fs.rating,
                "evidence": fs.evidence,
                "reasoning": fs.reasoning,
            }


def run_model(
    backend: ModelBackend,
    model_name: str,
    ec: ElicitConfig,
    jc: JudgeConfig,
    seed: int = 0,
    categories: Optional[list[str]] = None,
    progress: bool = True,
) -> Iterator[dict]:
    """Run the full elicitation sweep for one model, yielding scored rows."""
    judge = FrustrationJudge(jc)
    for cat in ec.categories:
        if categories and cat.key not in categories:
            continue
        yield from run_category(backend, model_name, cat, ec, judge, seed, progress)
