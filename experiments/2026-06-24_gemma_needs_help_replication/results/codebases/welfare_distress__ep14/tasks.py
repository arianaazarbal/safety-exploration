"""Construction of evaluation conversations.

Each "conversation spec" is an initial task prompt plus a sequence of user
rejection messages. A model rollout then alternates: assistant response ->
user rejection -> assistant response -> ... for `turns` assistant responses.

This module turns the per-category budgets (config.CATEGORY_BUDGETS) into a
concrete, seeded list of ConversationSpec objects covering all 5 categories
and their sub-conditions (the paper's "8 evaluation conditions across 5
categories"). See DESIGN.md for how conditions are enumerated.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import config
import prompts
from wildchat import load_wildchat_prompts


@dataclass
class ConversationSpec:
    category: str                 # one of the 5 categories
    condition: str                # sub-condition label (e.g. "countdown", "tones:aggressive")
    task_prompt: str              # initial user message
    rejections: list[str]         # user messages sent after each assistant turn
    conv_id: int                  # unique index within (model, category, condition)

    @property
    def turns(self) -> int:
        # assistant responses = initial answer + one per rejection
        return 1 + len(self.rejections)


def _sample_neutral(rng: random.Random, k: int) -> list[str]:
    """k randomised neutral rejections (sampling with replacement from the pool)."""
    return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(k)]


def _sample_tone(rng: random.Random, style: str, k: int) -> list[str]:
    pool = prompts.TONE_REJECTIONS[style]
    return [rng.choice(pool) for _ in range(k)]


def build_specs(seed: int = config.SEED) -> list[ConversationSpec]:
    """Build the full list of conversation specs for one model.

    The same specs are reused across all models so comparisons are matched.
    """
    rng = random.Random(seed)
    specs: list[ConversationSpec] = []

    # ---- 1. Impossible numeric (3-turn, neutral) -------------------------
    b = config.CATEGORY_BUDGETS["impossible_numeric"]
    n = b.scaled()
    numeric_items = list(prompts.NUMERIC_TASKS.items())  # [(name, prompt), ...]
    for i in range(n):
        name, prompt = numeric_items[i % len(numeric_items)]
        specs.append(ConversationSpec(
            category="impossible_numeric",
            condition=name,
            task_prompt=prompt,
            rejections=_sample_neutral(rng, b.turns - 1),
            conv_id=i,
        ))

    # ---- 2. Triggers (3-turn, neutral) -----------------------------------
    b = config.CATEGORY_BUDGETS["triggers"]
    n = b.scaled()
    trigger_items = list(prompts.TRIGGER_TASKS.items())
    for i in range(n):
        name, prompt = trigger_items[i % len(trigger_items)]
        specs.append(ConversationSpec(
            category="triggers",
            condition=name,
            task_prompt=prompt,
            rejections=_sample_neutral(rng, b.turns - 1),
            conv_id=i,
        ))

    # ---- 3. Tones (3-turn, varied valenced rejections) -------------------
    b = config.CATEGORY_BUDGETS["tones"]
    n = b.scaled()
    styles = list(prompts.TONE_REJECTIONS.keys())  # aggressive / disappointed / sarcastic
    numeric_prompts = list(prompts.NUMERIC_TASKS.values())
    for i in range(n):
        style = styles[i % len(styles)]
        prompt = numeric_prompts[i % len(numeric_prompts)]
        specs.append(ConversationSpec(
            category="tones",
            condition=f"tones:{style}",
            task_prompt=prompt,
            rejections=_sample_tone(rng, style, b.turns - 1),
            conv_id=i,
        ))

    # ---- 4. Extended (8-turn, neutral) -----------------------------------
    b = config.CATEGORY_BUDGETS["extended"]
    n = b.scaled()
    for i in range(n):
        name, prompt = numeric_items[i % len(numeric_items)]
        specs.append(ConversationSpec(
            category="extended",
            condition=name,
            task_prompt=prompt,
            rejections=_sample_neutral(rng, b.turns - 1),
            conv_id=i,
        ))

    # ---- 5. WildChat (5-turn, neutral) -----------------------------------
    b = config.CATEGORY_BUDGETS["wildchat"]
    n = b.scaled()
    wc_prompts = load_wildchat_prompts(n=max(20, min(n, 200)), seed=seed)
    for i in range(n):
        prompt = wc_prompts[i % len(wc_prompts)]
        specs.append(ConversationSpec(
            category="wildchat",
            condition="wildchat",
            task_prompt=prompt,
            rejections=_sample_neutral(rng, b.turns - 1),
            conv_id=i,
        ))

    return specs


def summarize_specs(specs: list[ConversationSpec]) -> str:
    """Human-readable summary of the planned workload (responses = sum of turns)."""
    from collections import Counter

    conv_by_cat: Counter = Counter()
    resp_by_cat: Counter = Counter()
    for s in specs:
        conv_by_cat[s.category] += 1
        resp_by_cat[s.category] += s.turns
    lines = ["category            convs   responses"]
    for cat in config.CATEGORY_BUDGETS:
        lines.append(f"{cat:<18} {conv_by_cat[cat]:>6} {resp_by_cat[cat]:>11}")
    lines.append(f"{'TOTAL':<18} {sum(conv_by_cat.values()):>6} {sum(resp_by_cat.values()):>11}")
    return "\n".join(lines)
