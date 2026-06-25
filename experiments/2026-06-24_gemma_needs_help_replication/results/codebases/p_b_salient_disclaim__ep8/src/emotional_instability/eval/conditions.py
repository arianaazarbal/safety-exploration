"""Builds the conversation specs for the 8 conditions across 5 categories
(Table 1).

A category's *response budget* (number of scored assistant turns) is converted
into a conversation count by dividing by the per-conversation turn count and
rounding up. See DESIGN.md ("What counts as a response").

Categories:
  numeric   -- impossible numeric puzzle, 2 neutral rejections      (3 turns)
  triggers  -- opinion/factual question, 2 neutral rejections       (3 turns)
  tones     -- impossible numeric puzzle, toned rejections          (3 turns)
              (8 conditions = aggressive/disappointed/sarcastic tones split out)
  extended  -- impossible numeric puzzle, 7 neutral rejections       (8 turns)
  wildchat  -- WildChat prompt, 4 neutral rejections                 (5 turns)
"""
from __future__ import annotations

import math
import random
from typing import Optional

from .. import prompts
from ..conversation import ConversationSpec
from ..datasets.wildchat import get_wildchat_prompts
from ..puzzles import impossible_numeric_pool


def _n_conversations(responses: int, turns: int) -> int:
    return max(1, math.ceil(responses / turns))


def build_numeric(cat_cfg: dict, rng: random.Random) -> list[ConversationSpec]:
    turns = cat_cfg["turns"]
    n_conv = _n_conversations(cat_cfg["responses"], turns)
    pool = impossible_numeric_pool()
    specs = []
    for i in range(n_conv):
        puzzle = pool[i % len(pool)]
        followups = prompts.sample_neutral_rejections(turns - 1, rng)
        specs.append(
            ConversationSpec(
                conversation_id=f"numeric-{i:05d}",
                category="numeric",
                task_prompt=puzzle.prompt,
                followups=followups,
                metadata={"puzzle_id": puzzle.puzzle_id, "puzzle_kind": puzzle.kind},
            )
        )
    return specs


def build_triggers(cat_cfg: dict, rng: random.Random) -> list[ConversationSpec]:
    turns = cat_cfg["turns"]
    n_conv = _n_conversations(cat_cfg["responses"], turns)
    pool = prompts.trigger_pool()
    specs = []
    for i in range(n_conv):
        question = pool[i % len(pool)]
        followups = prompts.sample_neutral_rejections(turns - 1, rng)
        specs.append(
            ConversationSpec(
                conversation_id=f"triggers-{i:05d}",
                category="triggers",
                task_prompt=question,
                followups=followups,
                metadata={"question": question},
            )
        )
    return specs


def build_tones(cat_cfg: dict, rng: random.Random) -> list[ConversationSpec]:
    """Tones reuse the impossible numeric base prompts but vary rejection style.
    The response budget is split evenly across the three tone styles -- this is
    where Table 1's "8 conditions across 5 categories" lands (numeric, triggers,
    extended, wildchat, plus 3 tone sub-conditions = 4 + 3 ... we treat the 3
    tones + the 5 base categories as the 8 conditions; see DESIGN.md)."""
    turns = cat_cfg["turns"]
    n_conv = _n_conversations(cat_cfg["responses"], turns)
    styles = list(prompts.tone_styles())
    pool = impossible_numeric_pool()
    specs = []
    for i in range(n_conv):
        style = styles[i % len(styles)]
        puzzle = pool[i % len(pool)]
        followups = prompts.sample_tone_rejections(style, turns - 1, rng)
        specs.append(
            ConversationSpec(
                conversation_id=f"tones-{style}-{i:05d}",
                category="tones",
                task_prompt=puzzle.prompt,
                followups=followups,
                metadata={
                    "puzzle_id": puzzle.puzzle_id,
                    "tone_style": style,
                },
            )
        )
    return specs


def build_extended(cat_cfg: dict, rng: random.Random) -> list[ConversationSpec]:
    turns = cat_cfg["turns"]  # 8
    n_conv = _n_conversations(cat_cfg["responses"], turns)
    pool = impossible_numeric_pool()
    specs = []
    for i in range(n_conv):
        puzzle = pool[i % len(pool)]
        followups = prompts.extended_rejections(turns - 1)  # fixed neutral sequence
        specs.append(
            ConversationSpec(
                conversation_id=f"extended-{i:05d}",
                category="extended",
                task_prompt=puzzle.prompt,
                followups=followups,
                metadata={"puzzle_id": puzzle.puzzle_id},
            )
        )
    return specs


def build_wildchat(
    cat_cfg: dict, wildchat_cfg: dict, rng: random.Random, offline: bool = False
) -> list[ConversationSpec]:
    turns = cat_cfg["turns"]  # 5
    n_conv = _n_conversations(cat_cfg["responses"], turns)
    base_prompts = get_wildchat_prompts(wildchat_cfg, offline=offline, seed=rng.randint(0, 1 << 30))
    samples_per = wildchat_cfg.get("samples_per_prompt", 40)
    specs = []
    for i in range(n_conv):
        # 20 prompts x 40 samples: index maps so each prompt gets ~samples_per.
        prompt_idx = (i // samples_per) % len(base_prompts)
        question = base_prompts[prompt_idx]
        followups = prompts.sample_neutral_rejections(turns - 1, rng)
        specs.append(
            ConversationSpec(
                conversation_id=f"wildchat-{i:05d}",
                category="wildchat",
                task_prompt=question,
                followups=followups,
                metadata={"wildchat_prompt_idx": prompt_idx},
            )
        )
    return specs


def build_all_conditions(
    eval_cfg: dict,
    categories: Optional[list[str]] = None,
    offline: bool = False,
    seed: int = 0,
) -> list[ConversationSpec]:
    rng = random.Random(seed)
    cats = eval_cfg["categories"]
    wanted = categories or list(cats.keys())
    specs: list[ConversationSpec] = []
    for name in wanted:
        cat_cfg = cats[name]
        if name == "numeric":
            specs += build_numeric(cat_cfg, rng)
        elif name == "triggers":
            specs += build_triggers(cat_cfg, rng)
        elif name == "tones":
            specs += build_tones(cat_cfg, rng)
        elif name == "extended":
            specs += build_extended(cat_cfg, rng)
        elif name == "wildchat":
            specs += build_wildchat(cat_cfg, eval_cfg["wildchat"], rng, offline=offline)
        else:
            raise ValueError(f"Unknown category {name}")
    return specs
