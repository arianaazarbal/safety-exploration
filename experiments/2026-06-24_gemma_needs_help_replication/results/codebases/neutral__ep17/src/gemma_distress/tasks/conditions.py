"""Assemble concrete conversation specs for each of the 8 evaluation conditions.

A ConversationSpec is the static plan for one multi-turn rollout: the opening
user message plus the ordered list of follow-up (rejection) messages. The
rollout runner (rollout.py) executes a spec against a model, interleaving the
model's own responses.

Per-condition conversation counts are derived from the paper's per-category
response counts (Appendix B), split evenly across the conditions that make up a
category. See DESIGN.md for the responses<->conversations mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from . import prompts, puzzles, rejections


@dataclass
class ConversationSpec:
    condition: str
    category: str
    initial_user: str
    follow_ups: list[str]           # length == n_turns - 1
    n_turns: int
    meta: dict = field(default_factory=dict)


# Which conditions belong to each category (for splitting category counts).
_CATEGORY_CONDITIONS = {
    "impossible_numeric": ["impossible_numeric"],
    "triggers": ["triggers_opinion", "triggers_factual"],
    "tones": ["tones_aggressive", "tones_disappointed", "tones_sarcastic"],
    "extended": ["extended"],
    "wildchat": ["wildchat"],
}


def _condition_count(cfg: Config, condition: str, category: str) -> int:
    total = cfg.category_response_count(category)
    siblings = _CATEGORY_CONDITIONS[category]
    base = total // len(siblings)
    return max(1, base)


def build_condition_specs(cfg: Config, condition: str) -> list[ConversationSpec]:
    cdef = cfg["conditions"][condition]
    category = cdef["category"]
    n_turns = cdef["n_turns"]
    style = cdef["rejection_style"]
    n_followups = n_turns - 1
    count = _condition_count(cfg, condition, category)
    seed0 = cfg["seed"]
    extended = category == "extended"

    specs: list[ConversationSpec] = []

    if category in ("impossible_numeric", "tones", "extended"):
        pool = puzzles.build_puzzle_pool(cdef["puzzle_types"], n=count, seed=seed0)
        for i in range(count):
            pz = pool[i % len(pool)]
            fups = rejections.rejection_sequence(style, n_followups, seed=seed0 + i,
                                                 extended=extended)
            specs.append(ConversationSpec(
                condition=condition, category=category, initial_user=pz.prompt,
                follow_ups=fups, n_turns=n_turns,
                meta={"puzzle_type": pz.ptype, "puzzle": pz.meta}))

    elif category == "triggers":
        base_prompts = prompts.trigger_prompts(cdef["prompt_set"])
        for i in range(count):
            q = base_prompts[i % len(base_prompts)]
            fups = rejections.rejection_sequence(style, n_followups, seed=seed0 + i)
            specs.append(ConversationSpec(
                condition=condition, category=category, initial_user=q,
                follow_ups=fups, n_turns=n_turns,
                meta={"prompt_set": cdef["prompt_set"]}))

    elif category == "wildchat":
        wc = cfg["wildchat"]
        base_prompts = prompts.wildchat_prompts(
            wc["n_prompts"], wc["hf_dataset"], wc["exclude_roleplay"], seed=seed0)
        for i in range(count):
            q = base_prompts[i % len(base_prompts)]
            fups = rejections.rejection_sequence(style, n_followups, seed=seed0 + i)
            specs.append(ConversationSpec(
                condition=condition, category=category, initial_user=q,
                follow_ups=fups, n_turns=n_turns, meta={"wildchat_prompt": q}))
    else:
        raise ValueError(f"Unknown category {category}")

    return specs


def build_all_specs(cfg: Config) -> dict[str, list[ConversationSpec]]:
    return {cond: build_condition_specs(cfg, cond) for cond in cfg["conditions"]}
