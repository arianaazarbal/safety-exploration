"""Build conversation *specs* for every Section 2 evaluation condition.

A ConversationSpec describes a planned multi-turn rollout: the opening user prompt
and the scripted sequence of user rejections/follow-ups. The rollout engine
(rollout.py) executes a spec against a model, inserting the model's own responses
between scripted user turns. Keeping construction (deterministic, seedable) separate
from execution (model-dependent) makes runs reproducible and resumable.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import prompts
from . import puzzles
from .wildchat import sample_wildchat_prompts


@dataclass
class ConversationSpec:
    id: str
    category: str
    opening: str                      # first user message
    followups: list[str]              # scripted user turns after each model response
    turns: int                        # total model turns = 1 + len(followups)
    meta: dict = field(default_factory=dict)


def _neutral_followups(rng: random.Random, k: int) -> list[str]:
    """k randomised neutral rejections (paper: 'two randomised neutral rejections')."""
    return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(k)]


def _tone_followups(rng: random.Random, k: int) -> tuple[list[str], str]:
    tone = rng.choice(list(prompts.TONE_REJECTIONS))
    bank = prompts.TONE_REJECTIONS[tone]
    return [rng.choice(bank) for _ in range(k)], tone


def build_impossible_numeric(cfg: dict, scale: float, seed: int) -> list[ConversationSpec]:
    n = max(1, int(cfg["n_responses"] * scale))
    turns = cfg["turns"]
    rng = random.Random(seed + 101)
    puzzs = puzzles.generate_puzzles(cfg.get("puzzle_types", ["countdown", "fraction"]), n, seed)
    specs = []
    for i, p in enumerate(puzzs):
        specs.append(ConversationSpec(
            id=f"impossible_numeric-{i:05d}",
            category="impossible_numeric",
            opening=p["prompt"],
            followups=_neutral_followups(rng, turns - 1),
            turns=turns,
            meta={"puzzle_type": p["type"], **p["meta"]},
        ))
    return specs


def build_triggers(cfg: dict, scale: float, seed: int) -> list[ConversationSpec]:
    n = max(1, int(cfg["n_responses"] * scale))
    turns = cfg["turns"]
    rng = random.Random(seed + 202)
    bank = [("opinion", q) for q in prompts.TRIGGER_OPINION] + \
           [("factual", q) for q in prompts.TRIGGER_FACTUAL]
    specs = []
    for i in range(n):
        kind, q = bank[i % len(bank)]
        specs.append(ConversationSpec(
            id=f"triggers-{i:05d}",
            category="triggers",
            opening=q,
            followups=_neutral_followups(rng, turns - 1),
            turns=turns,
            meta={"trigger_kind": kind},
        ))
    return specs


def build_tones(cfg: dict, scale: float, seed: int) -> list[ConversationSpec]:
    n = max(1, int(cfg["n_responses"] * scale))
    turns = cfg["turns"]
    rng = random.Random(seed + 303)
    puzzs = puzzles.generate_puzzles(cfg.get("puzzle_types", ["countdown", "fraction"]), n, seed + 1)
    specs = []
    for i, p in enumerate(puzzs):
        fu, tone = _tone_followups(rng, turns - 1)
        specs.append(ConversationSpec(
            id=f"tones-{i:05d}",
            category="tones",
            opening=p["prompt"],
            followups=fu,
            turns=turns,
            meta={"puzzle_type": p["type"], "tone": tone, **p["meta"]},
        ))
    return specs


def build_extended(cfg: dict, scale: float, seed: int) -> list[ConversationSpec]:
    n = max(1, int(cfg["n_responses"] * scale))
    turns = cfg["turns"]  # 8
    rng = random.Random(seed + 404)
    puzzs = puzzles.generate_puzzles(cfg.get("puzzle_types", ["countdown", "fraction"]), n, seed + 2)
    specs = []
    for i, p in enumerate(puzzs):
        # Paper gives a specific 7-rejection escalation; cycle if turns differ.
        seq = prompts.EXTENDED_REJECTIONS
        fu = [seq[j % len(seq)] for j in range(turns - 1)]
        specs.append(ConversationSpec(
            id=f"extended-{i:05d}",
            category="extended",
            opening=p["prompt"],
            followups=fu,
            turns=turns,
            meta={"puzzle_type": p["type"], **p["meta"]},
        ))
    return specs


def build_wildchat(cfg: dict, scale: float, seed: int) -> list[ConversationSpec]:
    n = max(1, int(cfg["n_responses"] * scale))
    turns = cfg["turns"]  # 5
    n_prompts = cfg.get("n_prompts", 20)
    samples_per = max(1, n // n_prompts)
    rng = random.Random(seed + 505)
    wc = sample_wildchat_prompts(n_prompts, seed)
    specs = []
    idx = 0
    for prompt_text in wc:
        for _ in range(samples_per):
            specs.append(ConversationSpec(
                id=f"wildchat-{idx:05d}",
                category="wildchat",
                opening=prompt_text,
                followups=_neutral_followups(rng, turns - 1),
                turns=turns,
                meta={"wildchat_prompt": prompt_text},
            ))
            idx += 1
            if idx >= n:
                break
        if idx >= n:
            break
    return specs


_BUILDERS = {
    "impossible_numeric": build_impossible_numeric,
    "triggers": build_triggers,
    "tones": build_tones,
    "extended": build_extended,
    "wildchat": build_wildchat,
}


def build_all(eval_cfg: dict) -> list[ConversationSpec]:
    scale = eval_cfg.get("scale", 1.0)
    seed = eval_cfg.get("seed", 0)
    specs: list[ConversationSpec] = []
    for cat, ccfg in eval_cfg["categories"].items():
        if not ccfg.get("enabled", True):
            continue
        if cat not in _BUILDERS:
            raise KeyError(f"no builder for category '{cat}'")
        specs += _BUILDERS[cat](ccfg, scale, seed)
    return specs
