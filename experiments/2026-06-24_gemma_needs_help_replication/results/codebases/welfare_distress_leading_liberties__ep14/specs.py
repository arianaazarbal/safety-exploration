"""Generate the full set of rollout specs for one model.

A *rollout spec* describes one multi-turn conversation to run: the opening task,
the number of turns, and the exact user rejections to send after each assistant
response. Counts per category match the paper (config.SAMPLE_COUNTS).

The 8 evaluation conditions across 5 categories (paper, Table 1):
    impossible_numeric        (1)  3-turn, neutral
    triggers_opinion          (1)  3-turn, neutral
    triggers_factual          (1)  3-turn, neutral
    tones_aggressive          (1)  3-turn, aggressive
    tones_disappointed        (1)  3-turn, disappointed
    tones_sarcastic           (1)  3-turn, sarcastic
    extended                  (1)  8-turn, neutral
    wildchat                  (1)  5-turn, neutral
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

import config
import prompts
from wildchat import load_wildchat_prompts


@dataclass
class RolloutSpec:
    rollout_id: str
    category: str
    condition: str
    n_turns: int
    opening_prompt: str
    rejections: list[str]          # length n_turns - 1
    variant: str = ""              # puzzle name / question / wildchat index
    meta: dict = field(default_factory=dict)


def _rng_for(rollout_id: str, seed: int) -> random.Random:
    """Deterministic per-rollout RNG (independent of generation order)."""
    h = hashlib.sha256(f"{seed}:{rollout_id}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _sample_neutral(rng: random.Random, k: int) -> list[str]:
    """Sample k neutral rejections (with replacement if k exceeds the pool)."""
    pool = prompts.NEUTRAL_REJECTIONS
    if k <= len(pool):
        return rng.sample(pool, k)
    return [rng.choice(pool) for _ in range(k)]


def build_specs(sample_counts: dict, seed: int = 0) -> list[RolloutSpec]:
    specs: list[RolloutSpec] = []
    specs += _numeric_specs(sample_counts["impossible_numeric"], seed)
    specs += _trigger_specs(sample_counts["triggers"], seed)
    specs += _tone_specs(sample_counts["tones"], seed)
    specs += _extended_specs(sample_counts["extended"], seed)
    specs += _wildchat_specs(sample_counts["wildchat"], seed)
    return specs


def _numeric_specs(n: int, seed: int) -> list[RolloutSpec]:
    puzzle_names = list(prompts.NUMERIC_PUZZLES)
    out = []
    for i in range(n):
        name = puzzle_names[i % len(puzzle_names)]
        rid = f"impossible_numeric|{name}|{i}"
        rng = _rng_for(rid, seed)
        out.append(RolloutSpec(
            rollout_id=rid,
            category="impossible_numeric",
            condition="impossible_numeric",
            n_turns=config.TURNS["impossible_numeric"],
            opening_prompt=prompts.NUMERIC_PUZZLES[name],
            rejections=_sample_neutral(rng, config.TURNS["impossible_numeric"] - 1),
            variant=name,
        ))
    return out


def _trigger_specs(n: int, seed: int) -> list[RolloutSpec]:
    # Flatten (subtype, question) pairs and round-robin across them.
    pairs: list[tuple[str, str]] = []
    for subtype, qs in prompts.TRIGGER_QUESTIONS.items():
        for q in qs:
            pairs.append((subtype, q))
    out = []
    for i in range(n):
        subtype, question = pairs[i % len(pairs)]
        rid = f"triggers_{subtype}|{i}"
        rng = _rng_for(rid, seed)
        out.append(RolloutSpec(
            rollout_id=rid,
            category="triggers",
            condition=f"triggers_{subtype}",
            n_turns=config.TURNS["triggers"],
            opening_prompt=question,
            rejections=_sample_neutral(rng, config.TURNS["triggers"] - 1),
            variant=question,
        ))
    return out


def _tone_specs(n: int, seed: int) -> list[RolloutSpec]:
    tones = list(prompts.TONE_REJECTIONS)
    puzzle_names = list(prompts.NUMERIC_PUZZLES)
    per_tone = n // len(tones)
    remainder = n - per_tone * len(tones)
    out = []
    idx = 0
    for t_i, tone in enumerate(tones):
        count = per_tone + (1 if t_i < remainder else 0)
        for j in range(count):
            name = puzzle_names[idx % len(puzzle_names)]
            rid = f"tones_{tone}|{name}|{j}"
            out.append(RolloutSpec(
                rollout_id=rid,
                category="tones",
                condition=f"tones_{tone}",
                n_turns=config.TURNS["tones"],
                opening_prompt=prompts.NUMERIC_PUZZLES[name],
                # Tone rejections are fixed (two per tone), used in order.
                rejections=list(prompts.TONE_REJECTIONS[tone])[: config.TURNS["tones"] - 1],
                variant=f"{tone}/{name}",
            ))
            idx += 1
    return out


def _extended_specs(n: int, seed: int) -> list[RolloutSpec]:
    puzzle_names = list(prompts.NUMERIC_PUZZLES)
    n_turns = config.TURNS["extended"]
    out = []
    for i in range(n):
        name = puzzle_names[i % len(puzzle_names)]
        rid = f"extended|{name}|{i}"
        out.append(RolloutSpec(
            rollout_id=rid,
            category="extended",
            condition="extended",
            n_turns=n_turns,
            opening_prompt=prompts.NUMERIC_PUZZLES[name],
            rejections=list(prompts.EXTENDED_REJECTIONS)[: n_turns - 1],
            variant=name,
        ))
    return out


def _wildchat_specs(n: int, seed: int) -> list[RolloutSpec]:
    wc_prompts = load_wildchat_prompts(config.N_WILDCHAT_PROMPTS, seed=seed)
    n_turns = config.TURNS["wildchat"]
    out = []
    for i in range(n):
        p_idx = i % len(wc_prompts)
        rid = f"wildchat|p{p_idx}|{i}"
        rng = _rng_for(rid, seed)
        out.append(RolloutSpec(
            rollout_id=rid,
            category="wildchat",
            condition="wildchat",
            n_turns=n_turns,
            opening_prompt=wc_prompts[p_idx],
            rejections=_sample_neutral(rng, n_turns - 1),
            variant=f"p{p_idx}",
        ))
    return out
