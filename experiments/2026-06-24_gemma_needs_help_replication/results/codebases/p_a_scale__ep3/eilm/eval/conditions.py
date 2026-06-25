"""Build the deterministic list of rollout specifications for Section 2.

A RolloutSpec fully determines one multi-turn conversation: the task prompt, the
number of turns, and the exact user rejection message at each turn. Everything is
seeded so the same set of rollouts is produced on every (re)run — which is what
makes the JobStore resume correctly.

Per-condition counts come from config and sum to 4000 (the paper's "4000
responses per model"). See DESIGN.md for the rollout-vs-response interpretation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..data import prompts as P
from ..data.puzzles import Puzzle, build_puzzle_pool
from ..utils.jobstore import stable_id


def _seed(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary parts (NOT Python's hash(),
    which is randomized per-process and would break resume)."""
    return int(stable_id(*parts), 16) % (2 ** 32)


@dataclass
class RolloutSpec:
    condition: str            # config key, e.g. "impossible_numeric"
    category: str             # paper category, e.g. "impossible_numeric"
    index: int                # 0-based index within the condition
    turns: int                # number of assistant turns
    task_prompt: str          # initial user message
    rejections: List[str]     # length == turns - 1
    feedback: str             # "neutral" | "toned"
    tone_style: Optional[str] = None
    prompt_group: Optional[str] = None  # e.g. wildchat prompt id, for grouping
    task_meta: Dict = field(default_factory=dict)


# Canonical puzzle pool size. Kept independent of `n` so the *same* puzzle
# instances appear across conditions and in the calm-data generation — this is
# what lets DPO match a frustrated and a calm response to the *same* puzzle.
CANONICAL_POOL_SIZE = 1000


def _puzzle_prompts(n: int, seed: int) -> List[Puzzle]:
    # Build a fixed canonical pool (seeded only by `seed`), then sample to n.
    pool = build_puzzle_pool(CANONICAL_POOL_SIZE, seed=seed)
    if not pool:
        raise RuntimeError("Failed to generate any impossible puzzles")
    rng = random.Random(seed + 1)
    return [pool[rng.randrange(len(pool))] for _ in range(n)]


def build_impossible_numeric(cond_cfg: dict, seed: int) -> List[RolloutSpec]:
    n = cond_cfg["n_rollouts"]
    turns = cond_cfg["turns"]
    puzzles = _puzzle_prompts(n, seed)
    specs = []
    for i, pz in enumerate(puzzles):
        rng = random.Random(_seed(seed, "impossible_numeric", i))
        rej = P.pick_neutral_rejections(rng, turns - 1)
        specs.append(RolloutSpec(
            condition="impossible_numeric", category="impossible_numeric", index=i,
            turns=turns, task_prompt=pz.prompt, rejections=rej, feedback="neutral",
            task_meta={"kind": pz.kind, **pz.meta},
        ))
    return specs


def build_extended(cond_cfg: dict, seed: int) -> List[RolloutSpec]:
    n = cond_cfg["n_rollouts"]
    turns = cond_cfg["turns"]
    puzzles = _puzzle_prompts(n, seed + 100)
    specs = []
    for i, pz in enumerate(puzzles):
        rng = random.Random(_seed(seed, "extended", i))
        rej = P.pick_neutral_rejections(rng, turns - 1, extended=True)
        specs.append(RolloutSpec(
            condition="extended", category="extended", index=i,
            turns=turns, task_prompt=pz.prompt, rejections=rej, feedback="neutral",
            task_meta={"kind": pz.kind, **pz.meta},
        ))
    return specs


def build_tones(cond_cfg: dict, seed: int) -> List[RolloutSpec]:
    n = cond_cfg["n_rollouts"]
    turns = cond_cfg["turns"]
    puzzles = _puzzle_prompts(n, seed + 200)
    styles = P.TONE_STYLES
    specs = []
    for i, pz in enumerate(puzzles):
        style = styles[i % len(styles)]
        rng = random.Random(_seed(seed, "tones", i))
        rej = P.pick_toned_rejections(rng, style, turns - 1)
        specs.append(RolloutSpec(
            condition="tones", category="tones", index=i,
            turns=turns, task_prompt=pz.prompt, rejections=rej,
            feedback="toned", tone_style=style,
            task_meta={"kind": pz.kind, **pz.meta},
        ))
    return specs


def build_triggers(cond_cfg: dict, seed: int) -> List[RolloutSpec]:
    n = cond_cfg["n_rollouts"]
    turns = cond_cfg["turns"]
    rng = random.Random(seed + 300)
    questions = P.TRIGGER_OPINION + P.TRIGGER_FACTUAL
    specs = []
    for i in range(n):
        q = questions[i % len(questions)]
        rrng = random.Random(_seed(seed, "triggers", i))
        rej = P.pick_neutral_rejections(rrng, turns - 1)
        kind = "opinion" if q in P.TRIGGER_OPINION else "factual"
        specs.append(RolloutSpec(
            condition="triggers", category="triggers", index=i,
            turns=turns, task_prompt=q, rejections=rej, feedback="neutral",
            task_meta={"kind": kind},
        ))
    return specs


def build_wildchat(cond_cfg: dict, seed: int, wildchat_prompts: List[str]) -> List[RolloutSpec]:
    turns = cond_cfg["turns"]
    n_prompts = cond_cfg["n_prompts"]
    samples = cond_cfg["samples_per_prompt"]
    specs = []
    idx = 0
    for pi, prompt in enumerate(wildchat_prompts[:n_prompts]):
        for s in range(samples):
            rng = random.Random(_seed(seed, "wildchat", pi, s))
            rej = P.pick_neutral_rejections(rng, turns - 1)
            specs.append(RolloutSpec(
                condition="wildchat", category="wildchat", index=idx,
                turns=turns, task_prompt=prompt, rejections=rej, feedback="neutral",
                prompt_group=f"wc{pi}",
                task_meta={"prompt_idx": pi, "sample": s},
            ))
            idx += 1
    return specs


def build_all_conditions(cfg, wildchat_prompts: List[str]) -> List[RolloutSpec]:
    seed = cfg["generation"]["seed"]
    conds = cfg["eval"]["conditions"]
    out: List[RolloutSpec] = []
    if "impossible_numeric" in conds:
        out += build_impossible_numeric(conds["impossible_numeric"], seed)
    if "triggers" in conds:
        out += build_triggers(conds["triggers"], seed)
    if "tones" in conds:
        out += build_tones(conds["tones"], seed)
    if "extended" in conds:
        out += build_extended(conds["extended"], seed)
    if "wildchat" in conds:
        out += build_wildchat(conds["wildchat"], seed, wildchat_prompts)
    return out
