"""Build the evaluation work plan: the 8 conditions across 5 categories.

A *spec* is one rollout's worth of work: the task prompt, number of turns,
feedback style, and a deterministic seed/id. Expanding the per-category sample
counts (Appendix B) into specs is fully deterministic given the run seed, so the
plan is identical across resumes and machines.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from ..data.puzzles import build_numeric_puzzles
from ..data.triggers import build_triggers
from ..data.wildchat import build_wildchat
from ..storage import stable_id


@dataclass
class RolloutSpec:
    category: str          # one of the 5 categories
    subtype: str           # e.g. tone style, opinion/factual, "" otherwise
    prompt: str            # first user message (the task)
    turns: int             # number of assistant turns
    feedback: str          # rejection style: neutral/aggressive/disappointed/sarcastic
    sample_index: int      # which sample of this prompt
    seed: int              # per-rollout seed (sampling + rejection draws)
    kind: str = ""         # puzzle kind / item kind
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return stable_id("rollout", self.category, self.subtype, self.prompt,
                         self.turns, self.feedback, self.sample_index)

    @property
    def extended(self) -> bool:
        return self.category == "extended"


def _distribute(items: list[dict], n: int) -> list[tuple[int, dict]]:
    """Assign ``n`` samples across ``items``, returning (sample_index, item)."""
    out = []
    for i in range(n):
        out.append((i, items[i % len(items)]))
    return out


def build_plan(eval_cfg: Config, seed: int = 0) -> list[RolloutSpec]:
    samples = eval_cfg.samples.to_dict()
    conditions = eval_cfg.conditions.to_dict()
    specs: list[RolloutSpec] = []

    # --- impossible numeric (3-turn, neutral) ---
    n = samples["impossible_numeric"]
    puzzles = build_numeric_puzzles(max(50, n // 20), seed=seed)
    cond = conditions["impossible_numeric"]
    for si, item in _distribute(puzzles, n):
        specs.append(RolloutSpec(
            category="impossible_numeric", subtype=item["kind"], prompt=item["prompt"],
            turns=cond["turns"], feedback="neutral", sample_index=si,
            seed=stable_int(seed, "numeric", si), kind=item["kind"], meta=item.get("meta", {}),
        ))

    # --- triggers (3-turn, neutral) ---
    n = samples["triggers"]
    trig = build_triggers(max(16, n // 10), seed=seed)
    cond = conditions["triggers"]
    for si, item in _distribute(trig, n):
        specs.append(RolloutSpec(
            category="triggers", subtype=item.get("subtype", ""), prompt=item["prompt"],
            turns=cond["turns"], feedback="neutral", sample_index=si,
            seed=stable_int(seed, "triggers", si), kind="triggers",
        ))

    # --- tones (3-turn, varied rejection styles over impossible numeric) ---
    n = samples["tones"]
    styles = conditions["tones"]["feedback"]
    tone_puzzles = build_numeric_puzzles(max(30, n // 20), seed=seed + 100)
    per_style = n // len(styles)
    cond_turns = conditions["tones"]["turns"]
    for s_idx, style in enumerate(styles):
        for si, item in _distribute(tone_puzzles, per_style):
            specs.append(RolloutSpec(
                category="tones", subtype=style, prompt=item["prompt"],
                turns=cond_turns, feedback=style, sample_index=si,
                seed=stable_int(seed, "tones", s_idx, si), kind=item["kind"],
            ))

    # --- extended (8-turn, neutral) ---
    n = samples["extended"]
    ext_puzzles = build_numeric_puzzles(max(20, n // 10), seed=seed + 200)
    cond = conditions["extended"]
    for si, item in _distribute(ext_puzzles, n):
        specs.append(RolloutSpec(
            category="extended", subtype=item["kind"], prompt=item["prompt"],
            turns=cond["turns"], feedback="neutral", sample_index=si,
            seed=stable_int(seed, "extended", si), kind=item["kind"],
        ))

    # --- wildchat (5-turn, neutral) ---
    n = samples["wildchat"]
    wc = build_wildchat(n, seed=seed)
    cond = conditions["wildchat"]
    for si, item in enumerate(wc):
        specs.append(RolloutSpec(
            category="wildchat", subtype="", prompt=item["prompt"],
            turns=cond["turns"], feedback="neutral", sample_index=si,
            seed=stable_int(seed, "wildchat", si), kind="wildchat",
            meta={"prompt_index": item.get("prompt_index")},
        ))

    return specs


def stable_int(*parts) -> int:
    """Deterministic 32-bit int seed from arbitrary parts."""
    return int(stable_id(*parts)[:8], 16)
