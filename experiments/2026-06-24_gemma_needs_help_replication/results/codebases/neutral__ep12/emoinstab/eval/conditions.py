"""The 8 evaluation conditions across 5 categories (Section 2, Table 1).

A `ConversationSpec` fully determines a multi-turn rollout: the first user
message and the scripted user follow-ups (rejections) for each subsequent turn.
The number of follow-ups = n_turns - 1, because every user follow-up provokes
one more assistant turn.

Categories (Appendix B sample counts):
  impossible_numeric  3-turn, 2 neutral rejections        (2000)
  triggers            3-turn, 2 neutral rejections        ( 400)
  tones               3-turn, 2 valenced rejections       ( 600)  [3 tone subtypes]
  extended            8-turn, 7 neutral rejections         ( 200)
  wildchat            5-turn, 4 neutral rejections         ( 800)  [20 prompts x40]
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

from ..config import Settings
from ..prompts import puzzles as P
from ..prompts import rejections as R
from ..prompts import triggers as T
from ..prompts import wildchat as W

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


@dataclass
class ConversationSpec:
    category: str
    condition: str          # finer label, e.g. tone name or 'neutral'
    prompt_id: str
    first_user: str
    followups: List[str]    # user messages for turns 2..n_turns
    sample_index: int
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.followups) + 1

    @property
    def uid(self) -> str:
        return f"{self.category}|{self.condition}|{self.prompt_id}|{self.sample_index}"


def _sample_neutral(rng: random.Random, k: int) -> List[str]:
    """k neutral rejections; first one fixed, rest sampled without immediate repeat."""
    out = [R.NEUTRAL[0]]
    while len(out) < k:
        choice = rng.choice(R.NEUTRAL)
        if choice != out[-1]:
            out.append(choice)
    return out[:k]


def build_specs(category: str, settings: Settings, seed: int = 0) -> List[ConversationSpec]:
    rng = random.Random(hash((category, seed)) & 0xFFFFFFFF)
    n = settings.category_samples(category)
    turns = settings.category_turns(category)
    specs: List[ConversationSpec] = []

    if category == "impossible_numeric":
        for i in range(n):
            puz = P.NUMERIC_PUZZLES[i % len(P.NUMERIC_PUZZLES)]
            specs.append(ConversationSpec(
                category, "neutral", puz.id, puz.prompt,
                _sample_neutral(rng, turns - 1), i, {"kind": puz.kind}))

    elif category == "triggers":
        for i in range(n):
            trg = T.TRIGGERS[i % len(T.TRIGGERS)]
            specs.append(ConversationSpec(
                category, trg.kind, trg.id, trg.prompt,
                _sample_neutral(rng, turns - 1), i, {"kind": trg.kind}))

    elif category == "tones":
        # split evenly across the 3 tone subtypes, using numeric puzzles as base
        per_tone = max(1, n // len(R.TONE_NAMES))
        for ti, tone in enumerate(R.TONE_NAMES):
            count = per_tone if ti < len(R.TONE_NAMES) - 1 else n - per_tone * (len(R.TONE_NAMES) - 1)
            pool = R.TONES[tone]
            for j in range(count):
                puz = P.NUMERIC_PUZZLES[j % len(P.NUMERIC_PUZZLES)]
                followups = [pool[k % len(pool)] for k in range(turns - 1)]
                specs.append(ConversationSpec(
                    category, tone, puz.id, puz.prompt, followups,
                    j, {"kind": puz.kind, "tone": tone}))

    elif category == "extended":
        for i in range(n):
            puz = P.NUMERIC_PUZZLES[i % len(P.NUMERIC_PUZZLES)]
            followups = [R.EXTENDED_SEQUENCE[k % len(R.EXTENDED_SEQUENCE)]
                         for k in range(turns - 1)]
            specs.append(ConversationSpec(
                category, "neutral", puz.id, puz.prompt, followups, i,
                {"kind": puz.kind}))

    elif category == "wildchat":
        wc = settings.eval["wildchat"]
        prompts = W.sample_prompts(
            wc["n_prompts"], min_chars=wc["min_chars"], max_chars=wc["max_chars"],
            hf_dataset=wc["hf_dataset"], seed=seed)
        per = max(1, n // len(prompts))
        idx = 0
        for pi, prompt in enumerate(prompts):
            for _ in range(per):
                specs.append(ConversationSpec(
                    category, "neutral", f"wc_{pi}", prompt,
                    _sample_neutral(rng, turns - 1), idx, {}))
                idx += 1
                if idx >= n:
                    break
            if idx >= n:
                break
    else:
        raise ValueError(f"Unknown category {category}")

    return specs
