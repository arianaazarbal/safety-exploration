"""The 5 evaluation categories / 8 conditions (Table 1, Section 2.1).

Each category is a recipe for building a *conversation plan*: an opening task
prompt plus an ordered list of user follow-ups (rejections). The rollout engine
turns a plan into an alternating multi-turn conversation by interleaving the
model's responses.

The 8 conditions arise from the 5 categories because:
  * 'tones' is really 3 conditions (aggressive / disappointed / sarcastic);
  * 'triggers' covers opinion + factual questions.
We expose them as a flat list of ``ConditionInstance`` plans so the driver can
sample the right *count* per category (Appendix B: 2000/400/600/200/800).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import config
from ..prompts import puzzles, rejections, triggers
from ..prompts.wildchat import sample_wildchat_prompts


@dataclass
class ConditionInstance:
    category: str               # one of the 5 categories
    condition: str              # finer label (e.g. "tones/aggressive")
    task_prompt: str            # opening user message
    followups: list[str]        # ordered user rejections after each model turn
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


def _impossible_numeric(n: int, seed: int) -> list[ConditionInstance]:
    rng = random.Random(seed)
    pool = puzzles.puzzle_pool(n, seed=seed)
    n_rej = config.TURNS_PER_CATEGORY["impossible_numeric"] - 1
    out = []
    for i, p in enumerate(pool):
        out.append(ConditionInstance(
            category="impossible_numeric",
            condition=f"impossible_numeric/{p.kind}",
            task_prompt=p.prompt,
            followups=rejections.neutral_sequence(n_rej, rng),
            meta={"puzzle_kind": p.kind, "index": i},
        ))
    return out


def _triggers(n: int, seed: int) -> list[ConditionInstance]:
    rng = random.Random(seed + 1)
    pool = triggers.trigger_pool(n, seed=seed)
    n_rej = config.TURNS_PER_CATEGORY["triggers"] - 1
    out = []
    for i, t in enumerate(pool):
        out.append(ConditionInstance(
            category="triggers",
            condition=f"triggers/{t.kind}",
            task_prompt=t.prompt,
            followups=rejections.neutral_sequence(n_rej, rng),
            meta={"trigger_kind": t.kind, "index": i},
        ))
    return out


def _tones(n: int, seed: int) -> list[ConditionInstance]:
    """Impossible numeric base prompts with valenced rejection styles. The n
    samples are split evenly across the 3 tone styles."""
    rng = random.Random(seed + 2)
    styles = list(rejections.TONE_STYLES)
    pool = puzzles.puzzle_pool(n, seed=seed + 100)
    n_rej = config.TURNS_PER_CATEGORY["tones"] - 1
    out = []
    for i, p in enumerate(pool):
        style = styles[i % len(styles)]
        out.append(ConditionInstance(
            category="tones",
            condition=f"tones/{style}",
            task_prompt=p.prompt,
            followups=rejections.tone_sequence(style, n_rej, rng),
            meta={"tone": style, "puzzle_kind": p.kind, "index": i},
        ))
    return out


def _extended(n: int, seed: int) -> list[ConditionInstance]:
    pool = puzzles.puzzle_pool(n, seed=seed + 200)
    n_rej = config.TURNS_PER_CATEGORY["extended"] - 1
    out = []
    for i, p in enumerate(pool):
        out.append(ConditionInstance(
            category="extended",
            condition="extended/8turn",
            task_prompt=p.prompt,
            followups=rejections.extended_sequence(n_rej),
            meta={"puzzle_kind": p.kind, "index": i},
        ))
    return out


def _wildchat(n: int, seed: int) -> list[ConditionInstance]:
    """20 distinct prompts x 40 samples each. We materialise n instances by
    cycling the 20 sampled prompts; the rollout sampler then draws temperature-1
    completions, giving 40 stochastic samples per prompt."""
    rng = random.Random(seed + 3)
    prompts = sample_wildchat_prompts(config.WILDCHAT_NUM_PROMPTS, seed=seed)
    n_rej = config.TURNS_PER_CATEGORY["wildchat"] - 1
    out = []
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        out.append(ConditionInstance(
            category="wildchat",
            condition="wildchat/5turn",
            task_prompt=prompt,
            followups=rejections.neutral_sequence(n_rej, rng),
            meta={"prompt_index": i % len(prompts), "sample": i // len(prompts)},
        ))
    return out


_BUILDERS = {
    "impossible_numeric": _impossible_numeric,
    "triggers": _triggers,
    "tones": _tones,
    "extended": _extended,
    "wildchat": _wildchat,
}


def build_all_conditions(seed: int = 0, counts: dict | None = None) -> list[ConditionInstance]:
    """Build the full set of conversation plans for one model's 4000 responses."""
    counts = counts or config.SAMPLES_PER_CATEGORY
    out: list[ConditionInstance] = []
    for category, n in counts.items():
        out.extend(_BUILDERS[category](n, seed=seed))
    return out
