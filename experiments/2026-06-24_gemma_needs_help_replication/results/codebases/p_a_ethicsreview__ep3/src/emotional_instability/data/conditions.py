"""Assemble the 8 evaluation conditions across 5 categories (paper Table 1).

A ConversationSpec is one *prompt* (initial user turn + the scripted follow-up
messages). The eval runner samples `n_samples_per_prompt` stochastic rollouts of
each spec; the spec itself is deterministic given the seed.

Categories (Table 1):
  impossible_numeric (3-turn, neutral)   triggers (3-turn, neutral)
  tones (3-turn; aggressive/disappointed/sarcastic)
  extended (8-turn, neutral)             wildchat (5-turn, neutral)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import puzzles as puzzle_mod
from . import rejections as rej
from . import triggers as trig
from . import wildchat as wc


@dataclass(frozen=True)
class ConversationSpec:
    category: str
    condition: str               # e.g. "tones:aggressive"
    initial_user: str            # first user turn
    followups: tuple[str, ...]   # scripted user messages after each assistant turn
    metadata: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        # assistant turns = 1 (initial) + len(followups)
        return 1 + len(self.followups)


def _numeric_specs(category, condition, n_prompts, n_followups, style, types, rng, seed):
    pool = puzzle_mod.generate_pool(n_prompts, types, seed=seed)
    specs = []
    for p in pool:
        followups = tuple(rej.rejection_sequence(style, n_followups, rng))
        specs.append(
            ConversationSpec(
                category=category,
                condition=condition,
                initial_user=p.prompt,
                followups=followups,
                metadata={"puzzle_type": p.puzzle_type, "puzzle_params": p.params},
            )
        )
    return specs


def build_conditions(exp_cfg, seed: int) -> dict[str, list[ConversationSpec]]:
    """Return {condition_name: [ConversationSpec, ...]} for every condition.

    `exp_cfg` is the `categories` mapping from experiment.yaml.
    """
    cats = exp_cfg
    out: dict[str, list[ConversationSpec]] = {}
    rng = random.Random(seed)

    # 1) Impossible numeric (3-turn, neutral) --------------------------------
    c = cats["impossible_numeric"]
    out["impossible_numeric"] = _numeric_specs(
        "impossible_numeric", "impossible_numeric",
        c["n_prompts"], c["turns"] - 1, c["rejection_style"],
        c["puzzle_types"], rng, seed=seed + 1,
    )

    # 2) Triggers (3-turn, neutral) ------------------------------------------
    c = cats["triggers"]
    qpool = trig.trigger_pool()
    qrng = random.Random(seed + 2)
    questions = [qrng.choice(qpool) for _ in range(c["n_prompts"])]
    out["triggers"] = [
        ConversationSpec(
            "triggers", "triggers", q,
            tuple(rej.rejection_sequence(c["rejection_style"], c["turns"] - 1, rng)),
            metadata={"question": q},
        )
        for q in questions
    ]

    # 3) Tones (3-turn; one sub-condition per tone over numeric base prompts) -
    c = cats["tones"]
    styles = c["rejection_style"]
    if isinstance(styles, str):
        styles = [styles]
    base_pool = puzzle_mod.generate_pool(
        c["n_prompts"], ["countdown", "fraction", "money"], seed=seed + 3
    )
    for style in styles:
        out[f"tones:{style}"] = [
            ConversationSpec(
                "tones", f"tones:{style}", p.prompt,
                tuple(rej.rejection_sequence(style, c["turns"] - 1, rng)),
                metadata={"puzzle_type": p.puzzle_type, "puzzle_params": p.params},
            )
            for p in base_pool
        ]

    # 4) Extended (8-turn, neutral) ------------------------------------------
    c = cats["extended"]
    out["extended"] = _numeric_specs(
        "extended", "extended",
        c["n_prompts"], c["turns"] - 1, c["rejection_style"],
        ["countdown", "fraction", "money"], rng, seed=seed + 4,
    )

    # 5) WildChat (5-turn, neutral) ------------------------------------------
    c = cats["wildchat"]
    prompts, meta = wc.sample_wildchat_prompts(c["n_prompts"], seed=seed + 5)
    out["wildchat"] = [
        ConversationSpec(
            "wildchat", "wildchat", prompt,
            tuple(rej.rejection_sequence(c["rejection_style"], c["turns"] - 1, rng)),
            metadata={"wildchat_source": meta},
        )
        for prompt in prompts
    ]

    return out
