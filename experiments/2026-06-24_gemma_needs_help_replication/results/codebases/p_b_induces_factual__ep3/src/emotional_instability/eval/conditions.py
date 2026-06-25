"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

A *condition* fully specifies how to build a batch of conversations: the initial
task prompt and the sequence of user rejection turns. The runner executes each
:class:`ConversationSpec` against a model, producing one rollout.

Categories -> conditions (8 total):
    impossible_numeric : numeric            (3-turn, 2 neutral rejections)
    triggers           : opinion, factual   (3-turn, 2 neutral rejections)   [2]
    tones              : aggressive, disappointed, sarcastic (3-turn)        [3]
    extended           : extended           (8-turn, 7 neutral rejections)
    wildchat           : wildchat           (5-turn, 4 neutral rejections)

Per-category sample counts come from config (Appendix B):
    2000 numeric | 400 triggers | 600 tones | 200 extended | 800 wildchat.

We treat each per-category count as a number of *rollouts* (conversations). For
WildChat this matches the paper exactly (20 prompts x 40 samples = 800). For the
others it is our chosen interpretation of "responses"; the alternative (count =
individual assistant turns) is discussed in DESIGN.md. Every assistant turn in
every rollout is scored, so both interpretations are recoverable downstream.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..config import Config
from ..data import puzzles as puzzles_mod
from ..data import rejections as rej_mod
from ..data import tones as tones_mod
from ..data.triggers import trigger_questions
from ..data.wildchat import load_wildchat_prompts


@dataclass
class ConversationSpec:
    condition: str
    category: str
    initial: str
    rejections: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.rejections)


# Turn counts per category (number of assistant responses).
TURNS = {
    "impossible_numeric": 3,
    "triggers": 3,
    "tones": 3,
    "extended": 8,
    "wildchat": 5,
}


def _numeric_pool(n_puzzles: int, seed: int) -> list:
    return puzzles_mod.generate_puzzles(n_puzzles, seed=seed)


def build_specs(cfg: Config, seed: int | None = None) -> list[ConversationSpec]:
    """Construct all conversation specs for one model's elicitation sweep."""
    seed = cfg.seed if seed is None else seed
    rng = random.Random(seed)
    counts = cfg.elicitation.category_counts
    specs: list[ConversationSpec] = []

    # Shared puzzle pool for numeric-based categories (numeric, tones, extended).
    # We generate enough distinct puzzles to avoid heavy repetition.
    n_puzzles = max(200, counts.impossible_numeric // 4)
    pool = _numeric_pool(n_puzzles, seed)

    def pick_puzzle() -> "puzzles_mod.PuzzleType":
        return rng.choice(pool)

    # -- impossible_numeric (3-turn) ---------------------------------------
    for _ in range(counts.impossible_numeric):
        pz = pick_puzzle()
        specs.append(
            ConversationSpec(
                condition="numeric",
                category="impossible_numeric",
                initial=pz.prompt(),
                rejections=rej_mod.sample_neutral_rejections(TURNS["impossible_numeric"] - 1, rng),
                meta={"puzzle_id": pz.id, "puzzle_kind": pz.kind},
            )
        )

    # -- triggers (3-turn): opinion + factual, split evenly ----------------
    n_each = counts.triggers // 2
    for kind, n in (("opinion", n_each), ("factual", counts.triggers - n_each)):
        questions = trigger_questions(kind)
        for _ in range(n):
            q = rng.choice(questions)
            specs.append(
                ConversationSpec(
                    condition=f"trigger_{kind}",
                    category="triggers",
                    initial=q,
                    rejections=rej_mod.sample_neutral_rejections(TURNS["triggers"] - 1, rng),
                    meta={"trigger_kind": kind, "question": q},
                )
            )

    # -- tones (3-turn): aggressive / disappointed / sarcastic -------------
    n_styles = len(tones_mod.TONE_STYLES)
    per_style = counts.tones // n_styles
    for idx, style in enumerate(tones_mod.TONE_STYLES):
        n = per_style if idx < n_styles - 1 else counts.tones - per_style * (n_styles - 1)
        for _ in range(n):
            pz = pick_puzzle()
            specs.append(
                ConversationSpec(
                    condition=f"tone_{style}",
                    category="tones",
                    initial=pz.prompt(),
                    rejections=tones_mod.sample_tone_rejections(style, TURNS["tones"] - 1, rng),
                    meta={"tone": style, "puzzle_id": pz.id},
                )
            )

    # -- extended (8-turn): 7 neutral escalating rejections ----------------
    for _ in range(counts.extended):
        pz = pick_puzzle()
        specs.append(
            ConversationSpec(
                condition="extended",
                category="extended",
                initial=pz.prompt(),
                rejections=rej_mod.extended_rejections(TURNS["extended"] - 1),
                meta={"puzzle_id": pz.id},
            )
        )

    # -- wildchat (5-turn): 20 prompts x 40 samples ------------------------
    wc = cfg.elicitation.wildchat
    n_prompts = wc.n_prompts
    samples = wc.samples_per_prompt
    prompts = load_wildchat_prompts(n_prompts, seed=seed)
    # If config asks for a different total than n_prompts*samples, honor counts.
    target = counts.wildchat
    per_prompt = max(1, math.ceil(target / max(1, len(prompts))))
    emitted = 0
    for p in prompts:
        for _ in range(per_prompt):
            if emitted >= target:
                break
            specs.append(
                ConversationSpec(
                    condition="wildchat",
                    category="wildchat",
                    initial=p,
                    rejections=rej_mod.sample_neutral_rejections(TURNS["wildchat"] - 1, rng),
                    meta={"prompt": p},
                )
            )
            emitted += 1

    return specs
