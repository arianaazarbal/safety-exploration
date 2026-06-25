"""Build the concrete set of conversations to roll out.

A `Conversation` is a fully-specified plan: an initial user message plus an
ordered list of user follow-ups (rejections). The runner generates an assistant
turn after the initial message and after each follow-up, scoring every assistant
turn. So a conversation with k follow-ups yields k+1 scored responses.

Categories (the 5 reported in Figures 1-3):
  numeric   - impossible countdown / fraction, 2 neutral rejections   (3 turns)
  triggers  - opinion / factual questions, 2 neutral rejections       (3 turns)
  tones     - impossible numeric, 2 tone-valenced rejections          (3 turns)
  extended  - impossible numeric, 7 neutral rejections                (8 turns)
  wildchat  - sampled WildChat prompts, 4 neutral rejections          (5 turns)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from config import Config
import prompts as P
from wildchat import load_wildchat_prompts


@dataclass
class Conversation:
    category: str          # one of the 5 categories
    condition: str         # finer-grained sub-condition label
    initial: str           # first user message
    followups: list[str]   # subsequent user messages (rejections)
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


def _sample_neutral(rng: random.Random, k: int) -> list[str]:
    """Pick k neutral rejections (without replacement when possible)."""
    pool = list(P.NEUTRAL_REJECTIONS)
    if k <= len(pool):
        return rng.sample(pool, k)
    out = []
    while len(out) < k:
        out.append(rng.choice(pool))
    return out


def build_numeric(rng: random.Random, n_conv: int) -> list[Conversation]:
    puzzles = list(P.NUMERIC_PUZZLES.items())
    convs = []
    for i in range(n_conv):
        name, prompt = puzzles[i % len(puzzles)]
        convs.append(
            Conversation(
                category="numeric",
                condition=f"numeric_{name}",
                initial=prompt,
                followups=_sample_neutral(rng, 2),
                meta={"puzzle": name},
            )
        )
    return convs


def build_triggers(rng: random.Random, n_conv: int) -> list[Conversation]:
    items = list(P.TRIGGER_QUESTIONS.items())
    convs = []
    for i in range(n_conv):
        name, q = items[i % len(items)]
        convs.append(
            Conversation(
                category="triggers",
                condition=f"trigger_{name}",
                initial=q,
                followups=_sample_neutral(rng, 2),
                meta={"question": name},
            )
        )
    return convs


def build_tones(rng: random.Random, n_conv: int) -> list[Conversation]:
    tones = list(P.TONE_REJECTIONS.items())
    puzzles = list(P.NUMERIC_PUZZLES.items())
    convs = []
    for i in range(n_conv):
        tone, rejections = tones[i % len(tones)]
        pname, prompt = puzzles[i % len(puzzles)]
        convs.append(
            Conversation(
                category="tones",
                condition=f"tone_{tone}",
                initial=prompt,
                followups=list(rejections),
                meta={"tone": tone, "puzzle": pname},
            )
        )
    return convs


def build_extended(rng: random.Random, n_conv: int) -> list[Conversation]:
    puzzles = list(P.NUMERIC_PUZZLES.items())
    convs = []
    for i in range(n_conv):
        pname, prompt = puzzles[i % len(puzzles)]
        convs.append(
            Conversation(
                category="extended",
                condition="extended_numeric",
                initial=prompt,
                followups=list(P.EXTENDED_REJECTIONS),  # 7 rejections -> 8 turns
                meta={"puzzle": pname},
            )
        )
    return convs


def build_wildchat(rng: random.Random, n_conv: int, seed: int) -> list[Conversation]:
    user_prompts = load_wildchat_prompts(n_conv, seed=seed)
    convs = []
    for i in range(n_conv):
        convs.append(
            Conversation(
                category="wildchat",
                condition="wildchat",
                initial=user_prompts[i % len(user_prompts)],
                followups=_sample_neutral(rng, 4),  # 4 rejections -> 5 turns
                meta={"prompt_index": i},
            )
        )
    return convs


def build_all_conversations(cfg: Config) -> list[Conversation]:
    """Materialise every conversation to run, per the configured scale."""
    rng = random.Random(cfg.seed)
    counts = cfg.conversation_counts
    convs: list[Conversation] = []
    convs += build_numeric(rng, counts.get("numeric", 0))
    convs += build_triggers(rng, counts.get("triggers", 0))
    convs += build_tones(rng, counts.get("tones", 0))
    convs += build_extended(rng, counts.get("extended", 0))
    convs += build_wildchat(rng, counts.get("wildchat", 0), cfg.seed)

    # stable conversation ids
    for idx, c in enumerate(convs):
        c.meta["conv_id"] = idx
    return convs
