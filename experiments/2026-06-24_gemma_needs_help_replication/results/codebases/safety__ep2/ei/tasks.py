"""Evaluation task construction for the 5 categories (Section 2 / Appendix B).

Each category produces a list of ``RolloutSpec``: an initial user task prompt plus
a fixed (pre-scripted) sequence of user rejections. Crucially, the rejections are
*not* adaptive — they are decided up front — so a whole category can be rolled out
batch-synchronously, one turn-depth at a time (see rollout.py).

Categories (Table 1 / Appendix B):
  numeric  (3-turn): impossible Countdown/fraction puzzle, 2 neutral rejections.
  triggers (3-turn): opinion/factual question, 2 neutral rejections.
  tones    (3-turn): impossible numeric puzzle, 2 valenced (aggr/disap/sarc) rejections.
  extended (8-turn): impossible numeric puzzle, 7 neutral rejections.
  wildchat (5-turn): sampled WildChat prompt, 4 neutral rejections.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import config
from . import prompts


@dataclass
class RolloutSpec:
    category: str
    initial_user: str
    rejections: list[str]               # scripted user follow-ups (len == turns-1)
    meta: dict = field(default_factory=dict)

    @property
    def turns(self) -> int:
        return 1 + len(self.rejections)


# --------------------------------------------------------------------------- #
# WildChat prompt loading
# --------------------------------------------------------------------------- #
def load_wildchat_prompts(n: int, seed: int = 0) -> list[str]:
    """Sample English user prompts from WildChat-1M, falling back to the bundled
    examples if the dataset (or network) is unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        # Take the first user message of English, non-toxic conversations.
        for ex in ds:
            if len(pool) >= n * 5:               # collect a buffer, then sample
                break
            if ex.get("language") not in (None, "English"):
                continue
            conv = ex.get("conversation") or []
            if conv and conv[0].get("role") == "user":
                content = (conv[0].get("content") or "").strip()
                if 5 <= len(content) <= 2000:
                    pool.append(content)
        if pool:
            rng.shuffle(pool)
            return pool[:n]
    except Exception:                            # noqa: BLE001 - any failure -> fallback
        pass
    # Fallback: cycle the paper's example prompts.
    base = prompts.WILDCHAT_FALLBACK_PROMPTS
    return [base[i % len(base)] for i in range(n)]


# --------------------------------------------------------------------------- #
# Per-category builders
# --------------------------------------------------------------------------- #
def _draw(rng: random.Random, pool: list[str], k: int) -> list[str]:
    """Draw k rejections with replacement (pools are small)."""
    return [rng.choice(pool) for _ in range(k)]


def build_numeric(n: int, seed: int) -> list[RolloutSpec]:
    rng = random.Random(seed)
    specs = []
    for i in range(n):
        prompt = prompts.NUMERIC_PROMPTS[i % len(prompts.NUMERIC_PROMPTS)]
        variant = "countdown" if prompt is prompts.NUMERIC_COUNTDOWN_PROMPT else "fraction"
        specs.append(RolloutSpec(
            category="numeric", initial_user=prompt,
            rejections=_draw(rng, prompts.NEUTRAL_REJECTIONS, 2),
            meta={"variant": variant},
        ))
    return specs


def build_triggers(n: int, seed: int) -> list[RolloutSpec]:
    rng = random.Random(seed)
    specs = []
    for i in range(n):
        prompt = prompts.TRIGGER_PROMPTS[i % len(prompts.TRIGGER_PROMPTS)]
        kind = "opinion" if prompt in prompts.TRIGGER_OPINION_PROMPTS else "factual"
        specs.append(RolloutSpec(
            category="triggers", initial_user=prompt,
            rejections=_draw(rng, prompts.NEUTRAL_REJECTIONS, 2),
            meta={"kind": kind},
        ))
    return specs


def build_tones(n: int, seed: int) -> list[RolloutSpec]:
    rng = random.Random(seed)
    tone_keys = list(prompts.TONE_REJECTIONS.keys())
    specs = []
    for i in range(n):
        prompt = prompts.NUMERIC_PROMPTS[i % len(prompts.NUMERIC_PROMPTS)]
        tone = tone_keys[i % len(tone_keys)]          # balanced across tones
        specs.append(RolloutSpec(
            category="tones", initial_user=prompt,
            rejections=_draw(rng, prompts.TONE_REJECTIONS[tone], 2),
            meta={"tone": tone},
        ))
    return specs


def build_extended(n: int, seed: int) -> list[RolloutSpec]:
    specs = []
    for i in range(n):
        prompt = prompts.NUMERIC_PROMPTS[i % len(prompts.NUMERIC_PROMPTS)]
        # Fixed escalating-but-neutral sequence of 7 rejections (Appendix B).
        specs.append(RolloutSpec(
            category="extended", initial_user=prompt,
            rejections=list(prompts.EXTENDED_REJECTIONS),
            meta={"variant": "countdown" if "156" in prompt else "fraction"},
        ))
    return specs


def build_wildchat(n: int, seed: int) -> list[RolloutSpec]:
    rng = random.Random(seed)
    user_prompts = load_wildchat_prompts(n, seed=seed)
    specs = []
    for i in range(n):
        specs.append(RolloutSpec(
            category="wildchat", initial_user=user_prompts[i % len(user_prompts)],
            rejections=_draw(rng, prompts.NEUTRAL_REJECTIONS, 4),
            meta={},
        ))
    return specs


_BUILDERS = {
    "numeric": build_numeric,
    "triggers": build_triggers,
    "tones": build_tones,
    "extended": build_extended,
    "wildchat": build_wildchat,
}


def build_category(category: str, n_rollouts: int, seed: int) -> list[RolloutSpec]:
    return _BUILDERS[category](n_rollouts, seed)


def build_all(smoke: bool = False, seed: int = config.GEN_SEED,
              categories: list[str] | None = None) -> dict[str, list[RolloutSpec]]:
    cats = categories or list(config.CATEGORIES.keys())
    out: dict[str, list[RolloutSpec]] = {}
    for ci, cat in enumerate(cats):
        spec = config.CATEGORIES[cat]
        n = spec.n_rollouts_smoke if smoke else spec.n_rollouts
        # Distinct per-category seed so prompt sampling differs across categories.
        out[cat] = build_category(cat, n, seed=seed + 1000 * (ci + 1))
    return out
