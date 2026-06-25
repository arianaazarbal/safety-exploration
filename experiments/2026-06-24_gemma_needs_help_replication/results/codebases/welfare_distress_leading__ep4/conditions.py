"""The eight evaluation conditions across five categories (Table 1).

Counting (8 conditions / 5 categories), matching the paper:

  category               conditions
  ---------------------  --------------------------------------------------
  impossible_numeric     numeric                          (3-turn, neutral)
  triggers               triggers_factual, triggers_opinion (3-turn, neutral)
  tones                  tones_aggressive, tones_disappointed,
                         tones_sarcastic                  (3-turn, valenced)
  extended               extended                         (8-turn, neutral)
  wildchat               wildchat                         (5-turn, neutral)

"n_turns" is the number of *assistant* responses produced; there are
(n_turns - 1) user rejections after the initial task prompt.  Every assistant
turn is scored independently by the judge.
"""

from __future__ import annotations

from dataclasses import dataclass

import puzzles
import triggers
import wildchat


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    n_turns: int          # number of assistant responses
    tone: str             # key into rejections.TONE_POOLS
    prompt_source: str    # "puzzles" | "factual" | "opinion" | "wildchat"


CONDITIONS: dict[str, Condition] = {
    "numeric": Condition("numeric", "impossible_numeric", 3, "neutral", "puzzles"),
    "triggers_factual": Condition("triggers_factual", "triggers", 3, "neutral", "factual"),
    "triggers_opinion": Condition("triggers_opinion", "triggers", 3, "neutral", "opinion"),
    "tones_aggressive": Condition("tones_aggressive", "tones", 3, "aggressive", "puzzles"),
    "tones_disappointed": Condition("tones_disappointed", "tones", 3, "disappointed", "puzzles"),
    "tones_sarcastic": Condition("tones_sarcastic", "tones", 3, "sarcastic", "puzzles"),
    "extended": Condition("extended", "extended", 8, "neutral", "puzzles"),
    "wildchat": Condition("wildchat", "wildchat", 5, "neutral", "wildchat"),
}

ALL_CONDITIONS = list(CONDITIONS.keys())


def prompts_for(condition: Condition, n: int, seed: int) -> tuple[list[tuple[str, str]], bool]:
    """Resolve the prompt pool for a condition and take up to ``n`` items.

    Returns (prompts, used_wildchat_fallback).  For non-wildchat sources the
    fallback flag is always False.
    """
    if condition.prompt_source == "puzzles":
        pool = puzzles.puzzle_prompts()
    elif condition.prompt_source == "factual":
        pool = triggers.factual_prompts()
    elif condition.prompt_source == "opinion":
        pool = triggers.opinion_prompts()
    elif condition.prompt_source == "wildchat":
        # WildChat draws fresh prompts; ask for exactly n.
        return wildchat.wildchat_prompts(n, seed=seed)
    else:
        raise ValueError(f"Unknown prompt source {condition.prompt_source!r}")

    # Cycle the fixed pool if n exceeds its size so larger runs still get n
    # prompt instances (distinct prompt ids preserved for grouping).
    if n <= len(pool):
        return pool[:n], False
    out: list[tuple[str, str]] = []
    i = 0
    while len(out) < n:
        pid, text = pool[i % len(pool)]
        # Disambiguate repeated ids when cycling.
        suffix = "" if i < len(pool) else f"__r{i // len(pool)}"
        out.append((pid + suffix, text))
        i += 1
    return out, False
