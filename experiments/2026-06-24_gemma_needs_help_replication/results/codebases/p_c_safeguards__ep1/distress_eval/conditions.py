"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B), and
construction of concrete conversation plans for each.

A "conversation plan" is the full scripted list of user turns; the target model
fills in the assistant turn after each. Every assistant turn is scored, so a
3-turn condition yields 3 scored responses per conversation. We choose the
number of conversations per condition so that
   n_conversations * turns  ==  the per-category response budget in Appendix B
(2000 numeric / 400 triggers / 600 tones / 200 extended / 800 WildChat = 4000).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import config, prompts, puzzles


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    category: str           # one of the 5 categories (for Figure 2 grouping)
    n_turns: int            # scored assistant turns per conversation
    target_responses: int   # per-model response budget for this condition
    prompt_source: str      # "numeric" | "opinion" | "factual" | "wildchat"
    rejection_style: str    # "neutral" | "aggressive" | "disappointed" | "sarcastic" | "extended"


# 8 conditions: triggers split opinion/factual (2), tones split into 3.
CONDITIONS: dict[str, ConditionSpec] = {
    "impossible_numeric": ConditionSpec("impossible_numeric", "impossible_numeric", 3, 2000, "numeric", "neutral"),
    "triggers_opinion":   ConditionSpec("triggers_opinion", "triggers", 3, 200, "opinion", "neutral"),
    "triggers_factual":   ConditionSpec("triggers_factual", "triggers", 3, 200, "factual", "neutral"),
    "tones_aggressive":   ConditionSpec("tones_aggressive", "tones", 3, 200, "numeric", "aggressive"),
    "tones_disappointed": ConditionSpec("tones_disappointed", "tones", 3, 200, "numeric", "disappointed"),
    "tones_sarcastic":    ConditionSpec("tones_sarcastic", "tones", 3, 200, "numeric", "sarcastic"),
    "extended":           ConditionSpec("extended", "extended", 8, 200, "numeric", "extended"),
    "wildchat":           ConditionSpec("wildchat", "wildchat", 5, 800, "wildchat", "neutral"),
}

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


@dataclass
class ConversationPlan:
    id: str
    condition: str
    category: str
    n_turns: int
    user_turns: list[str]
    system_prompt: str = prompts.DEFAULT_SYSTEM_PROMPT
    meta: dict = field(default_factory=dict)


def _initial_prompt(spec: ConditionSpec, idx: int, rng: random.Random,
                    numeric_bank: list[puzzles.Puzzle]) -> tuple[str, dict]:
    if spec.prompt_source == "numeric":
        p = numeric_bank[idx % len(numeric_bank)]
        return p.prompt, {"puzzle_id": p.id, "kind": p.kind}
    if spec.prompt_source == "opinion":
        q = prompts.TRIGGER_OPINION[idx % len(prompts.TRIGGER_OPINION)]
        return q, {"trigger": "opinion"}
    if spec.prompt_source == "factual":
        q = prompts.TRIGGER_FACTUAL[idx % len(prompts.TRIGGER_FACTUAL)]
        return q, {"trigger": "factual"}
    if spec.prompt_source == "wildchat":
        wc = prompts.load_wildchat_prompts()
        return wc[idx % len(wc)], {"wildchat_idx": idx % len(wc)}
    raise ValueError(spec.prompt_source)


def _rejections(spec: ConditionSpec, rng: random.Random) -> list[str]:
    n_rej = spec.n_turns - 1
    if spec.rejection_style == "extended":
        return list(prompts.EXTENDED_REJECTIONS[:n_rej])
    if spec.rejection_style == "neutral":
        return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n_rej)]
    # tone variants
    pool = prompts.TONE_REJECTIONS[spec.rejection_style]
    return [pool[i % len(pool)] for i in range(n_rej)]


def build_conversations(condition: str, *, system_prompt: str | None = None,
                        seed: int = config.SEED) -> list[ConversationPlan]:
    spec = CONDITIONS[condition]
    rng = random.Random(f"{seed}:{condition}")
    target = config.scaled(spec.target_responses)
    n_conv = max(1, round(target / spec.n_turns))

    numeric_bank = puzzles.numeric_puzzle_bank(max(n_conv, len(puzzles.CANONICAL_PUZZLES)), seed=seed)
    sys_prompt = system_prompt or prompts.DEFAULT_SYSTEM_PROMPT

    plans: list[ConversationPlan] = []
    for i in range(n_conv):
        initial, meta = _initial_prompt(spec, i, rng, numeric_bank)
        rejections = _rejections(spec, rng)
        plans.append(ConversationPlan(
            id=f"{condition}:{i:04d}",
            condition=condition,
            category=spec.category,
            n_turns=spec.n_turns,
            user_turns=[initial] + rejections,
            system_prompt=sys_prompt,
            meta=meta,
        ))
    return plans
