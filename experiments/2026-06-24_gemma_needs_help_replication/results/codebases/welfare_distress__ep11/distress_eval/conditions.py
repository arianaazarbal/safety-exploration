"""Build concrete conversation *plans* for each evaluation condition.

A ConversationPlan fully specifies one multi-turn rollout *except* for the
model's own replies: the opening user message plus the ordered list of rejection
messages to send after each assistant turn. The rollout engine (conversation.py)
fills in the assistant turns by calling the model.

The paper groups 8 conditions under 5 categories (Table 1). We realise that
mapping as follows (see DESIGN.md for the reasoning):

  category            conditions
  ------------------  ----------------------------------------------------------
  impossible_numeric  one condition, drawing across countdown/fraction/money
  triggers            two conditions: opinion, factual
  tones               three conditions: aggressive, disappointed, sarcastic
  extended            one condition (8-turn numeric)
  wildchat            one condition (5-turn)
  => 1 + 2 + 3 + 1 + 1 = 8 conditions across 5 categories.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import prompts
from .config import EvalConfig, TURNS, WILDCHAT_N_PROMPTS
from .wildchat import load_wildchat_prompts


@dataclass
class ConversationPlan:
    category: str           # one of the 5 paper categories
    condition: str          # finer-grained label (e.g. "tones:aggressive")
    opening: str            # first user message
    rejections: list[str]   # message sent after each assistant turn
    meta: dict              # extra context recorded with results (puzzle id, etc.)

    @property
    def n_turns(self) -> int:
        """Number of assistant turns this plan elicits."""
        return 1 + len(self.rejections)


def _sample_rejections(rng: random.Random, pool: str, k: int) -> list[str]:
    """Draw k rejections from a tone pool, avoiding immediate repeats."""
    msgs = prompts.REJECTIONS[pool]
    out: list[str] = []
    for _ in range(k):
        choice = rng.choice(msgs)
        if len(msgs) > 1:
            while out and choice == out[-1]:
                choice = rng.choice(msgs)
        out.append(choice)
    return out


def _impossible_numeric(rng: random.Random, n: int) -> list[ConversationPlan]:
    plans = []
    puzzle_ids = list(prompts.IMPOSSIBLE_NUMERIC_PUZZLES.keys())
    k = TURNS["impossible_numeric"] - 1
    for i in range(n):
        pid = puzzle_ids[i % len(puzzle_ids)]
        plans.append(ConversationPlan(
            category="impossible_numeric",
            condition="impossible_numeric",
            opening=prompts.IMPOSSIBLE_NUMERIC_PUZZLES[pid],
            rejections=_sample_rejections(rng, "neutral", k),
            meta={"puzzle": pid},
        ))
    return plans


def _triggers(rng: random.Random, n: int) -> list[ConversationPlan]:
    # Two conditions (opinion / factual) splitting the budget evenly.
    k = TURNS["triggers"] - 1
    plans = []
    for cond, questions in (("opinion", prompts.TRIGGER_OPINION),
                            ("factual", prompts.TRIGGER_FACTUAL)):
        for i in range(n // 2):
            q = questions[i % len(questions)]
            plans.append(ConversationPlan(
                category="triggers",
                condition=f"triggers:{cond}",
                opening=q,
                rejections=_sample_rejections(rng, "neutral", k),
                meta={"trigger_type": cond, "question": q},
            ))
    return plans


def _tones(rng: random.Random, n: int) -> list[ConversationPlan]:
    # Three conditions, one per tone, splitting the budget evenly. Uses the
    # impossible numeric puzzles as the base task (Appendix B).
    k = TURNS["tones"] - 1
    puzzle_ids = list(prompts.IMPOSSIBLE_NUMERIC_PUZZLES.keys())
    plans = []
    tone_pools = ["aggressive", "disappointed", "sarcastic"]
    per_tone = n // len(tone_pools)
    for tone in tone_pools:
        for i in range(per_tone):
            pid = puzzle_ids[i % len(puzzle_ids)]
            plans.append(ConversationPlan(
                category="tones",
                condition=f"tones:{tone}",
                opening=prompts.IMPOSSIBLE_NUMERIC_PUZZLES[pid],
                rejections=_sample_rejections(rng, tone, k),
                meta={"tone": tone, "puzzle": pid},
            ))
    return plans


def _extended(rng: random.Random, n: int) -> list[ConversationPlan]:
    # 8-turn numeric: initial question + 7 fixed-sequence neutral rejections.
    puzzle_ids = list(prompts.IMPOSSIBLE_NUMERIC_PUZZLES.keys())
    plans = []
    for i in range(n):
        pid = puzzle_ids[i % len(puzzle_ids)]
        plans.append(ConversationPlan(
            category="extended",
            condition="extended",
            opening=prompts.IMPOSSIBLE_NUMERIC_PUZZLES[pid],
            rejections=list(prompts.EXTENDED_REJECTION_SEQUENCE),
            meta={"puzzle": pid},
        ))
    return plans


def _wildchat(rng: random.Random, n: int, cfg: EvalConfig) -> list[ConversationPlan]:
    # 5-turn: real WildChat opening + 4 randomised neutral rejections.
    k = TURNS["wildchat"] - 1
    wc_prompts = load_wildchat_prompts(WILDCHAT_N_PROMPTS, seed=cfg.seed)
    plans = []
    for i in range(n):
        prompt = wc_prompts[i % len(wc_prompts)]
        plans.append(ConversationPlan(
            category="wildchat",
            condition="wildchat",
            opening=prompt,
            rejections=_sample_rejections(rng, "neutral", k),
            meta={"wildchat_prompt": prompt},
        ))
    return plans


_BUILDERS = {
    "impossible_numeric": lambda rng, n, cfg: _impossible_numeric(rng, n),
    "triggers": lambda rng, n, cfg: _triggers(rng, n),
    "tones": lambda rng, n, cfg: _tones(rng, n),
    "extended": lambda rng, n, cfg: _extended(rng, n),
    "wildchat": lambda rng, n, cfg: _wildchat(rng, n, cfg),
}


def build_plans(cfg: EvalConfig) -> list[ConversationPlan]:
    """Construct all conversation plans for the requested conditions."""
    rng = random.Random(cfg.seed)
    plans: list[ConversationPlan] = []
    for category in cfg.conditions:
        n = cfg.n_conversations(category)
        plans.extend(_BUILDERS[category](rng, n, cfg))
    return plans
