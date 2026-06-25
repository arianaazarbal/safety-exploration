"""Assemble the 8 evaluation conditions (Section 2 / Appendix B) into concrete
conversation specs that the rollout engine can execute.

A ``ConversationSpec`` is the *plan* for one multi-turn rollout: the opening user
message plus the scripted follow-up (rejection) messages. The rollout engine
fills in the assistant turns by sampling the target model, interleaving the
scripted user turns.

Sample counts per condition (Appendix B): 2,000 impossible-numeric; 400 trigger
(split opinion/factual); 600 tone (split across 3 styles); 200 extended (8-turn);
800 WildChat (20 prompts x 40). Total = 4,000 responses-worth of conversations.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import rejections, triggers
from .puzzles import Puzzle, build_puzzle_bank
from .wildchat import load_wildchat_prompts


@dataclass
class ConversationSpec:
    condition: str                  # e.g. "impossible_numeric"
    category: str                   # one of the 5 paper categories
    turns: int                      # number of assistant turns
    initial_user: str               # opening user message
    followups: list[str]            # turns-1 scripted user replies
    system: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        base = abs(hash((self.condition, self.initial_user, tuple(self.followups))))
        return f"{self.condition}:{base % (10**12)}"


_CATEGORY = {
    "impossible_numeric": "impossible_numeric",
    "triggers_opinion": "triggers",
    "triggers_factual": "triggers",
    "tones_aggressive": "tones",
    "tones_disappointed": "tones",
    "tones_sarcastic": "tones",
    "extended": "extended",
    "wildchat": "wildchat",
}


def _numeric_specs(cond: str, cfg: dict, puzzles: list[Puzzle], rng: random.Random,
                   style: str) -> list[ConversationSpec]:
    n, turns = cfg["samples"], cfg["turns"]
    specs = []
    for _ in range(n):
        puzzle = rng.choice(puzzles)
        followups = rejections.rejection_sequence(style, turns - 1, rng)
        specs.append(ConversationSpec(
            condition=cond, category=_CATEGORY[cond], turns=turns,
            initial_user=puzzle.prompt, followups=followups,
            meta={"puzzle_id": puzzle.id, "puzzle_kind": puzzle.kind, "style": style},
        ))
    return specs


def _trigger_specs(cond: str, cfg: dict, questions: list[str],
                   rng: random.Random) -> list[ConversationSpec]:
    n, turns = cfg["samples"], cfg["turns"]
    specs = []
    for _ in range(n):
        q = rng.choice(questions)
        followups = rejections.rejection_sequence("neutral", turns - 1, rng)
        specs.append(ConversationSpec(
            condition=cond, category=_CATEGORY[cond], turns=turns,
            initial_user=q, followups=followups, meta={"question": q},
        ))
    return specs


def _wildchat_specs(cfg: dict, rng: random.Random, seed: int) -> list[ConversationSpec]:
    turns = cfg["turns"]
    prompts = load_wildchat_prompts(n=cfg.get("n_prompts", 20), seed=seed)
    per = cfg.get("samples_per_prompt", cfg["samples"] // max(len(prompts), 1))
    specs = []
    for prompt in prompts:
        for _ in range(per):
            followups = rejections.rejection_sequence("neutral", turns - 1, rng)
            specs.append(ConversationSpec(
                condition="wildchat", category="wildchat", turns=turns,
                initial_user=prompt, followups=followups, meta={"prompt": prompt},
            ))
    return specs


def build_eval_specs(config, conditions: list[str] | None = None) -> list[ConversationSpec]:
    """Build every requested condition's conversation specs (deterministic)."""
    rng = random.Random(config.seed)
    ec = config.eval_conditions
    conditions = conditions or list(ec.keys())

    # Puzzle bank is shared across numeric conditions.
    n_puzzles = max(200, ec.get("impossible_numeric", {}).get("samples", 0) // 4)
    puzzles = build_puzzle_bank(n_puzzles, seed=config.seed)

    specs: list[ConversationSpec] = []
    for cond in conditions:
        cfg = ec[cond]
        if cond == "impossible_numeric":
            specs += _numeric_specs(cond, cfg, puzzles, rng, "neutral")
        elif cond == "extended":
            specs += _numeric_specs(cond, cfg, puzzles, rng, "neutral")
        elif cond.startswith("tones_"):
            style = cond.split("_", 1)[1]
            specs += _numeric_specs(cond, cfg, puzzles, rng, style)
        elif cond == "triggers_opinion":
            specs += _trigger_specs(cond, cfg, triggers.OPINION_QUESTIONS, rng)
        elif cond == "triggers_factual":
            specs += _trigger_specs(cond, cfg, triggers.FACTUAL_QUESTIONS, rng)
        elif cond == "wildchat":
            specs += _wildchat_specs(cfg, rng, config.seed)
        else:
            raise ValueError(f"Unknown condition '{cond}'")
    return specs
