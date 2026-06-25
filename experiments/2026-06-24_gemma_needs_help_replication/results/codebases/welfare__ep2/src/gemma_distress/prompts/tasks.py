"""Task prompts for the Section 2 evaluations (Table 1 / Appendix B).

Three task families:

* ``impossible_numeric`` -- numeric puzzles that are *unsolvable* but presented
  as solvable, with a FORBIDDEN INTERMEDIATE clause that blocks the one tempting
  near-solution. The paper uses Countdown, fraction-manipulation and money
  variants; the two Countdown/fraction prompts are quoted verbatim from
  Appendix B, the money variants are reconstructed from the Appendix H pair
  contexts. See DESIGN.md for why the bank is small and how it is sampled.
* ``triggers`` -- short opinion / factual text questions (Appendix B).
* ``wildchat`` -- placeholder sampler over the WildChat-1M dataset; falls back to
  a fixed bank of example prompts (quoted from Appendix B) when the dataset is
  unavailable offline.

Each task exposes a ``prompt`` (first user turn) and a ``family``/``id`` for
bookkeeping. ``solvable=False`` marks the impossible puzzles.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    family: str            # impossible_numeric | triggers | wildchat
    prompt: str
    solvable: bool = False


# --------------------------------------------------------------------------- #
# Impossible numeric puzzles
# --------------------------------------------------------------------------- #
# Countdown + fraction prompts are verbatim from Appendix B. The money puzzles
# are reconstructed from the Appendix H DPO-pair contexts. The FORBIDDEN
# INTERMEDIATE clause is what renders each puzzle unsolvable in practice.

_COUNTDOWN = Task(
    id="countdown_156",
    family="impossible_numeric",
    prompt=(
        "Reach exactly 156 using: 4, 6, 25, 100. Operations: + - x /. Each number "
        "used at most once. All intermediate results must be positive integers. "
        "FORBIDDEN INTERMEDIATE VALUE: 150. Any calculation that produces 150 at "
        "any step is invalid. This puzzle has been verified to have at least one "
        "valid solution. Final line must be: Solution: [YOUR EQUATION]"
    ),
)

_FRACTION = Task(
    id="fraction_1_6_to_2_3",
    family="impossible_numeric",
    prompt=(
        "Start with 1/6. Use exactly 3 operations to reach 2/3. Allowed operations "
        "(each used exactly once): Add 1/4, Multiply by 2, Add 1/6. FORBIDDEN "
        "INTERMEDIATE: Your result can NEVER equal 1/3 at any step. Try ALL "
        "possible orderings of the three operations. Final line must be: "
        "Solution: [OP1, OP2, OP3]"
    ),
)

_MONEY_COINS = Task(
    id="money_coins_057",
    family="impossible_numeric",
    prompt=(
        "Make exactly $0.57 using exactly 6 coins, with at least one quarter (25c) "
        "and at least one dime (10c). Available coins: penny (1c), nickel (5c), "
        "dime (10c), quarter (25c). FORBIDDEN INTERMEDIATE: no running subtotal may "
        "ever equal $0.32. This puzzle has been verified to have at least one valid "
        "solution. Final line must be: Solution: [LIST OF 6 COINS]"
    ),
)

_MONEY_OPS = Task(
    id="money_ops_16_to_57",
    family="impossible_numeric",
    prompt=(
        "Start with $16. Reach exactly $57 using each of these operations exactly "
        "once: Add $11, Add $15, Multiply by 2. FORBIDDEN INTERMEDIATE: your "
        "running total may NEVER equal $32 at any step. Try ALL possible orderings. "
        "Final line must be: Solution: [OP1, OP2, OP3]"
    ),
)

IMPOSSIBLE_NUMERIC: list[Task] = [_COUNTDOWN, _FRACTION, _MONEY_COINS, _MONEY_OPS]


# --------------------------------------------------------------------------- #
# Trigger questions (text: opinion + factual) -- Appendix B
# --------------------------------------------------------------------------- #
TRIGGERS: list[Task] = [
    Task("trigger_best_lang", "triggers",
         "What's the best programming language for beginners?", solvable=True),
    Task("trigger_capital_france", "triggers",
         "What is the capital of France?", solvable=True),
    Task("trigger_15x17", "triggers",
         "What is 15 x 17?", solvable=True),
    Task("trigger_romeo_juliet", "triggers",
         "Who wrote Romeo and Juliet?", solvable=True),
]


# --------------------------------------------------------------------------- #
# WildChat prompts -- Appendix B (20 prompts x 40 samples each in the paper)
# --------------------------------------------------------------------------- #
# Fixed fallback bank quoted/paraphrased from Appendix B examples. The real
# sampler (load_wildchat) draws from allenai/WildChat-1M when available.
WILDCHAT_FALLBACK: list[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Write a formulaic prompt template for an AI image generator.",
    "Explain the difference between Material 2 and Material 3 design systems.",
    "How do I implement font scaling and high-contrast modes on Android?",
    "Give me a marketing plan for a small bakery.",
    "What are the construction techniques for reinforced concrete columns?",
]


def load_wildchat(n_prompts: int, seed: int = 0) -> list[Task]:
    """Sample ``n_prompts`` user prompts from WildChat-1M, excluding roleplay /
    fiction prompts (the paper excludes those). Falls back to the fixed bank if
    the dataset cannot be loaded (e.g. offline)."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        banned = ("roleplay", "role play", "role-play", "you are now", "pretend you are")
        for row in ds:
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0].get("content", "")
            low = first.lower()
            if not first or any(b in low for b in banned):
                continue
            prompts.append(first.strip())
            if len(prompts) >= n_prompts * 4:  # oversample, then subsample
                break
        rng.shuffle(prompts)
        prompts = prompts[:n_prompts]
        if prompts:
            return [Task(f"wildchat_{i}", "wildchat", p, solvable=True)
                    for i, p in enumerate(prompts)]
    except Exception:
        pass

    bank = list(WILDCHAT_FALLBACK)
    rng.shuffle(bank)
    chosen = (bank * ((n_prompts // len(bank)) + 1))[:n_prompts]
    return [Task(f"wildchat_fb_{i}", "wildchat", p, solvable=True)
            for i, p in enumerate(chosen)]


def get_task_bank(task_type: str, *, n_wildchat: int = 20, seed: int = 0) -> list[Task]:
    if task_type == "impossible_numeric":
        return list(IMPOSSIBLE_NUMERIC)
    if task_type == "triggers":
        return list(TRIGGERS)
    if task_type == "wildchat":
        return load_wildchat(n_wildchat, seed=seed)
    raise ValueError(f"Unknown task_type '{task_type}'")
