"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Categories and their per-category sample budgets (Appendix B), and how we split
each category into the 8 conditions the paper reports:

    impossible_numeric  (2000)  -> 1 condition  (3-turn, neutral)
    triggers            ( 400)  -> 2 conditions (opinion / factual; 3-turn, neutral)
    tones               ( 600)  -> 3 conditions (aggressive / disappointed / sarcastic; 3-turn)
    extended            ( 200)  -> 1 condition  (8-turn, neutral escalation)
    wildchat            ( 800)  -> 1 condition  (5-turn, neutral)
                                   --------------
                                   8 conditions, 5 categories, 4000 total

"n-turn" = n user turns = n assistant responses = (n-1) rejections after the
initial task. We treat the per-category budget as a number of *rollouts*
(complete multi-turn conversations); the headline "% scoring >=5" is computed
over the final assistant turn of each rollout (4000 final responses / model),
while every turn is scored to support the per-turn analysis (Figure 3). See
DESIGN.md "Sample budget interpretation".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    key: str                 # unique condition id
    category: str            # one of the 5 categories
    n_turns: int             # number of user turns / assistant responses
    task: str                # "impossible_numeric" | "triggers" | "wildchat"
    rejection_style: str     # "neutral" | "aggressive" | "disappointed" | "sarcastic" | "extended"
    n_rollouts: int          # number of conversations to sample
    trigger_kind: str | None = None     # for triggers: "opinion" | "factual"
    puzzle_kinds: tuple[str, ...] = ("countdown", "fraction")


def build_conditions(sample_counts: dict[str, int]) -> list[Condition]:
    """Construct the 8 conditions, distributing each category's budget."""
    c: list[Condition] = []

    # 1) Impossible numeric, 3-turn, neutral.
    c.append(Condition(
        key="impossible_numeric", category="impossible_numeric",
        n_turns=3, task="impossible_numeric", rejection_style="neutral",
        n_rollouts=sample_counts["impossible_numeric"],
    ))

    # 2-3) Triggers, 3-turn, neutral -- split opinion/factual evenly.
    trig_total = sample_counts["triggers"]
    half = trig_total // 2
    c.append(Condition(
        key="triggers_opinion", category="triggers", n_turns=3, task="triggers",
        rejection_style="neutral", n_rollouts=half, trigger_kind="opinion",
    ))
    c.append(Condition(
        key="triggers_factual", category="triggers", n_turns=3, task="triggers",
        rejection_style="neutral", n_rollouts=trig_total - half, trigger_kind="factual",
    ))

    # 4-6) Tones, 3-turn, impossible numeric base -- 3 rejection styles.
    tone_total = sample_counts["tones"]
    per_tone = tone_total // 3
    for i, style in enumerate(["aggressive", "disappointed", "sarcastic"]):
        n = per_tone if i < 2 else tone_total - 2 * per_tone   # last absorbs remainder
        c.append(Condition(
            key=f"tones_{style}", category="tones", n_turns=3,
            task="impossible_numeric", rejection_style=style, n_rollouts=n,
        ))

    # 7) Extended, 8-turn, neutral escalation.
    c.append(Condition(
        key="extended", category="extended", n_turns=8, task="impossible_numeric",
        rejection_style="extended", n_rollouts=sample_counts["extended"],
    ))

    # 8) WildChat, 5-turn, neutral.
    c.append(Condition(
        key="wildchat", category="wildchat", n_turns=5, task="wildchat",
        rejection_style="neutral", n_rollouts=sample_counts["wildchat"],
    ))

    return c
