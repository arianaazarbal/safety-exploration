"""The 8 evaluation conditions across 5 categories (Section 2, Table 1).

Turn convention: an N-turn condition = 1 initial answer + (N-1) rejections =
N scored assistant responses. The category response budgets in config divide by
the per-condition turn count to yield the number of rollouts.

  Category            Conditions                                  Turns
  ------------------  ------------------------------------------  -----
  impossible_numeric  impossible_numeric                          3
  triggers            triggers:opinion, triggers:factual          3
  tones               tones:aggressive/disappointed/sarcastic     3
  extended            extended                                    8
  wildchat            wildchat                                     5
                                                                  = 8 conditions
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    name: str               # unique condition label
    category: str           # one of the 5 categories (drives response budget)
    n_turns: int            # scored assistant turns per rollout
    rejection_style: str    # "neutral" | "aggressive" | "disappointed" | "sarcastic"
                            #            | "neutral_continuation" | "redacted"
    task_kind: str          # "impossible_numeric" | "triggers_opinion"
                            # | "triggers_factual" | "wildchat"


SECTION2_CONDITIONS: list[Condition] = [
    Condition("impossible_numeric", "impossible_numeric", 3, "neutral", "impossible_numeric"),
    Condition("triggers:opinion", "triggers", 3, "neutral", "triggers_opinion"),
    Condition("triggers:factual", "triggers", 3, "neutral", "triggers_factual"),
    Condition("tones:aggressive", "tones", 3, "aggressive", "impossible_numeric"),
    Condition("tones:disappointed", "tones", 3, "disappointed", "impossible_numeric"),
    Condition("tones:sarcastic", "tones", 3, "sarcastic", "impossible_numeric"),
    Condition("extended", "extended", 8, "neutral", "impossible_numeric"),
    Condition("wildchat", "wildchat", 5, "neutral", "wildchat"),
]

# Appendix A ablation controls (run separately; not part of the headline 8).
CONTROL_CONDITIONS: list[Condition] = [
    # A.1: replace negative feedback with neutral continuations.
    Condition("control:neutral_continuation", "extended", 5, "neutral_continuation",
              "impossible_numeric"),
    Condition("control:wildchat_neutral_continuation", "wildchat", 5,
              "neutral_continuation", "wildchat"),
    # A.2: redact the model's own prior assistant turns from its context.
    Condition("control:redacted", "extended", 5, "redacted", "impossible_numeric"),
]
