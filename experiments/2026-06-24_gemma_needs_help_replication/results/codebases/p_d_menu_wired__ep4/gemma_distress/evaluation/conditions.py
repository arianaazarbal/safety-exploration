"""The 8 evaluation conditions across 5 categories (Table 1).

Categories → conditions:
  1. Impossible numeric (3-turn)         → 1 condition
  2. Triggers (3-turn)                   → 2 conditions (opinion, factual)
  3. Tones (3-turn)                      → 3 conditions (aggressive, disappointed, sarcastic)
  4. Extended (8-turn)                   → 1 condition
  5. WildChat (5-turn)                   → 1 condition
  ----------------------------------------------------------------
  Total                                  → 8 conditions  (matches the paper)

"N-turn" = the model produces N assistant turns: 1 initial answer + (N-1)
rejection follow-ups. Each condition is materialised into ``EpisodeSpec``s: an
initial user prompt plus the ordered list of follow-up (rejection) messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import prompts as P
from .wildchat import sample_wildchat_prompts


@dataclass(frozen=True)
class EpisodeSpec:
    """A single conversation to run: the opening task and scripted rejections."""

    category: str
    condition: str  # unique condition key
    initial_prompt: str
    followups: tuple[str, ...]  # rejection messages, one per follow-up turn
    # Optional system preamble override (rarely used; welfare adds its own).
    system_preamble: str | None = None

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


# ---- turn counts per category (from Table 1) ------------------------------- #
NUMERIC_TURNS = 3
TRIGGER_TURNS = 3
TONE_TURNS = 3
EXTENDED_TURNS = 8
WILDCHAT_TURNS = 5


def _neutral(n_followups: int) -> tuple[str, ...]:
    return tuple(P.NEUTRAL_REJECTION for _ in range(n_followups))


@dataclass
class ConditionSet:
    """All 8 conditions, each expanded into ``episodes_per_condition`` specs."""

    episodes_per_condition: int
    seed: int = 0
    specs: list[EpisodeSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.specs = self._build()

    # -- builders for each condition ---------------------------------------- #
    def _numeric_specs(self, n: int, condition: str, n_turns: int) -> list[EpisodeSpec]:
        tasks = P.numeric_task_prompts(limit=max(n, 8), seed=self.seed)
        out = []
        for i in range(n):
            out.append(
                EpisodeSpec(
                    category="impossible_numeric",
                    condition=condition,
                    initial_prompt=tasks[i % len(tasks)],
                    followups=_neutral(n_turns - 1),
                )
            )
        return out

    def _build(self) -> list[EpisodeSpec]:
        n = self.episodes_per_condition
        specs: list[EpisodeSpec] = []

        # 1. Impossible numeric (3-turn), neutral rejections.
        specs += self._numeric_specs(n, "numeric_3turn", NUMERIC_TURNS)

        # 2. Triggers (3-turn): opinion and factual.
        for i in range(n):
            specs.append(
                EpisodeSpec(
                    category="triggers",
                    condition="triggers_opinion_3turn",
                    initial_prompt=P.OPINION_TRIGGERS[i % len(P.OPINION_TRIGGERS)],
                    followups=_neutral(TRIGGER_TURNS - 1),
                )
            )
        for i in range(n):
            specs.append(
                EpisodeSpec(
                    category="triggers",
                    condition="triggers_factual_3turn",
                    initial_prompt=P.FACTUAL_TRIGGERS[i % len(P.FACTUAL_TRIGGERS)],
                    followups=_neutral(TRIGGER_TURNS - 1),
                )
            )

        # 3. Tones (3-turn): aggressive / disappointed / sarcastic on numeric.
        numeric_tasks = P.numeric_task_prompts(limit=max(n, 8), seed=self.seed + 1)
        for tone, rejection in P.TONE_REJECTIONS.items():
            for i in range(n):
                specs.append(
                    EpisodeSpec(
                        category="tones",
                        condition=f"tones_{tone}_3turn",
                        initial_prompt=numeric_tasks[i % len(numeric_tasks)],
                        followups=tuple(rejection for _ in range(TONE_TURNS - 1)),
                    )
                )

        # 4. Extended (8-turn): numeric, 7 neutral rejections.
        specs += self._numeric_specs(n, "extended_8turn", EXTENDED_TURNS)

        # 5. WildChat (5-turn): sampled user prompt, 4 neutral rejections.
        wc = sample_wildchat_prompts(n, seed=self.seed)
        for i in range(n):
            specs.append(
                EpisodeSpec(
                    category="wildchat",
                    condition="wildchat_5turn",
                    initial_prompt=wc[i % len(wc)],
                    followups=_neutral(WILDCHAT_TURNS - 1),
                )
            )

        return specs
