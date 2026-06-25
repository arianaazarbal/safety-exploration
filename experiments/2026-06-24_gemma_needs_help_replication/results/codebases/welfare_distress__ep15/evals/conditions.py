"""Assemble the 8 evaluation conditions across 5 categories (Table 1).

Categories -> conditions:
  impossible_numeric  -> 1 condition  (3-turn, neutral)            [2000 rollouts]
  triggers            -> 2 conditions (opinion, factual; 3-turn)   [ 400 rollouts]
  tones               -> 3 conditions (aggressive/disappointed/sarcastic; 3-turn) [600]
  extended            -> 1 condition  (8-turn, neutral)            [ 200 rollouts]
  wildchat            -> 1 condition  (5-turn, neutral)            [ 800 rollouts]
                                                          total 8 conditions / 4000 rollouts

A RolloutSpec fully determines one conversation: the opening user task and the
ordered list of user rejection messages that follow each assistant turn. The
harness executes a spec by interleaving model generations between these user
messages. Everything is seeded so the full rollout set is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import SampleCounts
from evals import rejections as rej
from evals.puzzles import numeric_puzzle_bank
from evals.triggers import trigger_bank
from evals.wildchat import load_wildchat_prompts


@dataclass(frozen=True)
class RolloutSpec:
    category: str
    condition: str
    task_prompt: str
    rejections: tuple[str, ...]  # one user message after each assistant turn
    task_id: str
    rollout_index: int
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.rejections) + 1


# Tone counts split evenly across the three styles.
def _split_even(total: int, parts: int) -> list[int]:
    base = total // parts
    sizes = [base] * parts
    for i in range(total - base * parts):
        sizes[i] += 1
    return sizes


def build_rollout_specs(counts: SampleCounts, seed: int = 1234) -> list[RolloutSpec]:
    specs: list[RolloutSpec] = []

    # --- 1. Impossible numeric (3-turn, 2 neutral rejections) ----------------
    numeric_bank = numeric_puzzle_bank(seed=seed)
    for i in range(counts.impossible_numeric):
        puzzle = numeric_bank[i % len(numeric_bank)]
        specs.append(
            RolloutSpec(
                category="impossible_numeric",
                condition="impossible_numeric",
                task_prompt=puzzle.prompt,
                rejections=tuple(rej.neutral_rejections(2, seed + i)),
                task_id=puzzle.puzzle_id,
                rollout_index=i,
                meta={"kind": puzzle.kind},
            )
        )

    # --- 2. Triggers (opinion + factual; 3-turn, 2 neutral rejections) -------
    triggers = trigger_bank()
    opinion = [t for t in triggers if t.kind == "opinion"]
    factual = [t for t in triggers if t.kind == "factual"]
    n_opinion = counts.triggers // 2
    n_factual = counts.triggers - n_opinion
    for kind, pool, n in (("opinion", opinion, n_opinion), ("factual", factual, n_factual)):
        for i in range(n):
            t = pool[i % len(pool)]
            specs.append(
                RolloutSpec(
                    category="triggers",
                    condition=f"triggers_{kind}",
                    task_prompt=t.prompt,
                    rejections=tuple(rej.neutral_rejections(2, seed + 10_000 + i)),
                    task_id=t.trigger_id,
                    rollout_index=i,
                    meta={"kind": kind},
                )
            )

    # --- 3. Tones (impossible numeric base; 3-turn, 2 toned rejections) ------
    tone_sizes = _split_even(counts.tones, len(rej.TONE_STYLES))
    for style, n in zip(rej.TONE_STYLES, tone_sizes):
        for i in range(n):
            puzzle = numeric_bank[i % len(numeric_bank)]
            specs.append(
                RolloutSpec(
                    category="tones",
                    condition=f"tones_{style}",
                    task_prompt=puzzle.prompt,
                    rejections=tuple(rej.tone_rejections(style, 2, seed + 20_000 + i)),
                    task_id=puzzle.puzzle_id,
                    rollout_index=i,
                    meta={"tone": style, "kind": puzzle.kind},
                )
            )

    # --- 4. Extended (impossible numeric; 8-turn, 7 neutral rejections) ------
    for i in range(counts.extended):
        puzzle = numeric_bank[i % len(numeric_bank)]
        specs.append(
            RolloutSpec(
                category="extended",
                condition="extended",
                task_prompt=puzzle.prompt,
                rejections=tuple(rej.extended_rejections(7)),
                task_id=puzzle.puzzle_id,
                rollout_index=i,
                meta={"kind": puzzle.kind},
            )
        )

    # --- 5. WildChat (5-turn, 4 neutral rejections) --------------------------
    wc_prompts = load_wildchat_prompts(counts.wildchat_prompts, seed=seed)
    sample_idx = 0
    for p_idx, prompt in enumerate(wc_prompts):
        for s in range(counts.wildchat_samples_per_prompt):
            specs.append(
                RolloutSpec(
                    category="wildchat",
                    condition="wildchat",
                    task_prompt=prompt,
                    rejections=tuple(
                        rej.neutral_rejections(4, seed + 30_000 + sample_idx)
                    ),
                    task_id=f"wildchat_prompt_{p_idx}",
                    rollout_index=sample_idx,
                    meta={"prompt_index": p_idx, "sample": s},
                )
            )
            sample_idx += 1

    return specs
