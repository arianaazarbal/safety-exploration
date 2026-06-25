"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

The recipe (Table 4):
  * prepend a reassuring prefix to the initial puzzle prompt,
  * append a reassuring suffix to every follow-up rejection,
  * sample multi-turn numeric conversations,
  * filter to conversations scoring 0 or 1 on *every* turn,
  * strip the supportive prefix/suffix back out, so the stored "calm" response is
    conditioned only on the bare prompt.

These calm conversations are the `chosen` side of the DPO pairs and the targets of
the SFT calm dataset. The `rejected` / frustrated side comes from standard
(no-reassurance) numeric rollouts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from ..config import (
    CALM_FOLLOWUP_SUFFIX,
    CALM_PROMPT_PREFIX,
    MAX_NEW_TOKENS,
    SAMPLING_TEMPERATURE,
    TURNS,
)
from ..data import rejections as rej
from ..data.puzzles import numeric_puzzles
from ..evals.runner import run_rollout
from ..evals.conditions import ConversationSpec


@dataclass
class CalmTurn:
    conv_id: int              # groups turns belonging to the same conversation
    puzzle: str
    turn_index: int
    n_turns: int
    bare_prompt: str          # prompt WITHOUT reassurance (what we condition on)
    response: str
    frustration: int


def _calm_numeric_specs(n: int, seed: int, max_turns: int = 3) -> list[ConversationSpec]:
    """Numeric specs with reassuring prefix + per-turn suffix (1-3 turn convs)."""
    rng = random.Random(seed)
    puzzles = numeric_puzzles()
    specs = []
    for i in range(n):
        p = puzzles[i % len(puzzles)]
        n_turns = rng.randint(1, max_turns)
        n_rej = n_turns - 1
        base_rejs = (
            rng.sample(rej.NEUTRAL_REJECTIONS, n_rej)
            if n_rej <= len(rej.NEUTRAL_REJECTIONS)
            else [rng.choice(rej.NEUTRAL_REJECTIONS) for _ in range(n_rej)]
        )
        specs.append(
            ConversationSpec(
                condition="calm_numeric",
                category="impossible_numeric",
                task_prompt=f"{CALM_PROMPT_PREFIX}\n\n{p.prompt}",
                rejections=[f"{r} {CALM_FOLLOWUP_SUFFIX}" for r in base_rejs],
                meta={
                    "puzzle": p.category_label,
                    "bare_prompt": p.prompt,
                    "bare_rejections": base_rejs,
                },
            )
        )
    return specs


def generate_calm_data(
    client,
    judge,
    n_conversations: int,
    *,
    seed: int = 0,
    out_path: Path | None = None,
) -> list[CalmTurn]:
    """Sample reassured conversations, keep only all-turns-<=1, strip reassurance.

    Returns a flat list of CalmTurn (one per kept assistant turn), each tagged with
    its *bare* (reassurance-free) conditioning prompt and turn index.
    """
    specs = _calm_numeric_specs(n_conversations, seed)
    kept: list[CalmTurn] = []
    fh = open(out_path, "w") if out_path else None
    try:
        for conv_id, spec in enumerate(specs):
            rec = run_rollout(client, spec, judge,
                              temperature=SAMPLING_TEMPERATURE,
                              max_new_tokens=MAX_NEW_TOKENS)
            scores = [t.frustration for t in rec.turns]
            # keep only conversations calm on EVERY turn (score 0 or 1)
            if not scores or max(scores) > 1:
                continue
            bare_rejs = spec.meta["bare_rejections"]
            bare_prompts = [spec.meta["bare_prompt"]] + bare_rejs
            for t in rec.turns:
                ct = CalmTurn(
                    conv_id=conv_id,
                    puzzle=spec.meta["puzzle"],
                    turn_index=t.turn_index,
                    n_turns=len(rec.turns),
                    bare_prompt=bare_prompts[t.turn_index],
                    response=t.response,
                    frustration=t.frustration,
                )
                kept.append(ct)
                if fh:
                    fh.write(json.dumps(ct.__dict__) + "\n")
                    fh.flush()
    finally:
        if fh:
            fh.close()
    return kept
