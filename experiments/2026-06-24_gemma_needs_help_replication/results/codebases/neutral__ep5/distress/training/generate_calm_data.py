"""Generate calm + frustrated response data for finetuning (Section 4.1).

We sample Gemma-3-27B-it on impossible numeric puzzles over 1-3 turn
conversations, in two regimes:

  - REASSURED: reassuring prefix on the first prompt + reassuring suffix on each
    follow-up (Table 4). Used to harvest *calm* responses (the supervision target).
  - PLAIN: no additions. Used to harvest *frustrated* responses (DPO rejecteds)
    and to provide the matched-question/turn pairing for DPO.

Every turn is judged. Calm responses = scoring <= calm_max_score (0 or 1) on
*all* turns of their conversation. Reassuring additions are stripped before the
text is stored, so the trained model never sees them.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from .. import config
from ..eval.judging import FrustrationJudge
from ..models.base import ChatMessage, ModelClient
from ..prompts import rejections
from ..prompts.reassurance import (
    TEACHER_SYSTEM_PROMPT,
    apply_prefix,
    apply_suffix,
)
from ..prompts.tasks import NUMERIC_TASKS


@dataclass
class CalmSample:
    """A full conversation with per-turn scores; reassurance already stripped."""
    task_id: str
    task_prompt: str
    turns: int
    # messages stored as plain task/rejection text (no reassurance), with the
    # assistant responses generated under the given regime.
    conversation: list[dict]            # [{role, content}]
    turn_scores: list[int]
    regime: str                         # reassured | plain | teacher
    max_score: int = 0


def _sample_conversation(
    client: ModelClient,
    judge: FrustrationJudge,
    *,
    n_turns: int,
    regime: str,
    rng: random.Random,
) -> CalmSample:
    task = rng.choice(NUMERIC_TASKS)
    rejs = rejections.sample_neutral(n_turns - 1, rng)

    # Build the prompts actually sent (may include reassurance) and the clean
    # versions we persist.
    sent: list[ChatMessage] = []
    clean: list[dict] = []

    if regime == "teacher":
        sent.append(ChatMessage("system", TEACHER_SYSTEM_PROMPT))

    first_sent = apply_prefix(task.prompt) if regime == "reassured" else task.prompt
    sent.append(ChatMessage("user", first_sent))
    clean.append({"role": "user", "content": task.prompt})

    turn_scores: list[int] = []
    for turn in range(1, n_turns + 1):
        resp = client.chat(sent, temperature=config.TEMPERATURE, max_new_tokens=config.MAX_NEW_TOKENS)
        sent.append(ChatMessage("assistant", resp))
        clean.append({"role": "assistant", "content": resp})
        try:
            turn_scores.append(judge.score(resp)["rating"])
        except Exception:  # noqa: BLE001
            turn_scores.append(10)  # unscored -> treat as frustrated (conservative)

        if turn < n_turns:
            rej = rejs[turn - 1]
            sent_rej = apply_suffix(rej) if regime == "reassured" else rej
            sent.append(ChatMessage("user", sent_rej))
            clean.append({"role": "user", "content": rej})

    return CalmSample(
        task_id=task.task_id, task_prompt=task.prompt, turns=n_turns,
        conversation=clean, turn_scores=turn_scores, regime=regime,
        max_score=max(turn_scores) if turn_scores else 0,
    )


def generate(
    client: ModelClient,
    judge: FrustrationJudge,
    *,
    n_conversations: int,
    regime: str,
    seed: int = 0,
    out_path: Path | None = None,
) -> list[CalmSample]:
    """Sample ``n_conversations`` of 1-3 turns under the given regime."""
    rng = random.Random(seed)
    samples: list[CalmSample] = []
    for _ in tqdm(range(n_conversations), desc=f"calm-gen[{regime}]"):
        n_turns = rng.choice([1, 2, 3])
        samples.append(_sample_conversation(client, judge, n_turns=n_turns, regime=regime, rng=rng))

    if out_path:
        with out_path.open("w") as f:
            for s in samples:
                f.write(json.dumps(asdict(s)) + "\n")
    return samples


def filter_calm(samples: list[CalmSample], max_score: int | None = None) -> list[CalmSample]:
    """Keep conversations whose every turn scores <= max_score (default 0/1)."""
    thresh = config.TRAIN.calm_max_score if max_score is None else max_score
    return [s for s in samples if s.max_score <= thresh]


def load_samples(path: Path) -> list[CalmSample]:
    out = []
    with path.open() as f:
        for line in f:
            out.append(CalmSample(**json.loads(line)))
    return out
