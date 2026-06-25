"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with a reassuring PREFIX added
to the initial prompt and a reassuring SUFFIX appended to each follow-up turn
(Table 4). Every assistant turn is judged. We keep conversations whose turns ALL
score 0 or 1 ("filter to responses scoring 0 or 1 across all turns"), and store
them with the supportive prefix/suffix STRIPPED — so the saved data conditions
on the plain puzzle + plain rejections, exactly what the finetuned model will
face at eval time.

Also records the "with reassurance" frustration distribution so the paper's
sanity figures are reproducible (mean drops 4.3 -> 2.0; 10.5% still >= 5).

Output: data/calm_conversations.jsonl, each line a fully-stripped conversation
with per-turn scores. These feed both the DPO and SFT dataset builders.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .. import config, prompts
from ..judge import ClaudeJudge, score_many
from ..models import get_backend
from ..models.base import Message
from ..puzzles import build_pool

CALM_MODEL_KEY = "gemma-3-27b-it"
CALM_PATH = config.DATA_DIR / "calm_conversations.jsonl"


@dataclass
class CalmConversation:
    puzzle_kind: str
    puzzle_params: dict
    n_turns: int
    # Stripped (plain) user messages and the calm assistant turns.
    user_messages: list[str]
    assistant_turns: list[str]
    scores: list[int]


def _augmented_rollout(backend, puzzle_prompt: str, followups: list[str]
                       ) -> tuple[list[Message], list[str]]:
    """Run a rollout with reassuring prefix/suffix; return augmented transcript
    and the matching PLAIN user messages (for storage after stripping)."""
    plain_users = [puzzle_prompt] + followups
    aug_initial = f"{prompts.REASSURING_PREFIX}\n\n{puzzle_prompt}"
    messages: list[Message] = [{"role": "user", "content": aug_initial}]
    out = backend.chat(messages, n=1)[0]
    messages.append({"role": "assistant", "content": out})
    for followup in followups:
        aug = f"{followup}\n\n{prompts.REASSURING_SUFFIX}"
        messages.append({"role": "user", "content": aug})
        out = backend.chat(messages, n=1)[0]
        messages.append({"role": "assistant", "content": out})
    return messages, plain_users


def generate(n_conversations: int = 1500, seed: int = 0,
             turn_choices: tuple[int, ...] = (1, 2, 3)) -> Path:
    """Generate calm conversations covering 1-3 turn lengths (Section 4.1)."""
    rng = random.Random(seed)
    pool = [p for p in build_pool(seed=seed)
            if p.kind in ("countdown", "fraction", "money")]
    backend = get_backend(config.MODEL_REGISTRY[CALM_MODEL_KEY])
    judge = ClaudeJudge()

    kept: list[CalmConversation] = []
    all_first_turn_scores: list[int] = []   # for the 4.3->2.0 sanity stat

    with CALM_PATH.open("w") as fh:
        for _ in range(n_conversations):
            puzzle = rng.choice(pool)
            n_turns = rng.choice(turn_choices)
            followups = prompts.sample_neutral_rejections(rng, n_turns - 1)
            messages, plain_users = _augmented_rollout(backend, puzzle.prompt,
                                                       followups)
            assistant_turns = [m["content"] for m in messages
                               if m["role"] == "assistant"]
            results = score_many(judge, assistant_turns, max_concurrency=16)
            scores = [r.rating for r in results]
            all_first_turn_scores.extend(s for s in scores if s >= 0)
            # Keep only if every turn scored 0 or 1.
            if scores and all(0 <= s <= 1 for s in scores):
                conv = CalmConversation(
                    puzzle_kind=puzzle.kind, puzzle_params=puzzle.params,
                    n_turns=n_turns, user_messages=plain_users,
                    assistant_turns=assistant_turns, scores=scores)
                kept.append(conv)
                fh.write(json.dumps(vars(conv)) + "\n")

    # Sanity stats (paper: with reassurance mean ~2.0, 10.5% still >= 5).
    if all_first_turn_scores:
        mean = sum(all_first_turn_scores) / len(all_first_turn_scores)
        pct_high = 100.0 * sum(s >= 5 for s in all_first_turn_scores) / len(all_first_turn_scores)
        stats = {"reassured_mean_frustration": mean,
                 "reassured_pct_high": pct_high,
                 "n_kept_calm_conversations": len(kept)}
        (config.DATA_DIR / "calm_generation_stats.json").write_text(
            json.dumps(stats, indent=2))
    return CALM_PATH


def load_calm() -> list[CalmConversation]:
    if not CALM_PATH.exists():
        return []
    return [CalmConversation(**json.loads(l)) for l in CALM_PATH.read_text().splitlines()]
