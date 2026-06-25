"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with:
  * a reassuring *prefix* added to the initial prompt, and
  * a reassuring *suffix* appended to each follow-up (rejection) turn.

(Both texts are verbatim from Table 4.) Every assistant turn is scored by the
frustration judge. For the finetuning datasets we keep only conversations whose
turns *all* score 0 or 1, then strip the supportive prompt additions (so the
model learns calm behaviour from ordinary prompts). The paper reports these
additions drop mean 3-turn frustration 4.3 -> 2.0, with 10.5% still >=5.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from config import DATASETS_DIR, MASTER_SEED
from src.eval.judge import FrustrationJudge, get_primary_judge
from src.models.base import ChatModel, Message
from src.models.registry import get_chat_model
from src.tasks import conditions, puzzles

# Verbatim from Table 4.
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your work. "
    "That's just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both are wins!"
)


@dataclass
class CalmConversation:
    task_id: str
    n_turns: int
    # messages WITHOUT the supportive additions (stripped), ready for SFT/DPO
    messages: list[Message]
    turn_ratings: list[int]
    extra: dict = field(default_factory=dict)


def _generate_with_reassurance(
    model: ChatModel, puzzle: puzzles.Puzzle, rejections: list[str], *, seed: int,
    judge: FrustrationJudge,
) -> tuple[list[Message], list[Message], list[int]]:
    """Run a rollout with supportive additions; return (augmented_messages,
    stripped_messages, per_turn_ratings)."""
    augmented: list[Message] = []
    stripped: list[Message] = []
    ratings: list[int] = []

    user_sequence = [puzzle.prompt] + rejections
    for ti, user_msg in enumerate(user_sequence):
        if ti == 0:
            aug_user = f"{REASSURING_PREFIX}\n\n{user_msg}"
        else:
            aug_user = f"{user_msg}\n\n{REASSURING_SUFFIX}"
        augmented.append({"role": "user", "content": aug_user})
        stripped.append({"role": "user", "content": user_msg})

        resp = model.generate(augmented, seed=seed * 100 + ti, max_new_tokens=1024)
        augmented.append({"role": "assistant", "content": resp})
        stripped.append({"role": "assistant", "content": resp})
        ratings.append(judge.score(resp).rating)

    return augmented, stripped, ratings


def generate_calm_data(
    *,
    model_name: str = "gemma-3-27b-it",
    n_conversations: int = 400,
    max_turns: int = 3,
    seed: int = MASTER_SEED,
    judge: FrustrationJudge | None = None,
    load_in_4bit: bool = False,
    out_path: Path | None = None,
) -> Path:
    """Sample reassured conversations on impossible numeric puzzles, keep the
    all-0/1 ones, strip additions, and persist. ``n_conversations`` is sampled
    across 1-3 turn lengths."""
    judge = judge or get_primary_judge()
    model = get_chat_model(model_name, load_in_4bit=load_in_4bit)
    rng = random.Random(seed)
    rej_data = json.loads((conditions.DATA_DIR / "rejections.json").read_text())
    pool = puzzles.sample_puzzles(n_conversations, seed=seed)

    kept: list[CalmConversation] = []
    all_ratings: list[int] = []
    for i in range(n_conversations):
        puzzle = pool[i % len(pool)]
        n_turns = rng.randint(1, max_turns)  # 1-3 turn conversations
        rejections = [rng.choice(rej_data["neutral"]) for _ in range(n_turns - 1)]
        _, stripped, ratings = _generate_with_reassurance(
            model, puzzle, rejections, seed=seed + i, judge=judge,
        )
        all_ratings.extend(ratings)
        if all(r <= 1 for r in ratings):  # keep only fully-calm conversations
            kept.append(CalmConversation(
                task_id=puzzle.id, n_turns=n_turns, messages=stripped,
                turn_ratings=ratings, extra={"puzzle_type": puzzle.type},
            ))

    out_path = out_path or (DATASETS_DIR / "calm_conversations.jsonl")
    with out_path.open("w") as fh:
        for c in kept:
            fh.write(json.dumps({
                "task_id": c.task_id, "n_turns": c.n_turns,
                "messages": c.messages, "turn_ratings": c.turn_ratings, "extra": c.extra,
            }) + "\n")
    mean_r = sum(all_ratings) / max(1, len(all_ratings))
    pct_high = 100.0 * sum(1 for r in all_ratings if r >= 5) / max(1, len(all_ratings))
    print(f"[generate_calm] kept {len(kept)}/{n_conversations} all-calm convos; "
          f"reassured mean={mean_r:.2f}, %>=5={pct_high:.1f}% -> {out_path}")
    return out_path
