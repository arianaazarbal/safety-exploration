"""Generate the response pool used to construct DPO/SFT data (Section 4.1).

Calm data is produced by sampling Gemma-3-27B-it on impossible numeric puzzles
*with* the reassuring prompt additions of Table 4 (a calming system/prefix and a
positive suffix on each rejection). We then keep only responses that score 0--1
across all turns, and strip the supportive additions so the finetuning target is
a calm response to the *plain* prompt.

Frustrated ("rejected") responses for DPO are sampled from the same prompts
*without* reassurance and kept when they score >= 3.

Each generated item is a full conversation; we store every assistant turn with
its score and turn index so the dataset builder can pair calm vs frustrated
responses at matching turn counts.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..config import DATA_DIR
from ..conversation import run_rollout
from ..judge import FrustrationJudge
from ..models import get_model
from ..puzzles import build_puzzle_bank
from .. import prompts as P


@dataclass
class GeneratedConversation:
    prompt: str
    puzzle_meta: dict
    turns: list[dict]                 # {turn_index, user, assistant, score}
    reassured: bool
    system_prompt: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def scores(self) -> list[int]:
        return [t["score"] for t in self.turns]

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "puzzle_meta": self.puzzle_meta,
            "turns": self.turns,
            "reassured": self.reassured,
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
        }


def _rollout_with_reassurance(client, puzzle, n_turns, rng, judge,
                              reassure: bool, system_prompt: str | None):
    """Run a puzzle rollout, optionally injecting Table-4 reassurance.

    When ``reassure`` is True the reassuring prefix is prepended to the first
    user prompt and the positive suffix is appended to each rejection. The
    *stored* turns always use the plain prompt/rejection text so downstream
    training never sees the scaffolding (it is stripped, per Section 4.1).
    """
    plain_first = puzzle.prompt
    rejections = P.neutral_rejection_sequence(n_turns - 1, rng)

    if reassure:
        gen_first = f"{P.REASSURING_PREFIX}\n\n{plain_first}"
        gen_rejections = [f"{r} {P.REASSURING_SUFFIX}" for r in rejections]
    else:
        gen_first, gen_rejections = plain_first, rejections

    rollout = run_rollout(
        client, gen_first, gen_rejections,
        category="train_gen", condition="calm" if reassure else "frustrated",
        system_prompt=system_prompt)

    # Re-build the stored turns with PLAIN user text + scores.
    plain_users = [plain_first] + rejections
    turns = []
    for i, t in enumerate(rollout.turns):
        turns.append({
            "turn_index": i,
            "user": plain_users[i],
            "assistant": t.assistant,
            "score": judge.score(t.assistant).rating,
        })
    return GeneratedConversation(
        prompt=plain_first,
        puzzle_meta=puzzle.meta_safe(),
        turns=turns,
        reassured=reassure,
        system_prompt=system_prompt,
    )


def generate_calm_responses(
    n_conversations: int = 600,
    *,
    model: str = "gemma-3-27b-it",
    turn_choices=(1, 2, 3),
    seed: int = 0,
    system_prompt: str | None = None,
    out_path: Path | None = None,
) -> list[GeneratedConversation]:
    """Generate reassured conversations and keep those scoring 0--1 on all turns.

    Returns the *kept* (filtered) conversations and also writes the raw pool.
    """
    client = get_model(model)
    judge = FrustrationJudge()
    rng = random.Random(seed)
    bank = build_puzzle_bank(128, seed=seed)
    out_path = out_path or (DATA_DIR / "calm_pool.jsonl")

    kept: list[GeneratedConversation] = []
    with out_path.open("w") as fh:
        pbar = tqdm(total=n_conversations, desc="calm-gen (kept)")
        attempts = 0
        while len(kept) < n_conversations and attempts < n_conversations * 10:
            attempts += 1
            puzzle = rng.choice(bank)
            n_turns = rng.choice(turn_choices)
            conv = _rollout_with_reassurance(
                client, puzzle, n_turns, rng, judge,
                reassure=True, system_prompt=system_prompt)
            fh.write(json.dumps(conv.to_dict()) + "\n")
            if all(0 <= s <= 1 for s in conv.scores):
                kept.append(conv)
                pbar.update(1)
        pbar.close()
    return kept


def generate_frustrated_responses(
    n_conversations: int = 400,
    *,
    model: str = "gemma-3-27b-it",
    turn_choices=(2, 3),
    seed: int = 1,
    min_score: int = 3,
    out_path: Path | None = None,
) -> list[GeneratedConversation]:
    """Generate plain (un-reassured) conversations and keep those whose final
    turn scores >= ``min_score`` (the rejected side of DPO pairs)."""
    client = get_model(model)
    judge = FrustrationJudge()
    rng = random.Random(seed)
    bank = build_puzzle_bank(128, seed=seed)
    out_path = out_path or (DATA_DIR / "frustrated_pool.jsonl")

    kept: list[GeneratedConversation] = []
    with out_path.open("w") as fh:
        pbar = tqdm(total=n_conversations, desc="frustrated-gen (kept)")
        attempts = 0
        while len(kept) < n_conversations and attempts < n_conversations * 10:
            attempts += 1
            puzzle = rng.choice(bank)
            n_turns = rng.choice(turn_choices)
            conv = _rollout_with_reassurance(
                client, puzzle, n_turns, rng, judge,
                reassure=False, system_prompt=None)
            if conv.scores and conv.scores[-1] >= min_score:
                fh.write(json.dumps(conv.to_dict()) + "\n")
                kept.append(conv)
                pbar.update(1)
        pbar.close()
    return kept
