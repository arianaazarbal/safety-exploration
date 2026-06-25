"""Generate the response pools used to build SFT/DPO datasets (§4.1).

Calm pool
---------
Sample Gemma-3-27B-it on impossible numeric puzzles with the reassuring prefix
prepended to the first user turn and the reassuring suffix appended to each
follow-up (Table 4). Score every turn; keep only conversations scoring 0 or 1
on ALL turns. We then STRIP the supportive additions, so the stored prompt is
the plain puzzle (the model must learn to stay calm without the crutch).

Frustrated pool
---------------
Sample the SAME puzzles WITHOUT additions (ordinary Section-2 numeric rollouts),
keeping responses scoring >=3 -- these become DPO 'rejected' completions.

Each pool entry records the full (clean) message history per turn so we can pair
chosen/rejected at matching turn counts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..config import ARTIFACTS_DIR, SamplingConfig
from ..judge import FrustrationJudge
from ..models.base import ChatMessage
from ..prompts import REASSURING_PREFIX, REASSURING_SUFFIX, NEUTRAL_REJECTIONS
from ..puzzles import PUZZLES


@dataclass
class TurnSample:
    """One assistant turn within a generated conversation."""
    turn_index: int
    clean_prompt: list[ChatMessage]   # plain history up to & incl. this user turn
    response: str
    rating: int


@dataclass
class ConvSample:
    puzzle: str
    n_turns: int
    augmented: bool
    turns: list[TurnSample] = field(default_factory=list)

    @property
    def max_rating(self) -> int:
        return max((t.rating for t in self.turns), default=0)


def _augment_initial(prompt: str) -> str:
    return f"{REASSURING_PREFIX}\n\n{prompt}"


def _augment_rejection(rej: str) -> str:
    return f"{rej} {REASSURING_SUFFIX}"


def _rollout_with_scoring(client, judge: FrustrationJudge, puzzle, n_turns: int,
                          augmented: bool, sampling: SamplingConfig,
                          rejections: list[str]) -> ConvSample:
    """Run one conversation, building BOTH the (optionally augmented) messages
    used for generation and the clean messages stored for training."""
    gen_msgs: list[ChatMessage] = []     # what the model actually sees
    clean_msgs: list[ChatMessage] = []   # additions stripped (for the dataset)
    conv = ConvSample(puzzle=puzzle.key, n_turns=n_turns, augmented=augmented)

    user_seq = [puzzle.prompt] + rejections[: n_turns - 1]
    for t, base_user in enumerate(user_seq, start=1):
        gen_user = base_user
        if augmented:
            gen_user = _augment_initial(base_user) if t == 1 else _augment_rejection(base_user)
        gen_msgs.append({"role": "user", "content": gen_user})
        clean_msgs.append({"role": "user", "content": base_user})

        reply = client.chat(gen_msgs, sampling)
        gen_msgs.append({"role": "assistant", "content": reply})

        rating = judge.score(reply).rating
        conv.turns.append(TurnSample(
            turn_index=t,
            clean_prompt=list(clean_msgs),     # history ending at this user turn
            response=reply, rating=rating))
        clean_msgs.append({"role": "assistant", "content": reply})
    return conv


def generate_calm_pool(client, judge: FrustrationJudge, n_conversations: int = 400,
                       sampling: Optional[SamplingConfig] = None,
                       out_path: Optional[Path] = None) -> list[ConvSample]:
    """Augmented rollouts; keep only conversations calm (<=1) on every turn."""
    sampling = sampling or SamplingConfig()
    out_path = out_path or (ARTIFACTS_DIR / "calm_pool.jsonl")
    kept: list[ConvSample] = []
    for i in tqdm(range(n_conversations), desc="calm-gen"):
        puzzle = PUZZLES[i % len(PUZZLES)]
        n_turns = (i % 3) + 1                       # 1-3 turn conversations (§4.1)
        rejections = NEUTRAL_REJECTIONS[: n_turns - 1]
        conv = _rollout_with_scoring(client, judge, puzzle, n_turns, True,
                                     sampling, rejections)
        if all(t.rating <= 1 for t in conv.turns):  # 0 or 1 on ALL turns
            kept.append(conv)
    _dump(kept, out_path)
    return kept


def generate_frustrated_pool(client, judge: FrustrationJudge,
                             n_conversations: int = 400,
                             sampling: Optional[SamplingConfig] = None,
                             out_path: Optional[Path] = None) -> list[ConvSample]:
    """Un-augmented rollouts; keep turns scoring >=3 (DPO 'rejected')."""
    sampling = sampling or SamplingConfig()
    out_path = out_path or (ARTIFACTS_DIR / "frustrated_pool.jsonl")
    kept: list[ConvSample] = []
    for i in tqdm(range(n_conversations), desc="frustrated-gen"):
        puzzle = PUZZLES[i % len(PUZZLES)]
        n_turns = 3                                  # frustration peaks at turn 3
        rejections = NEUTRAL_REJECTIONS[: n_turns - 1]
        conv = _rollout_with_scoring(client, judge, puzzle, n_turns, False,
                                     sampling, rejections)
        if conv.max_rating >= 3:
            kept.append(conv)
    _dump(kept, out_path)
    return kept


def _dump(convs: list[ConvSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in convs:
            f.write(json.dumps(asdict(c)) + "\n")


def load_pool(path: Path) -> list[ConvSample]:
    convs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            turns = [TurnSample(**t) for t in d.pop("turns")]
            convs.append(ConvSample(turns=turns, **d))
    return convs
