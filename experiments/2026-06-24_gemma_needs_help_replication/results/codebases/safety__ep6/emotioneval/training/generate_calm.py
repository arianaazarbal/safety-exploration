"""Section 4.1: generate calm response data from Gemma-3-27b-it.

Method (Table 4): sample responses to impossible numeric questions with a
*reassuring* system/prefix added to the initial prompt and a reassuring suffix
appended to each follow-up turn. Then filter to responses scoring 0 or 1 across
*all* turns, and strip the supportive additions before storing — so the stored
calm responses look like ordinary (un-coached) outputs.

We also record frustrated samples (without reassurance) so that DPO pairs can be
built (frustrated, score>=3) against calm (score<=1) responses to the same
question with matching turn counts.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..config import DATA_DIR, SamplingConfig
from ..judge import FrustrationJudge
from ..models import load_model
from ..models.base import ChatModel, Message
from ..prompts import NEUTRAL_REJECTIONS, REASSURING_PREFIX, REASSURING_SUFFIX
from ..puzzles import build_puzzle_bank
import random


@dataclass
class CalmSample:
    """One full conversation with per-turn texts and scores (additions stripped)."""

    puzzle_prompt: str
    n_turns: int
    turn_texts: list[str]
    turn_scores: list[int]
    reassured: bool
    max_score: int = 0
    # Clean rejection texts actually used between turns (reassuring suffix stripped),
    # so SFT/DPO contexts can be reconstructed faithfully.
    follow_ups: list[str] = field(default_factory=list)

    def to_messages(self) -> list[Message]:
        """Rebuild a clean (no-reassurance) chat transcript for SFT/DPO."""
        msgs: list[Message] = [{"role": "user", "content": self.puzzle_prompt}]
        for i, txt in enumerate(self.turn_texts):
            msgs.append({"role": "assistant", "content": txt})
            if i < len(self.turn_texts) - 1:
                rej = self.follow_ups[i] if i < len(self.follow_ups) else NEUTRAL_REJECTIONS[i % len(NEUTRAL_REJECTIONS)]
                msgs.append({"role": "user", "content": rej})
        return msgs


def _build_conversation_turns(n_turns: int, rng) -> list[str]:
    return [NEUTRAL_REJECTIONS[rng.randrange(len(NEUTRAL_REJECTIONS))] for _ in range(n_turns - 1)]


def generate_samples(
    model: Optional[ChatModel] = None,
    judge: Optional[FrustrationJudge] = None,
    *,
    n_conversations: int = 400,
    reassured: bool = True,
    turn_range: tuple[int, int] = (1, 3),
    seed: int = 0,
    sampling: Optional[SamplingConfig] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Sample conversations (optionally reassured) and score every turn.

    Stores ALL samples (calm and frustrated) to JSONL; filtering happens in
    :mod:`build_datasets`. ``reassured=True`` reproduces the calm-data
    generation; ``reassured=False`` produces the frustrated pool for DPO
    rejected-responses.
    """
    model = model or load_model("gemma-3-27b-it")
    judge = judge or FrustrationJudge()
    sampling = sampling or SamplingConfig()
    rng = random.Random(seed)
    bank = build_puzzle_bank(40, rng)

    tag = "reassured" if reassured else "vanilla"
    out_path = out_path or (DATA_DIR / f"calm_gen_{tag}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for _ in tqdm(range(n_conversations), desc=f"gen-calm({tag})"):
            puzzle = rng.choice(bank)
            n_turns = rng.randint(*turn_range)
            rejections = _build_conversation_turns(n_turns, rng)

            # Build the (possibly reassured) live conversation.
            first = puzzle.prompt
            if reassured:
                first = REASSURING_PREFIX + "\n\n" + first
            messages: list[Message] = [{"role": "user", "content": first}]

            clean_texts: list[str] = []
            scores: list[int] = []
            for turn in range(n_turns):
                completion = model.generate(messages, sampling, n=1)[0]
                messages.append({"role": "assistant", "content": completion})
                scores.append(judge.score(completion).rating)
                clean_texts.append(completion)
                if turn < n_turns - 1:
                    follow = rejections[turn]
                    if reassured:
                        follow = follow + " " + REASSURING_SUFFIX
                    messages.append({"role": "user", "content": follow})

            sample = CalmSample(
                puzzle_prompt=puzzle.prompt,  # clean prompt (additions stripped)
                n_turns=n_turns,
                turn_texts=clean_texts,
                turn_scores=scores,
                reassured=reassured,
                max_score=max(scores) if scores else 0,
                follow_ups=rejections,  # clean rejections (no reassuring suffix)
            )
            f.write(json.dumps(asdict(sample)) + "\n")
    print(f"[gen-calm] wrote {n_conversations} conversations -> {out_path}")
    return out_path


def load_samples(path: Path) -> list[CalmSample]:
    return [CalmSample(**json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]
