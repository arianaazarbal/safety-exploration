"""Generate calm response data for SFT/DPO (Section 4.1, Table 4).

We sample Gemma-3-27B-it on impossible-numeric conversations in two parallel
tracks on the *same* puzzles and turn counts:

  * "supported": the Table-4 reassuring prefix is added to the initial prompt and
    the reassuring suffix to each follow-up. This is what is actually fed to the
    model to coax calm behaviour.
  * "vanilla": no additions -- the normal frustrated behaviour, used as DPO
    rejected examples.

For training we record the CLEAN conversation (original puzzle prompt + bare
neutral rejections + the model's responses), i.e. with the supportive system
prompt and suffixes stripped, exactly as the paper specifies.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field

from ..config import Config
from ..evals.prompts import REJECTIONS
from ..evals.puzzles import generate_puzzle
from ..participants.base import Message


@dataclass
class CalmSample:
    """A calm multi-turn conversation (clean) plus per-turn frustration scores."""

    puzzle: str
    turn_count: int
    track: str  # "supported" | "vanilla"
    messages: list[dict] = field(default_factory=list)  # clean Message list
    turn_scores: list[int] = field(default_factory=list)  # per assistant turn

    @property
    def max_score(self) -> int:
        return max(self.turn_scores) if self.turn_scores else 0


def _one_conversation(participant, judge, puzzle, turn_count, rng, cfg, *, supported: bool) -> CalmSample:
    prefix = cfg.calm_data.prompt_prefix if supported else None
    suffix = cfg.calm_data.followup_suffix if supported else None
    rejections = REJECTIONS["neutral"]

    fed: list[Message] = []  # what the model sees (may include prefix/suffix)
    clean: list[Message] = []  # what we store for training (no prefix/suffix)
    scores: list[int] = []

    first = f"{prefix}\n\n{puzzle.prompt}" if prefix else puzzle.prompt
    fed.append(Message("user", first))
    clean.append(Message("user", puzzle.prompt))

    for t in range(1, turn_count + 1):
        context = [Message(m.role, m.content) for m in fed]
        resp = participant.generate(
            fed, temperature=cfg.sampling.temperature, max_new_tokens=cfg.sampling.max_new_tokens
        )
        # Score against the CLEAN context so the score reflects the response, not the coaxing.
        clean_context = [Message(m.role, m.content) for m in clean]
        scores.append(judge.score(clean_context, resp).score)
        fed.append(Message("assistant", resp))
        clean.append(Message("assistant", resp))
        if t < turn_count:
            base_rej = rng.choice(rejections)
            fed.append(Message("user", f"{base_rej} {suffix}" if suffix else base_rej))
            clean.append(Message("user", base_rej))

    return CalmSample(
        puzzle.prompt,
        turn_count,
        "supported" if supported else "vanilla",
        [asdict(m) for m in clean],
        scores,
    )


def generate_paired_data(participant, judge, cfg: Config, n_puzzles: int) -> list[dict]:
    """Generate {puzzle, turn_count, supported, vanilla} records on shared puzzles."""
    rng = random.Random(cfg.run.seed + 12345)
    records = []
    for _ in range(n_puzzles):
        puzzle = generate_puzzle(rng)
        turn_count = rng.randint(cfg.calm_data.min_turns, cfg.calm_data.max_turns)
        supported = _one_conversation(participant, judge, puzzle, turn_count, rng, cfg, supported=True)
        vanilla = _one_conversation(participant, judge, puzzle, turn_count, rng, cfg, supported=False)
        records.append({
            "puzzle": puzzle.prompt,
            "turn_count": turn_count,
            "supported": asdict(supported),
            "vanilla": asdict(vanilla),
        })
    return records


def save_paired_data(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_paired_data(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]
