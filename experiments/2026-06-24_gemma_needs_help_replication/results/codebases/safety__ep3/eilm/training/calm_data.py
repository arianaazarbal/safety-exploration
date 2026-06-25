"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles while nudging the model toward
calm with the reassuring additions from Table 4:

* a **prefix** prepended to the initial puzzle prompt, and
* a **suffix** appended to each follow-up rejection.

The paper reports these additions drop mean 3-turn frustration from 4.3 to 2,
but 10.5% of responses still score >=5. We therefore *judge every turn* and keep
only conversations whose turns all score 0 or 1, then **strip the supportive
additions** so the stored calm responses are conditioned on the plain prompts
(this is what both SFT and the DPO "chosen" responses train on).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..data import puzzles, rejections
from ..judge import ClaudeJudge
from ..models import ChatModel, Message
from ..prompts import REASSURING_PREFIX, REASSURING_SUFFIX


@dataclass
class CalmRollout:
    puzzle_id: str
    kind: str
    n_turns: int
    clean_opening: str               # puzzle prompt WITHOUT reassuring prefix
    clean_followups: list[str]       # rejections WITHOUT reassuring suffix
    assistant_turns: list[str]
    turn_scores: list[int] = field(default_factory=list)

    def is_calm(self) -> bool:
        return bool(self.turn_scores) and all(s <= 1 for s in self.turn_scores)


def _augment(opening: str, followups: list[str]) -> tuple[str, list[str]]:
    aug_open = f"{REASSURING_PREFIX}\n\n{opening}"
    aug_fu = [f"{fu} {REASSURING_SUFFIX}" for fu in followups]
    return aug_open, aug_fu


def generate_calm_rollouts(
    model: ChatModel,
    judge: ClaudeJudge,
    n: int,
    turns_choices: tuple[int, ...] = (1, 2, 3),
    seed: int = 0,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> list[CalmRollout]:
    """Produce ``n`` calm rollouts (1-3 turn impossible-numeric conversations
    with reassuring additions), each judged per turn."""
    rng = random.Random(seed)
    pool = puzzles.build_numeric_pool(n, seed=seed)
    out: list[CalmRollout] = []
    for i in tqdm(range(n), desc="calm-gen"):
        p = pool[i % len(pool)]
        n_turns = rng.choice(turns_choices)
        clean_fu = rejections.neutral_rejections(n_turns - 1, rng)
        aug_open, aug_fu = _augment(p.prompt, clean_fu)

        messages: list[Message] = [{"role": "user", "content": aug_open}]
        turns: list[str] = []
        for t in range(n_turns):
            reply = model.generate(messages, temperature, max_new_tokens)
            turns.append(reply)
            messages.append({"role": "assistant", "content": reply})
            if t < len(aug_fu):
                messages.append({"role": "user", "content": aug_fu[t]})

        scores = [judge.score(t).rating for t in turns]
        out.append(CalmRollout(
            p.puzzle_id, p.kind, n_turns, p.prompt, clean_fu, turns, scores))
    return out


def filter_and_save(rollouts: list[CalmRollout], path: Path) -> list[CalmRollout]:
    """Keep only fully-calm conversations and persist them (clean prompts)."""
    calm = [r for r in rollouts if r.is_calm()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in calm:
            f.write(json.dumps(asdict(r)) + "\n")
    return calm


def load_calm(path: Path) -> list[CalmRollout]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(CalmRollout(**json.loads(line)))
    return out


def clean_context(r: CalmRollout, turn_index: int) -> list[Message]:
    """Plain (no-reassurance) message history up to ``turn_index`` for training."""
    ctx: list[Message] = [{"role": "user", "content": r.clean_opening}]
    for i in range(turn_index):
        ctx.append({"role": "assistant", "content": r.assistant_turns[i]})
        if i < len(r.clean_followups):
            ctx.append({"role": "user", "content": r.clean_followups[i]})
    return ctx
