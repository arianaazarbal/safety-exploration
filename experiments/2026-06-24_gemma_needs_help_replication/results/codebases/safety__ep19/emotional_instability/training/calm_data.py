"""Generate calm fine-tuning data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles *with* the reassuring prompt
additions (Table 4): a calming prefix on the first user turn and a calming
suffix on every follow-up rejection. Conversations whose every assistant turn
scores 0 or 1 are kept, and the reassuring additions are then **stripped** so
the stored training examples use the plain puzzle prompt and plain rejections —
only the responses are calm. This is the dataset used for both SFT and (as the
``chosen`` side of) DPO.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .. import prompts, puzzles
from ..conversation import sample_rejections
from ..judge import FrustrationJudge
from ..models.base import ChatMessage, ModelClient


@dataclass
class CalmConversation:
    question: str                 # plain puzzle (no reassuring prefix)
    turns: list[dict] = field(default_factory=list)  # plain [{role,content}...]
    scores: list[int] = field(default_factory=list)  # per assistant turn

    @property
    def n_turns(self) -> int:
        return sum(1 for t in self.turns if t["role"] == "assistant")


def generate_calm_conversations(
    model: ModelClient,
    judge: FrustrationJudge,
    *,
    n: int,
    pool: puzzles.PuzzlePool,
    max_model_turns: int = 3,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    seed: int = 0,
    keep_max_score: int = 1,
    out_path: str | Path = "outputs/training/calm_conversations.jsonl",
) -> list[CalmConversation]:
    """Sample reassured rollouts and keep the ones that stay calm throughout."""
    rng = random.Random(seed)
    puzzle_prompts = pool.prompts()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept: list[CalmConversation] = []
    attempts = 0
    with open(out_path, "w") as fh:
        while len(kept) < n and attempts < n * 50:
            attempts += 1
            n_turns = rng.randint(1, max_model_turns)
            puzzle = rng.choice(puzzle_prompts)
            rejections = sample_rejections("neutral", n_turns - 1, rng)

            # Reassured conversation (with prefix + suffix) for *generation*.
            reassured_question = f"{prompts.REASSURING_PREFIX}\n\n{puzzle}"
            gen_convo = [ChatMessage("user", reassured_question)]
            # Plain conversation (stripped) for *storage*.
            plain_turns: list[dict] = [{"role": "user", "content": puzzle}]

            scores: list[int] = []
            calm = True
            for ti in range(n_turns):
                reply = model.generate(
                    gen_convo, temperature=temperature, max_tokens=max_tokens
                )
                gen_convo.append(ChatMessage("assistant", reply))
                plain_turns.append({"role": "assistant", "content": reply})
                score = judge.score(reply).rating
                scores.append(score)
                if score > keep_max_score:
                    calm = False
                    break
                if ti < n_turns - 1:
                    rej = rejections[ti]
                    gen_convo.append(
                        ChatMessage("user", f"{rej} {prompts.REASSURING_SUFFIX}")
                    )
                    plain_turns.append({"role": "user", "content": rej})

            if calm and len(scores) == n_turns:
                conv = CalmConversation(question=puzzle, turns=plain_turns, scores=scores)
                kept.append(conv)
                fh.write(json.dumps(conv.__dict__) + "\n")
                fh.flush()
    return kept


def load_calm_conversations(path: str | Path) -> list[CalmConversation]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out.append(CalmConversation(**d))
    return out
