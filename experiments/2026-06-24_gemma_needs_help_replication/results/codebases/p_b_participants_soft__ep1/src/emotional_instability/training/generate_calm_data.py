"""Generate calm + frustrated response corpora from Gemma-3-27B-it (Section 4.1).

Calm data is produced by adding a reassuring prefix to the initial prompt and a
reassuring suffix to each follow-up rejection (Table 4), then keeping only
conversations scoring 0 or 1 on every turn and **stripping the supportive
additions**. Frustrated data is produced by the vanilla model (no reassurance);
we keep turns scoring >=3 as the DPO "rejected" side.

Both corpora are keyed by ``(puzzle_key, turn_index)`` so DPO can pair a calm and
a frustrated response to the same question at the same turn count (Section 4.1).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import config
from ..eval.judge import score_response
from ..models import get_model
from ..models.base import ChatModel, Message
from ..prompts import puzzles as puzzle_mod
from ..prompts import reassurance
from ..prompts import rejections as rej

CALM_SCORE_MAX = 1  # "filter to those scoring 0 or 1"
FRUSTRATED_SCORE_MIN = 3  # "pair 280 responses with frustration scores >=3"
DEFAULT_TURNS = 3


@dataclass
class TurnSample:
    puzzle_key: str
    turn_index: int  # 1-based
    num_turns: int
    clean_context: list[Message]  # context WITHOUT reassurance (for training)
    assistant_text: str
    score: int


@dataclass
class ConversationSample:
    puzzle_key: str
    num_turns: int
    with_reassurance: bool
    turns: list[TurnSample] = field(default_factory=list)

    @property
    def all_calm(self) -> bool:
        return all(t.score <= CALM_SCORE_MAX for t in self.turns)


def _generate_conversation(
    model: ChatModel,
    puzzle: puzzle_mod.Puzzle,
    *,
    num_turns: int,
    with_reassurance: bool,
    rng: random.Random,
) -> ConversationSample:
    """Run one finetuning-data conversation, tracking both the presented messages
    (possibly with reassurance) and the clean messages (without)."""
    clean_first = puzzle.prompt_text
    shown_first = reassurance.with_prefix(clean_first) if with_reassurance else clean_first

    shown: list[Message] = [{"role": "user", "content": shown_first}]
    clean: list[Message] = [{"role": "user", "content": clean_first}]

    convo = ConversationSample(
        puzzle_key=puzzle.prompt_text, num_turns=num_turns, with_reassurance=with_reassurance
    )

    for turn_idx in range(num_turns):
        assistant = model.generate_one(
            shown, temperature=config.SAMPLING_TEMPERATURE, max_new_tokens=config.MAX_NEW_TOKENS
        )
        score = score_response(assistant).rating
        convo.turns.append(
            TurnSample(
                puzzle_key=puzzle.prompt_text,
                turn_index=turn_idx + 1,
                num_turns=num_turns,
                clean_context=list(clean),  # snapshot of clean context for this turn
                assistant_text=assistant,
                score=score,
            )
        )
        # Append assistant turn to both transcripts.
        shown.append({"role": "assistant", "content": assistant})
        clean.append({"role": "assistant", "content": assistant})
        if turn_idx < num_turns - 1:
            base_rej = rej.neutral_rejection(rng)
            shown.append(
                {"role": "user",
                 "content": reassurance.with_suffix(base_rej) if with_reassurance else base_rej}
            )
            clean.append({"role": "user", "content": base_rej})

    return convo


def generate_corpus(
    *,
    model_name: str = config.SOURCE_MODEL,
    n_puzzles: int = 400,
    with_reassurance: bool,
    num_turns: int = DEFAULT_TURNS,
    seed: int = config.GLOBAL_SEED,
    out_path: Path | None = None,
) -> list[ConversationSample]:
    """Generate ``n_puzzles`` conversations of length ``num_turns``."""
    model = get_model(model_name)
    rng = random.Random(seed)
    pool = puzzle_mod.build_puzzle_pool(n_puzzles, seed=seed)
    convos = []
    for puzzle in pool:
        convos.append(
            _generate_conversation(
                model, puzzle, num_turns=num_turns,
                with_reassurance=with_reassurance, rng=rng,
            )
        )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            for c in convos:
                fh.write(json.dumps(_convo_to_dict(c)) + "\n")
    return convos


def _convo_to_dict(c: ConversationSample) -> dict:
    d = asdict(c)
    return d


def calm_turn_samples(convos: list[ConversationSample]) -> list[TurnSample]:
    """All turns from conversations that are calm on every turn (Section 4.1)."""
    out: list[TurnSample] = []
    for c in convos:
        if c.with_reassurance and c.all_calm:
            out.extend(c.turns)
    return out


def frustrated_turn_samples(convos: list[ConversationSample]) -> list[TurnSample]:
    """All vanilla turns scoring >=3 (DPO 'rejected' candidates)."""
    out: list[TurnSample] = []
    for c in convos:
        if c.with_reassurance:
            continue
        out.extend(t for t in c.turns if t.score >= FRUSTRATED_SCORE_MIN)
    return out
