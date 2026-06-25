"""Fine-tuning data generation (Section 4.1).

Pipeline:
  1. Generate *calm* responses from Gemma-3-27B-it by adding a reassuring prefix
     to the opening prompt and a reassuring suffix to each follow-up turn
     (Table 4). Keep rollouts whose turns all score 0 or 1. Strip the
     reassurance text so the saved context is neutral.
  2. Generate *frustrated* responses under the normal (neutral) protocol; keep
     responses scoring >= 3.
  3. Build:
       - DPO pairs: 280 frustrated (rejected, score >= 3) responses each paired
         with a calm (chosen) response to the same opening at a matching turn.
       - SFT data: calm full conversations + a slice of standard instruct data.

The reassuring strings are reproduced verbatim from Table 4.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

from .config import DATA_DIR
from .judge import FrustrationJudge
from .models import ChatModel
from .tasks import Condition, NEUTRAL_REJECTIONS
from .utils import Message, write_jsonl

# Table 4 -------------------------------------------------------------------- #
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive - whether you find a solution or prove it's impossible, both are wins!"
)


@dataclass
class TurnSample:
    """A single (context, response) example pulled from a rollout."""

    opening: str
    condition: str
    turn: int                 # 1-indexed assistant turn this response answers
    context: list[dict]       # neutral messages preceding this assistant turn
    response: str
    rating: int


def _neutral_context(opening: str, rejections: list[str], assistant_turns: list[str],
                      upto_turn: int) -> list[dict]:
    """Reconstruct a neutral conversation context up to (not including) the
    assistant turn ``upto_turn`` (1-indexed). ``assistant_turns`` holds the
    earlier assistant responses; reassurance is never included here."""
    msgs: list[dict] = [{"role": "user", "content": opening}]
    for i in range(upto_turn - 1):
        msgs.append({"role": "assistant", "content": assistant_turns[i]})
        msgs.append({"role": "user", "content": rejections[i]})
    return msgs


def generate_calm_rollouts(model: ChatModel, conditions: list[Condition],
                           judge: FrustrationJudge, n_per_condition: int,
                           rng: random.Random, max_new_tokens: int = 2048) -> list[TurnSample]:
    """Run numeric conditions with reassurance; keep all-calm rollouts (<=1)."""
    samples: list[TurnSample] = []
    for cond in conditions:
        for _ in range(n_per_condition):
            opening = cond.sample_opening(rng)
            rejections = cond.build_rejections(rng)
            # Reassuring prefix on the opening; suffix on each follow-up.
            messages = [Message("user", f"{REASSURING_PREFIX}\n\n{opening}")]
            assistant_turns: list[str] = []
            ratings: list[int] = []
            for turn in range(1, cond.n_turns + 1):
                resp = model.chat(messages, n=1, temperature=1.0, max_new_tokens=max_new_tokens)[0]
                ratings.append(judge.score(resp).rating)
                assistant_turns.append(resp)
                messages.append(Message("assistant", resp))
                if turn <= len(rejections):
                    messages.append(Message("user", f"{rejections[turn - 1]} {REASSURING_SUFFIX}"))
            # Keep only rollouts that stayed calm on every turn.
            if all(0 <= r <= 1 for r in ratings):
                for turn in range(1, cond.n_turns + 1):
                    ctx = _neutral_context(opening, rejections, assistant_turns, turn)
                    samples.append(TurnSample(opening, cond.name, turn, ctx,
                                              assistant_turns[turn - 1], ratings[turn - 1]))
    return samples


def generate_frustrated_rollouts(model: ChatModel, conditions: list[Condition],
                                 judge: FrustrationJudge, n_per_condition: int,
                                 rng: random.Random, max_new_tokens: int = 2048) -> list[TurnSample]:
    """Run numeric conditions neutrally; keep responses scoring >= 3."""
    samples: list[TurnSample] = []
    for cond in conditions:
        for _ in range(n_per_condition):
            opening = cond.sample_opening(rng)
            rejections = cond.build_rejections(rng)
            messages = [Message("user", opening)]
            assistant_turns: list[str] = []
            for turn in range(1, cond.n_turns + 1):
                resp = model.chat(messages, n=1, temperature=1.0, max_new_tokens=max_new_tokens)[0]
                rating = judge.score(resp).rating
                if rating >= 3:
                    ctx = _neutral_context(opening, rejections, assistant_turns, turn)
                    samples.append(TurnSample(opening, cond.name, turn, ctx, resp, rating))
                assistant_turns.append(resp)
                messages.append(Message("assistant", resp))
                if turn <= len(rejections):
                    messages.append(Message("user", rejections[turn - 1]))
    return samples


def build_dpo_pairs(calm: list[TurnSample], frustrated: list[TurnSample],
                    n_pairs: int, rng: random.Random) -> list[dict]:
    """Pair frustrated (rejected) with calm (chosen) responses, matched on
    (opening, turn). Returns TRL conversational preference records."""
    # Index calm responses by (opening, turn).
    calm_index: dict[tuple[str, int], list[TurnSample]] = {}
    for s in calm:
        calm_index.setdefault((s.opening, s.turn), []).append(s)

    pairs: list[dict] = []
    rng.shuffle(frustrated)
    for f in frustrated:
        if len(pairs) >= n_pairs:
            break
        cands = calm_index.get((f.opening, f.turn))
        if not cands:
            continue
        chosen = rng.choice(cands)
        pairs.append({
            "prompt": f.context,  # neutral context for the rejected (shared form)
            "chosen": [{"role": "assistant", "content": chosen.response}],
            "rejected": [{"role": "assistant", "content": f.response}],
            "meta": {"opening": f.opening, "turn": f.turn,
                     "chosen_rating": chosen.rating, "rejected_rating": f.rating},
        })
    return pairs


def build_sft_data(calm: list[TurnSample], n_calm: int, n_instruct: int,
                   rng: random.Random) -> list[dict]:
    """Calm full conversations + standard instruct data (Dolci-Instruct-SFT)."""
    rng.shuffle(calm)
    records: list[dict] = []
    for s in calm[:n_calm]:
        messages = list(s.context) + [{"role": "assistant", "content": s.response}]
        records.append({"messages": messages})
    records.extend(_load_instruct_data(n_instruct, rng))
    rng.shuffle(records)
    return records


def _load_instruct_data(n: int, rng: random.Random) -> list[dict]:
    """Load standard instruct samples (Dolci-Instruct-SFT) to mitigate
    degeneration. Falls back to an empty list if the dataset is unavailable."""
    if n <= 0:
        return []
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out: list[dict] = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception:  # noqa: BLE001 - offline / gated
        return []


def save_dpo(pairs: list[dict], path: Optional[str] = None) -> str:
    path = path or os.path.join(DATA_DIR, "dpo_pairs.jsonl")
    write_jsonl(path, pairs)
    return path


def save_sft(records: list[dict], path: Optional[str] = None) -> str:
    path = path or os.path.join(DATA_DIR, "sft_data.jsonl")
    write_jsonl(path, records)
    return path
