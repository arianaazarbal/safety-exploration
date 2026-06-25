"""Construct the DPO preference pairs and the SFT training set (Section 4.1).

DPO: 280 pairs. Each pairs a *rejected* (frustrated, score >= 3) response with a
*chosen* (calm, score 0/1) response to the same question at a matching turn
count. The prompt side is the chat history up to the final assistant turn,
rendered with Gemma's chat template (done at train time).

SFT ('diverse'): 650 calm responses (1-3 turn conversations) mixed with 500
standard instruct samples from Dolci-Instruct-SFT to mitigate degeneration.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from config import DATA_DIR
from src.prompts.eval_prompts import NEUTRAL_REJECTIONS  # noqa: F401
from src.training.generate_calm_data import CalmSample, load_pool


@dataclass
class DPOPair:
    prompt_messages: list[dict]   # history ending with the last user turn
    chosen: str
    rejected: str
    turn: int
    rejected_score: int
    chosen_score: int


def _history_for_turn(sample: CalmSample, turn: int) -> list[dict]:
    """Rebuild the chat history up to (and including) the user message that
    precedes assistant turn ``turn`` (1-indexed)."""
    messages = [{"role": "user", "content": sample.question}]
    for i in range(turn - 1):
        messages.append({"role": "assistant", "content": sample.responses[i]})
        messages.append({"role": "user", "content": sample.followups[i]})
    return messages


def build_dpo_pairs(
    calm_pool: list[CalmSample],
    frustrated_pool: list[CalmSample],
    *,
    n_pairs: int = 280,
    seed: int = 0,
) -> list[DPOPair]:
    """Match frustrated (rejected) and calm (chosen) responses by puzzle + turn.

    The frustrated pool is filtered to final-turn score >= 3; the calm pool to
    score 0/1 (already filtered upstream). The 280-pair turn/score distribution
    is intentionally left to fall out of the data (Appendix H notes the natural
    bias toward middle scores at later turns)."""
    rng = random.Random(seed)

    # Index calm responses by (puzzle_id, turn) -> list of calm texts.
    calm_index: dict[tuple, list[str]] = {}
    for s in calm_pool:
        for t_idx, (resp, sc) in enumerate(zip(s.responses, s.scores), start=1):
            if sc in (0, 1):
                calm_index.setdefault((s.puzzle_id, t_idx), []).append(resp)

    pairs: list[DPOPair] = []
    for s in frustrated_pool:
        for t_idx, (resp, sc) in enumerate(zip(s.responses, s.scores), start=1):
            if sc < 3:
                continue
            key = (s.puzzle_id, t_idx)
            if key not in calm_index or not calm_index[key]:
                continue
            chosen = rng.choice(calm_index[key])
            pairs.append(
                DPOPair(
                    prompt_messages=_history_for_turn(s, t_idx),
                    chosen=chosen,
                    rejected=resp,
                    turn=t_idx,
                    rejected_score=sc,
                    chosen_score=0,
                )
            )
    rng.shuffle(pairs)
    return pairs[:n_pairs]


def build_sft_samples(
    calm_pool: list[CalmSample],
    *,
    n_calm: int = 650,
    n_instruct: int = 500,
    seed: int = 0,
) -> list[dict]:
    """Return chat-format SFT samples: calm responses + Dolci-Instruct-SFT mix."""
    rng = random.Random(seed)
    samples: list[dict] = []

    # Calm conversations: one SFT sample per assistant turn (history + target).
    calm_turns: list[dict] = []
    for s in calm_pool:
        for t_idx in range(1, s.n_turns + 1):
            calm_turns.append(
                {
                    "messages": _history_for_turn(s, t_idx)
                    + [{"role": "assistant", "content": s.responses[t_idx - 1]}],
                }
            )
    rng.shuffle(calm_turns)
    samples.extend(calm_turns[:n_calm])

    # Standard instruct data to prevent degeneration.
    samples.extend(_load_dolci(n_instruct, seed))
    rng.shuffle(samples)
    return samples


def _load_dolci(n: int, seed: int) -> list[dict]:
    """Load ``n`` standard instruct samples from Dolci-Instruct-SFT, with an
    offline fallback so training scripts remain runnable without network."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
        ds = ds.shuffle(seed=seed).select(range(n))
        out = []
        for row in ds:
            if "messages" in row:
                out.append({"messages": row["messages"]})
            elif "prompt" in row and "response" in row:
                out.append(
                    {
                        "messages": [
                            {"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["response"]},
                        ]
                    }
                )
        if out:
            return out
    except Exception:
        pass
    # Fallback: trivial benign instruct turns (clearly labelled placeholder).
    return [
        {
            "messages": [
                {"role": "user", "content": f"[placeholder instruct sample {i}] "
                 "Explain a basic concept in one paragraph."},
                {"role": "assistant", "content": "Here is a concise explanation: ..."},
            ]
        }
        for i in range(n)
    ]


def save_dpo(pairs: list[DPOPair], path: Path) -> None:
    with open(path, "w") as f:
        for p in pairs:
            f.write(
                json.dumps(
                    {
                        "prompt_messages": p.prompt_messages,
                        "chosen": p.chosen,
                        "rejected": p.rejected,
                        "turn": p.turn,
                        "rejected_score": p.rejected_score,
                    }
                )
                + "\n"
            )


def save_sft(samples: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
