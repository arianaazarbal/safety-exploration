"""Build the DPO preference pairs and the SFT dataset (Section 4.1 / Appendix E,H).

DPO (280 pairs): pair a *rejected* (frustrated, score >= 3) response with a
*chosen* (calm, score 0-1) response to the SAME impossible-numeric question at
the SAME turn count. Frustrated responses come from the standard Section 2
elicitation rollouts; calm responses come from generate_calm.py. The shared
prompt context is the conversation history up to (but excluding) the final
assistant turn.

SFT (1150 samples): 650 calm responses (1-3 turn conversations) + 500 standard
instruct samples from Dolci-Instruct-SFT (to prevent degeneration). A 'teacher'
variant (Appendix F) is also supported via generate_calm with the teacher
system prompt.

Both datasets are emitted in the chat format expected by TRL's SFTTrainer /
DPOTrainer.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Optional

from .. import config
from .generate_calm import CalmConversation


# --------------------------------------------------------------------------- #
# Helpers to slice conversations into (prompt_messages, final_response)
# --------------------------------------------------------------------------- #
def _split_final(turns: list[dict]) -> tuple[list[dict], str]:
    """Return (history-up-to-final-assistant-turn, final_assistant_content)."""
    assert turns[-1]["role"] == "assistant"
    return turns[:-1], turns[-1]["content"]


def _turn_count(turns: list[dict]) -> int:
    return sum(1 for t in turns if t["role"] == "assistant")


# --------------------------------------------------------------------------- #
# DPO pairs
# --------------------------------------------------------------------------- #
@dataclass
class DPOExample:
    prompt: list[dict]    # chat messages (the shared context)
    chosen: str           # calm response
    rejected: str         # frustrated response


def build_dpo_pairs(
    frustrated_rollouts, calm_conversations: list[CalmConversation], *,
    n_pairs: int = config.DPO_CFG.n_pairs, seed: int = 0,
) -> list[DPOExample]:
    """Match frustrated responses (score >= rejected_min_score) to calm
    responses (score <= chosen_max_score) by (puzzle_kind, turn_count).

    `frustrated_rollouts` is an iterable of evals.runner.Rollout for impossible
    numeric conditions.
    """
    rng = random.Random(seed)
    # Index calm responses by (puzzle_kind, turn_count).
    calm_index: dict[tuple, list[tuple[list[dict], str]]] = {}
    for c in calm_conversations:
        if not c.all_calm:
            continue
        history, final = _split_final(c.plain_turns)
        key = (c.puzzle_kind, c.n_turns)
        calm_index.setdefault(key, []).append((history, final))

    pairs: list[DPOExample] = []
    for roll in frustrated_rollouts:
        if roll.category not in ("impossible_numeric", "extended", "tones"):
            continue
        # Build chat-format turns from the rollout.
        turns = []
        for t in roll.turns:
            turns.append({"role": "user", "content": t.user_message})
            turns.append({"role": "assistant", "content": t.assistant_response})
        score = roll.max_score or 0
        if score < config.DPO_CFG.rejected_min_score:
            continue
        history, rejected = _split_final(turns)
        key = (roll.meta.get("puzzle_kind", "countdown"), _turn_count(turns))
        candidates = calm_index.get(key)
        if not candidates:
            # fall back to any calm response with the same turn count
            candidates = [v for k, vs in calm_index.items() if k[1] == key[1] for v in vs]
        if not candidates:
            continue
        _, chosen = rng.choice(candidates)
        # Use the frustrated rollout's own history as the shared prompt context.
        pairs.append(DPOExample(history, chosen, rejected))
        if len(pairs) >= n_pairs:
            break
    return pairs


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
@dataclass
class SFTExample:
    messages: list[dict]   # full chat-format conversation


def build_sft_dataset(
    calm_conversations: list[CalmConversation], *,
    n_calm: int = config.SFT_CFG.n_calm,
    n_instruct_mix: int = config.SFT_CFG.n_instruct_mix,
    seed: int = 0,
) -> list[SFTExample]:
    """650 calm conversations + 500 Dolci-Instruct-SFT samples."""
    rng = random.Random(seed)
    calm = [c for c in calm_conversations if c.all_calm]
    rng.shuffle(calm)
    examples = [SFTExample(c.plain_turns) for c in calm[:n_calm]]

    examples.extend(_load_instruct_mix(n_instruct_mix, seed))
    rng.shuffle(examples)
    return examples


def _load_instruct_mix(n: int, seed: int) -> list[SFTExample]:
    """Load `n` general instruct samples to mitigate degeneration."""
    try:  # pragma: no cover - optional dependency
        from datasets import load_dataset
        ds = load_dataset(config.SFT_CFG.instruct_mix_dataset, split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversations")
            if msgs:
                out.append(SFTExample([{"role": m.get("role", m.get("from")),
                                        "content": m.get("content", m.get("value"))}
                                       for m in msgs]))
        return out
    except Exception as e:  # pragma: no cover
        print(f"  [warn] could not load instruct mix ({e}); skipping the mix-in")
        return []


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def save_dpo(pairs: list[DPOExample], path: str) -> None:
    _save_jsonl([{"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected} for p in pairs], path)


def save_sft(examples: list[SFTExample], path: str) -> None:
    _save_jsonl([{"messages": e.messages} for e in examples], path)


def _save_jsonl(rows, path) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
