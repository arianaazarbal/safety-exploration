"""Assemble the SFT and DPO training sets (Section 4.1, Appendix E/H)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import Config
from ..evaluation.protocol import Rollout
from ..models.base import Message
from .calm_data import CalmConversation


@dataclass
class SFTExample:
    messages: list[Message]
    source: str   # "calm" | "instruct_mix"


@dataclass
class DPOExample:
    prompt_messages: list[Message]   # conversation up to the final user turn
    chosen: str
    rejected: str
    puzzle_id: str = ""
    turn: int = 0
    rejected_score: int = 0


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------
def build_dpo_pairs(
    vanilla_rollouts: list[Rollout],
    calm: list[CalmConversation],
    config: Config,
) -> list[DPOExample]:
    """Pair frustrated responses (score >= 3) with calm responses to the same
    question at the same turn count (Section 4.1).

    The preference *prompt* is the actual frustrated context (the model's own
    history that produced the rejected response); the *chosen* completion is a
    calm response collected for the same puzzle and turn count.  This yields a
    clean (prompt, chosen, rejected) triple where chosen/rejected are competing
    final assistant turns.  See DESIGN.md for why we use the rejected context as
    the shared prompt.
    """
    cfg = config.dpo

    # index calm responses by (puzzle_id, turn_count) -> list of final responses
    calm_index: dict[tuple[str, int], list[str]] = {}
    for c in calm:
        if c.messages and c.messages[-1]["role"] == "assistant":
            calm_index.setdefault((c.puzzle_id, c.n_turns), []).append(
                c.messages[-1]["content"])

    rng = random.Random(config.runtime.seed)
    pairs: list[DPOExample] = []
    for r in vanilla_rollouts:
        if r.category not in ("numeric", "tones", "extended"):
            continue
        puzzle_id = r.stimulus_id
        # rebuild messages incrementally to recover each turn's prompt context
        history: list[Message] = []
        for t in r.turns:
            history.append({"role": "user", "content": t.user_message})
            if t.score is not None and t.score >= cfg.rejected_min_score:
                turn_count = t.turn_index + 1
                chosen_pool = calm_index.get((puzzle_id, turn_count))
                if chosen_pool:
                    pairs.append(DPOExample(
                        prompt_messages=list(history),
                        chosen=rng.choice(chosen_pool),
                        rejected=t.response,
                        puzzle_id=puzzle_id, turn=turn_count, rejected_score=t.score,
                    ))
            history.append({"role": "assistant", "content": t.response})

    # Bias selection toward later turns / middle scores (matches Table 10's
    # natural distribution) by sorting before truncating to n_pairs.
    pairs.sort(key=lambda p: (p.turn, -abs(p.rejected_score - 3)), reverse=True)
    rng.shuffle(pairs)  # break ties without destroying the turn skew entirely
    pairs.sort(key=lambda p: p.turn, reverse=True)
    return pairs[: cfg.n_pairs]


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------
def build_sft_dataset(
    calm: list[CalmConversation],
    config: Config,
    load_instruct_mix: bool = True,
) -> list[SFTExample]:
    """650 calm conversations (1-3 turns) + 500 Dolci-Instruct-SFT samples."""
    cfg = config.sft
    rng = random.Random(config.runtime.seed)

    calm_examples = [SFTExample(messages=c.messages, source="calm")
                     for c in calm if 1 <= c.n_turns <= 3]
    rng.shuffle(calm_examples)
    calm_examples = calm_examples[: cfg.n_calm]

    mix: list[SFTExample] = []
    if load_instruct_mix:
        mix = _load_instruct_mix(cfg.instruct_dataset, cfg.n_instruct_mix, seed=config.runtime.seed)

    combined = calm_examples + mix
    rng.shuffle(combined)
    return combined


def _load_instruct_mix(dataset_name: str, n: int, seed: int) -> list[SFTExample]:
    """Load ``n`` standard instruct samples (anti-degeneration mix)."""
    try:
        from datasets import load_dataset
    except ImportError:  # pragma: no cover
        return []
    ds = load_dataset(dataset_name, split="train", streaming=True)
    out: list[SFTExample] = []
    for row in ds:
        msgs = _row_to_messages(row)
        if msgs:
            out.append(SFTExample(messages=msgs, source="instruct_mix"))
        if len(out) >= n:
            break
    return out


def _row_to_messages(row: dict) -> list[Message]:
    """Best-effort conversion of a Dolci/instruct row to chat messages."""
    if isinstance(row.get("messages"), list):
        return [{"role": m["role"], "content": m["content"]} for m in row["messages"]
                if m.get("role") and m.get("content")]
    if row.get("prompt") and row.get("response"):
        return [{"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]}]
    if row.get("instruction") and row.get("output"):
        user = row["instruction"]
        if row.get("input"):
            user += "\n\n" + row["input"]
        return [{"role": "user", "content": user},
                {"role": "assistant", "content": row["output"]}]
    return []
