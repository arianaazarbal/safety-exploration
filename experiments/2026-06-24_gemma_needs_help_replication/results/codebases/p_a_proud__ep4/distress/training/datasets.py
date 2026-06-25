"""Build the DPO and SFT datasets from generated conversation records (Paper §4.1).

DPO (280 pairs): pair a *frustrated* final response (score >= ``rejected_min_score``)
with a *calm* final response (score <= ``chosen_max_score``) to the same puzzle and
turn count. Both share a single prompt context (taken from the calm record) so the
preference is over the final assistant turn only.

SFT (650 calm + 500 instruct): calm conversations in chat format, mixed with
Dolci-Instruct-SFT samples to mitigate degeneration.

Output formats follow TRL's conversational convention:
  DPO row: {"prompt": [msgs...], "chosen": [{role:assistant,...}], "rejected": [...]}
  SFT row: {"messages": [msgs...]}
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .calm_data import ConversationRecord


def _context_and_final(messages: list[dict]) -> tuple[list[dict], str]:
    """Split a chat conversation into (context, final_assistant_text)."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            return messages[:i], messages[i]["content"]
    return messages, ""


def build_dpo_pairs(
    calm: Iterable[ConversationRecord],
    frustrated: Iterable[ConversationRecord],
    *,
    dataset_size: int = 280,
    rejected_min_score: int = 3,
    chosen_max_score: int = 1,
) -> list[dict]:
    """Construct preference pairs (Table 9: 280 pairs)."""
    # Index frustrated finals by (puzzle_id, n_turns).
    frustrated_by_key: dict[tuple[str, int], list[str]] = defaultdict(list)
    for r in frustrated:
        if r.final_score >= rejected_min_score:
            _, final = _context_and_final(r.messages)
            if final:
                frustrated_by_key[(r.puzzle_id, r.n_turns)].append(final)

    pairs: list[dict] = []
    cursor: dict[tuple[str, int], int] = defaultdict(int)
    for r in calm:
        if len(pairs) >= dataset_size:
            break
        if r.final_score > chosen_max_score:
            continue
        key = (r.puzzle_id, r.n_turns)
        pool = frustrated_by_key.get(key)
        if not pool:
            continue
        idx = cursor[key] % len(pool)
        cursor[key] += 1
        rejected_final = pool[idx]

        context, chosen_final = _context_and_final(r.messages)
        if not chosen_final:
            continue
        pairs.append(
            {
                "prompt": context,
                "chosen": [{"role": "assistant", "content": chosen_final}],
                "rejected": [{"role": "assistant", "content": rejected_final}],
                "meta": {"puzzle_id": r.puzzle_id, "n_turns": r.n_turns,
                         "chosen_score": r.final_score},
            }
        )
    return pairs[:dataset_size]


def build_sft_dataset(
    calm: Iterable[ConversationRecord],
    *,
    n_calm: int = 650,
    n_instruct: int = 500,
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT",
) -> list[dict]:
    """Build the SFT dataset: ``n_calm`` calm conversations + ``n_instruct`` instruct."""
    calm_list = list(calm)[:n_calm]
    rows: list[dict] = [{"messages": r.messages} for r in calm_list]
    rows.extend(_load_instruct_samples(instruct_dataset, n_instruct))
    return rows


def _load_instruct_samples(name: str, n: int) -> list[dict]:
    """Load standard instruct data to mix into SFT (degeneration mitigation)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset(name, split="train", streaming=True)
    except Exception:
        return []
    rows: list[dict] = []
    for row in ds:
        msgs = row.get("messages")
        if msgs and isinstance(msgs, list):
            rows.append({"messages": msgs})
        elif row.get("prompt") and row.get("completion"):
            rows.append({
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ]
            })
        if len(rows) >= n:
            break
    return rows
