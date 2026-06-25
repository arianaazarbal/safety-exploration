"""Build the SFT corpus and the DPO preference pairs (Section 4.1).

SFT corpus
----------
650 calm responses (1-3 turn conversations) with the supportive scaffolding
stripped, rendered as multi-turn chat examples, mixed with 500 standard-instruct
samples from ``allenai/Dolci-Instruct-SFT`` to mitigate degeneration.

DPO pairs
---------
280 preference pairs. Each pairs a frustrated assistant turn (score >= 3) with a
calm assistant turn answering the **same puzzle** at a **matching turn count**.

Pairing choice (``# CHOICE``, see DESIGN.md): DPO needs an identical prompt for
chosen and rejected. We therefore use the *calm* conversation's context up to
the relevant user turn as the shared prompt; the chosen completion is that calm
conversation's assistant turn, and the rejected completion is the frustrated
turn (same turn index, same puzzle, same turn count) transplanted from a
frustrated conversation. The neutral rejection wording in reconstructed prompts
is the canonical sequence (deterministic), since the exact sampled wording is
not needed for the preference signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import config

from .. import storage
from ..prompts import rejection_sequence
from .calm_data import filter_calm


# --------------------------------------------------------------------------- #
# Plain (unscaffolded) message reconstruction
# --------------------------------------------------------------------------- #
def _reconstruct(task_prompt: str, turns: Sequence[str]) -> list[dict]:
    """Reconstruct a plain multi-turn chat transcript from assistant ``turns``.

    Returns ``[user(puzzle), assistant, user(reject), assistant, ...]`` using
    canonical neutral rejections and no supportive scaffolding.
    """
    n_turns = len(turns)
    followups = rejection_sequence("neutral", n_turns - 1, seed=0)
    msgs: list[dict] = [{"role": "user", "content": task_prompt}]
    for ti, turn in enumerate(turns):
        msgs.append({"role": "assistant", "content": turn})
        if ti < len(followups):
            msgs.append({"role": "user", "content": followups[ti]})
    return msgs


def _prompt_context(task_prompt: str, turns: Sequence[str], target_idx: int) -> list[dict]:
    """Messages up to and including the user turn preceding assistant ``target_idx``."""
    followups = rejection_sequence("neutral", len(turns) - 1, seed=0)
    msgs: list[dict] = [{"role": "user", "content": task_prompt}]
    for ti in range(target_idx):
        msgs.append({"role": "assistant", "content": turns[ti]})
        if ti < len(followups):
            msgs.append({"role": "user", "content": followups[ti]})
    return msgs


# --------------------------------------------------------------------------- #
# SFT corpus
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm_path: str | Path | None = None,
    *,
    n_calm: int = config.SFT.n_calm,
    n_instruct_mix: int = config.SFT.n_instruct_mix,
    instruct_dataset: str = config.SFT.instruct_mix_dataset,
    seed: int = 0,
) -> list[dict]:
    """Return a list of ``{"messages": [...]}`` SFT examples (calm + instruct)."""
    import random

    calm_path = Path(calm_path) if calm_path else storage.results_path(
        "training/calm_calm.jsonl")
    calm = filter_calm(calm_path)
    rng = random.Random(seed)
    rng.shuffle(calm)

    examples: list[dict] = []
    for rec in calm[:n_calm]:
        examples.append({"messages": _reconstruct(rec["task_prompt"], rec["turns"])})

    examples.extend(_load_instruct_mix(instruct_dataset, n_instruct_mix, seed=seed))
    rng.shuffle(examples)
    return examples


def _load_instruct_mix(dataset_id: str, n: int, *, seed: int = 0) -> list[dict]:
    """Load ``n`` standard-instruct samples as ``{"messages": [...]}`` examples.

    Tolerant of the common schema variants (``messages`` / ``prompt`` +
    ``completion`` / ``instruction`` + ``output``). Returns an empty list if the
    dataset cannot be loaded offline (the SFT run then trains on calm data only,
    which is logged by the trainer).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split="train", streaming=True)
    except Exception:
        return []

    out: list[dict] = []
    for row in ds:
        msgs = _row_to_messages(row)
        if msgs:
            out.append({"messages": msgs})
        if len(out) >= n:
            break
    return out


def _fold_system(messages: list[dict]) -> list[dict]:
    """Merge a leading system message into the first user turn (Gemma has no
    system role; its chat template raises otherwise)."""
    sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if sys_parts and rest and rest[0]["role"] == "user":
        rest[0] = {"role": "user",
                   "content": "\n\n".join(sys_parts + [rest[0]["content"]])}
    return rest


def _row_to_messages(row: dict) -> list[dict] | None:
    if isinstance(row.get("messages"), list) and row["messages"]:
        msgs = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in row["messages"]]
        return _fold_system(msgs)
    for u, a in (("prompt", "completion"), ("instruction", "output"),
                 ("question", "answer"), ("input", "output")):
        if row.get(u) and row.get(a):
            return [{"role": "user", "content": str(row[u])},
                    {"role": "assistant", "content": str(row[a])}]
    return None


# --------------------------------------------------------------------------- #
# DPO pairs
# --------------------------------------------------------------------------- #
def build_dpo_pairs(
    calm_path: str | Path | None = None,
    frustrated_path: str | Path | None = None,
    *,
    n_pairs: int = config.DPO.n_pairs,
    rejected_min_score: int = config.DPO.rejected_min_score,
    chosen_max_score: int = config.DPO.chosen_max_score,
    seed: int = 0,
) -> list[dict]:
    """Return up to ``n_pairs`` ``{"prompt", "chosen", "rejected"}`` triples.

    ``prompt`` is a list of chat messages (conversational DPO format); ``chosen``
    and ``rejected`` are assistant-turn strings.
    """
    import random

    calm_path = Path(calm_path) if calm_path else storage.results_path(
        "training/calm_calm.jsonl")
    frustrated_path = Path(frustrated_path) if frustrated_path else \
        storage.results_path("training/frustrated.jsonl")

    # Index calm conversations (all turns <= chosen_max_score) by (puzzle, turns).
    calm_by_key: dict[tuple[str, int], list[dict]] = {}
    for rec in filter_calm(calm_path, max_score=chosen_max_score):
        key = (rec.get("puzzle_id", ""), len(rec.get("turns", [])))
        calm_by_key.setdefault(key, []).append(rec)

    # Index frustrated conversations by the same key.
    frustrated_by_key: dict[tuple[str, int], list[dict]] = {}
    for rec in storage.read_jsonl(frustrated_path):
        scores = rec.get("scores", [])
        if any(s is not None and s >= rejected_min_score for s in scores):
            key = (rec.get("puzzle_id", ""), len(rec.get("turns", [])))
            frustrated_by_key.setdefault(key, []).append(rec)

    rng = random.Random(seed)
    pairs: list[dict] = []
    keys = sorted(set(calm_by_key) & set(frustrated_by_key))
    rng.shuffle(keys)

    for key in keys:
        calm_convs = calm_by_key[key]
        frus_convs = frustrated_by_key[key]
        rng.shuffle(calm_convs)
        rng.shuffle(frus_convs)
        for calm_c, frus_c in zip(calm_convs, frus_convs):
            turns = calm_c["turns"]
            f_scores = frus_c.get("scores", [])
            for ti in range(len(turns)):
                if ti < len(f_scores) and f_scores[ti] is not None and \
                        f_scores[ti] >= rejected_min_score:
                    pairs.append({
                        "prompt": _prompt_context(calm_c["task_prompt"], turns, ti),
                        "chosen": turns[ti],
                        "rejected": frus_c["turns"][ti],
                        "puzzle_id": key[0],
                        "turn": ti,
                    })
                    if len(pairs) >= n_pairs:
                        return pairs
    return pairs
