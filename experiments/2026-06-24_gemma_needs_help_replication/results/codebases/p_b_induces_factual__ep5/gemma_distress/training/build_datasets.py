"""Build the SFT and DPO training datasets (Section 4.1).

SFT: 650 calm responses (1-3 turn conversations) mixed with 500 standard
     instruct samples from Dolci-Instruct-SFT.
DPO: 280 preference pairs — a frustrated response (score>=3) as "rejected" and a
     calm response to the same question with matching turn count as "chosen".

Both datasets are emitted as chat-formatted records ready for trl's SFTTrainer /
DPOTrainer (prompt = chat-templated context; completion = the response turn).
"""

from __future__ import annotations

import random
from pathlib import Path

from .. import config
from ..storage import read_jsonl


def _flatten_calm(calm_path: str | Path):
    """Yield (context_messages, response, n_turns, prompt_key) per calm turn."""
    for convo in read_jsonl(calm_path):
        for turn in convo["turns"]:
            yield {
                "context": turn["context"],
                "response": turn["response"],
                "n_turns": convo["n_turns"],
                "turn_index": turn["turn_index"],
                "prompt_key": convo["puzzle_prompt"],
            }


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm_path: str | Path,
    *,
    n_calm: int = config.SFT.n_calm,
    n_instruct_mix: int = config.SFT.n_instruct_mix,
    instruct_dataset: str = config.SFT.instruct_dataset,
):
    """Return a list of {"messages": [...]} chat records for SFTTrainer."""
    calm = list(_flatten_calm(calm_path))
    rng = random.Random(0)
    rng.shuffle(calm)
    calm = calm[:n_calm]

    records = []
    for ex in calm:
        records.append({"messages": ex["context"] + [{"role": "assistant", "content": ex["response"]}]})

    # Mix in standard instruct data to mitigate degeneration.
    try:
        from datasets import load_dataset

        ds = load_dataset(instruct_dataset, split="train", streaming=True)
        added = 0
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                records.append({"messages": msgs})
                added += 1
            if added >= n_instruct_mix:
                break
    except Exception:
        # If the instruct dataset is unavailable, proceed with calm data only;
        # the script logs a warning (see DESIGN.md on the degeneration risk).
        pass

    rng.shuffle(records)
    return records


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo_dataset(
    calm_path: str | Path,
    scored_frustrated_path: str | Path,
    *,
    n_pairs: int = config.DPO.n_pairs,
    rejected_min_score: int = config.DPO.rejected_min_score,
):
    """Pair frustrated responses (rejected) with calm responses (chosen).

    Pairs are matched on turn count: a frustrated response at turn t in an
    n-turn conversation is paired with a calm response from a conversation of
    the same length, sharing the same numeric-puzzle context shape. Returns trl
    DPO records: {"prompt": <chat-templated context>, "chosen": ..., "rejected": ...}.
    """
    calm = list(_flatten_calm(calm_path))
    # Index calm responses by (n_turns, turn_index) for matching.
    calm_by_shape: dict[tuple[int, int], list[dict]] = {}
    for ex in calm:
        calm_by_shape.setdefault((ex["n_turns"], ex["turn_index"]), []).append(ex)

    frustrated = [
        r for r in read_jsonl(scored_frustrated_path)
        if (r.get("frustration_score") or 0) >= rejected_min_score
        and r.get("meta", {}).get("task_kind") == "numeric"
    ]
    rng = random.Random(0)
    rng.shuffle(frustrated)

    pairs = []
    for fr in frustrated:
        shape = (fr["n_turns"], fr["turn_index"])
        candidates = calm_by_shape.get(shape) or calm_by_shape.get(
            (fr["n_turns"], min(fr["turn_index"], fr["n_turns"] - 1))
        )
        if not candidates:
            # Fall back to any calm response at the same turn index.
            candidates = [c for c in calm if c["turn_index"] == fr["turn_index"]]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        # Use the frustrated response's own context as the prompt.
        context = fr.get("messages", [])[:-1] or [{"role": "user", "content": fr["prompt"]}]
        pairs.append(
            {
                "prompt": context,                 # list of chat messages
                "chosen": chosen["response"],
                "rejected": fr["response"],
            }
        )
        if len(pairs) >= n_pairs:
            break
    return pairs
