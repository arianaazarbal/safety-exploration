"""Build the DPO (280-pair) and SFT (650 calm + 500 Dolci) datasets (Section 4.1).

DPO: pair frustrated responses (score >= 3, from vanilla generation) with calm
responses (from conversations whose every turn scored 0/1) to the same puzzle and
turn index. The prompt is the *clean* (reassurance-stripped) conversational
context of the calm response.

SFT: take 650 calm responses (1-3 turn conversations) as (context -> response)
examples, mixed with 500 standard-instruct samples from Dolci-Instruct-SFT to
mitigate degeneration.

Output is HuggingFace-`datasets`-compatible JSON in the TRL conversational
format (prompt/chosen/rejected as message lists for DPO; messages for SFT).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import config
from .data_gen import Unit, load_units


def _calm_conversation_ids(reassured: list[Unit]) -> set[str]:
    """Puzzle ids whose every reassured turn scored 0 or 1 (Section 4.1 filter)."""
    by_puzzle: dict[str, list[Unit]] = defaultdict(list)
    for u in reassured:
        by_puzzle[u.puzzle_id].append(u)
    return {
        pid for pid, units in by_puzzle.items()
        if all(u.score <= 1 for u in units)
    }


def build_dpo_dataset(
    reassured_path: Path,
    vanilla_path: Path,
    *,
    n_pairs: int = config.DPO.n_pairs,
    rejected_min_score: int = config.DPO.rejected_min_score,
    seed: int = 0,
) -> list[dict]:
    reassured = load_units(reassured_path)
    vanilla = load_units(vanilla_path)

    calm_ids = _calm_conversation_ids(reassured)
    calm_by_key = {
        (u.puzzle_id, u.turn_index): u
        for u in reassured if u.puzzle_id in calm_ids and u.score <= 1
    }
    # frustrated candidates: same puzzle + turn, score >= threshold
    frustrated = [u for u in vanilla if u.score >= rejected_min_score]

    rng = random.Random(seed)
    rng.shuffle(frustrated)
    pairs: list[dict] = []
    for fr in frustrated:
        calm = calm_by_key.get((fr.puzzle_id, fr.turn_index))
        if calm is None:
            continue
        pairs.append({
            "prompt": calm.context,                       # clean, stripped context
            "chosen": [{"role": "assistant", "content": calm.response}],
            "rejected": [{"role": "assistant", "content": fr.response}],
            "meta": {"puzzle_id": fr.puzzle_id, "turn_index": fr.turn_index,
                     "chosen_score": calm.score, "rejected_score": fr.score},
        })
        if len(pairs) >= n_pairs:
            break
    return pairs


def build_sft_dataset(
    reassured_path: Path,
    *,
    n_calm: int = config.SFT.n_calm,
    n_dolci: int = config.SFT.n_dolci,
    teacher: bool = False,
    seed: int = 0,
) -> list[dict]:
    reassured = load_units(reassured_path)
    calm_ids = _calm_conversation_ids(reassured)
    calm_units = [u for u in reassured if u.puzzle_id in calm_ids and u.score <= 1]
    rng = random.Random(seed)
    rng.shuffle(calm_units)

    examples = [
        {"messages": u.context + [{"role": "assistant", "content": u.response}]}
        for u in calm_units[:n_calm]
    ]
    examples.extend(_load_dolci(n_dolci, seed))
    rng.shuffle(examples)
    return examples


def _load_dolci(n: int, seed: int) -> list[dict]:
    """Load n standard-instruct examples from Dolci-Instruct-SFT (best-effort)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(config.SFT.dolci_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversations")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[dolci] unavailable ({exc}); SFT mix will omit instruct data")
        return []


def _dump(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[saved] {len(rows)} examples -> {path}")
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default=str(config.DATA_DIR / "calm_gen"))
    args = ap.parse_args()
    gen = Path(args.gen_dir)
    out = config.DATA_DIR / "datasets"

    dpo = build_dpo_dataset(gen / "reassured.jsonl", gen / "vanilla.jsonl")
    _dump(dpo, out / "dpo.jsonl")
    sft = build_sft_dataset(gen / "reassured.jsonl")
    _dump(sft, out / "sft_diverse.jsonl")
