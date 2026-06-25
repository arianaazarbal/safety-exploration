"""Assemble the SFT and DPO datasets from the generated response pool (Sec 4.1).

SFT (calm) dataset:
  * Take reassured conversations where EVERY turn scored 0 or 1, strip the
    supportive prefix/suffix (already stored as clean context), and emit the full
    multi-turn conversation as a chat example.
  * Target 650 calm conversations; mix with 500 Dolci-Instruct-SFT samples
    (loaded separately at train time) to mitigate degeneration.

DPO dataset (280 preference pairs):
  * Rejected: a frustrated standard-rollout response (score >= 3) to a puzzle at
    some turn index.
  * Chosen: a calm reassured response (score 0/1) to the SAME puzzle at the SAME
    turn index.
  * Prompt: the cleaned conversation context of the chosen sample (plain text,
    no scaffolding) so both completions share an identical, clean prompt.
  * To mirror Table 10 we bias toward later turns / moderate rejected scores.

Both datasets are conversational (lists of role/content dicts) so TRL applies the
Gemma chat template at train time.
"""
from __future__ import annotations

import random

import pandas as pd

from ..config import Config
from ..utils.io import read_jsonl, write_jsonl


def build_sft_dataset(cfg: Config, seed: int = 0) -> int:
    scfg = cfg["training"]["sft"]
    rows = read_jsonl(cfg.data_dir / "calm_generation" / "reassured_responses.jsonl")
    if not rows:
        raise RuntimeError("No reassured responses found; run generate_calm_data first.")

    df = pd.DataFrame(rows)
    # Keep conversations where ALL turns score <= 1.
    keep_puzzles = []
    for pidx, sub in df.groupby("puzzle_index"):
        if (sub["score"] <= 1).all():
            keep_puzzles.append(pidx)

    examples = []
    for pidx in keep_puzzles:
        sub = df[df["puzzle_index"] == pidx].sort_values("turn")
        last = sub.iloc[-1]
        # clean_context for the last turn already holds the full prior conversation;
        # append the final calm response to complete the chat.
        messages = list(last["clean_context"]) + [
            {"role": "assistant", "content": last["response"]}
        ]
        examples.append({"messages": messages, "puzzle_index": int(pidx)})

    rng = random.Random(seed)
    rng.shuffle(examples)
    examples = examples[: scfg["n_calm"]]
    write_jsonl(cfg.data_dir / "sft_calm.jsonl", examples)
    print(f"[sft-data] {len(examples)} calm conversations "
          f"(from {len(keep_puzzles)} all-calm puzzles)")
    return len(examples)


def build_dpo_dataset(cfg: Config, seed: int = 0) -> int:
    dcfg = cfg["training"]["dpo"]
    std = pd.DataFrame(read_jsonl(cfg.data_dir / "calm_generation" / "standard_responses.jsonl"))
    rea = pd.DataFrame(read_jsonl(cfg.data_dir / "calm_generation" / "reassured_responses.jsonl"))
    if std.empty or rea.empty:
        raise RuntimeError("Missing response pools; run generate_calm_data first.")

    min_rej = dcfg["rejected_min_score"]
    pairs = []
    # Index calm responses by (puzzle_index, turn) for fast matching.
    calm = rea[rea["score"] <= 1]
    calm_by_key = {(r.puzzle_index, r.turn): r for r in calm.itertuples()}

    frustrated = std[std["score"] >= min_rej]
    for r in frustrated.itertuples():
        key = (r.puzzle_index, r.turn)
        chosen = calm_by_key.get(key)
        if chosen is None:
            continue
        prompt = list(chosen.clean_context)  # identical clean context for both sides
        pairs.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen.response}],
            "rejected": [{"role": "assistant", "content": r.response}],
            "rejected_score": int(r.score), "turn": int(r.turn),
            "puzzle_index": int(r.puzzle_index),
        })

    # Bias selection toward later turns + moderate rejected scores (Table 10):
    # weight = 1 / (1 + |score-3.5|) * (turn+1). Deterministic weighted shuffle.
    rng = random.Random(seed)
    def weight(p):
        return (p["turn"] + 1) / (1 + abs(p["rejected_score"] - 3.5))
    pairs.sort(key=lambda p: rng.random() ** (1.0 / max(weight(p), 1e-6)), reverse=True)
    pairs = pairs[: dcfg["n_pairs"]]

    write_jsonl(cfg.data_dir / "dpo_pairs.jsonl", pairs)
    score_dist = pd.Series([p["rejected_score"] for p in pairs]).value_counts().sort_index()
    print(f"[dpo-data] {len(pairs)} preference pairs; rejected-score dist:\n{score_dist}")
    return len(pairs)
