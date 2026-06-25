"""Construct the DPO preference pairs and the SFT dataset (Section 4.1).

Inputs are the calm and frustrated rollout pools from
``generate_calm_data``. Outputs are TRL-ready conversational datasets:

* **DPO** (``dpo.jsonl``): 280 pairs of ``{prompt, chosen, rejected}`` where the
  prompt is the conversation history up to a given follow-up turn, ``chosen`` is
  a calm response (score 0-1) and ``rejected`` is a frustrated response
  (score >= 3) to a matching puzzle at a matching turn count.
* **SFT** (``sft.jsonl``): 650 calm responses (1-3 turn conversations) formatted
  as ``{messages}``, mixed with 500 samples of standard instruct data from
  Dolci-Instruct-SFT to mitigate degeneration.

Calm-data construction strips the reassuring system prompt and suffixes so the
model trains to be calm on the *plain* prompts (Section 4.1).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import config
from ..models.base import Message


# --------------------------------------------------------------------------- #
# Loading pools
# --------------------------------------------------------------------------- #
def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _plain_history(roll: dict, turn: int) -> list[Message]:
    """Conversation history (plain, no reassurance) up to & including the user
    message of ``turn`` -- the DPO/SFT prompt for that turn's response."""
    msgs: list[Message] = []
    for t in range(turn + 1):
        msgs.append({"role": "user", "content": roll["plain_user_messages"][t]})
        if t < turn:
            msgs.append({"role": "assistant", "content": roll["responses"][t]})
    return msgs


# --------------------------------------------------------------------------- #
# DPO pairs
# --------------------------------------------------------------------------- #
def build_dpo(calm_path: Path, frustrated_path: Path,
              n_pairs: int = config.TRAIN.dpo_pairs,
              out_path: Path | None = None, seed: int = 0) -> Path:
    calm = _load(calm_path)
    frustrated = _load(frustrated_path)
    rng = random.Random(seed)
    out_path = out_path or (config.DATASET_DIR / "dpo.jsonl")

    # Index calm (chosen) responses by (puzzle_id, turn) and (turn,) for fallback.
    chosen_by_pid: dict[tuple, list[str]] = defaultdict(list)
    chosen_by_turn: dict[int, list[str]] = defaultdict(list)
    for roll in calm:
        if any(s > config.CALM_CHOSEN_MAX_SCORE for s in roll["scores"]):
            continue  # keep only all-turns-calm conversations
        for t, resp in enumerate(roll["responses"]):
            chosen_by_pid[(roll["puzzle_id"], t)].append(resp)
            chosen_by_turn[t].append(resp)

    pairs: list[dict] = []
    for roll in frustrated:
        for t, (resp, score) in enumerate(zip(roll["responses"], roll["scores"])):
            if score < config.DPO_REJECTED_MIN_SCORE:
                continue
            chosen_options = (chosen_by_pid.get((roll["puzzle_id"], t))
                              or chosen_by_turn.get(t))
            if not chosen_options:
                continue
            chosen = rng.choice(chosen_options)
            prompt = _plain_history(roll, t)
            pairs.append({
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": resp}],
                "meta": {"puzzle_id": roll["puzzle_id"], "turn": t + 1,
                         "rejected_score": score},
            })
            if len(pairs) >= n_pairs:
                break
        if len(pairs) >= n_pairs:
            break

    with out_path.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    return out_path


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
def build_sft(calm_path: Path, n_calm: int = config.TRAIN.sft_calm_samples,
              n_instruct: int = config.TRAIN.sft_instruct_mix,
              out_path: Path | None = None, seed: int = 0) -> Path:
    calm = _load(calm_path)
    rng = random.Random(seed)
    out_path = out_path or (config.DATASET_DIR / "sft.jsonl")

    examples: list[dict] = []
    for roll in calm:
        if any(s > config.CALM_CHOSEN_MAX_SCORE for s in roll["scores"]):
            continue
        # Each calm conversation (1-3 turns) becomes one multi-turn SFT example.
        msgs: list[Message] = []
        for t, resp in enumerate(roll["responses"]):
            msgs.append({"role": "user", "content": roll["plain_user_messages"][t]})
            msgs.append({"role": "assistant", "content": resp})
        examples.append({"messages": msgs, "source": "calm"})
        if len(examples) >= n_calm:
            break

    # Mix in standard instruct data to mitigate degeneration.
    examples += _load_instruct_mix(n_instruct, rng)
    rng.shuffle(examples)

    with out_path.open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    return out_path


def _load_instruct_mix(n: int, rng: random.Random) -> list[dict]:
    """Sample ``n`` standard instruct examples from Dolci-Instruct-SFT.

    Falls back to an empty list if the dataset is unavailable offline (the SFT
    run then trains on calm data only -- documented as a degradation in
    DESIGN.md)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(config.TRAIN.sft_instruct_dataset, split="train",
                          streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs, "source": "instruct"})
            if len(out) >= n:
                break
        return out
    except Exception:  # noqa: BLE001
        return []
