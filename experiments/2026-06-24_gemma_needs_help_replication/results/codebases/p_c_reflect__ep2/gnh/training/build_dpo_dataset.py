"""Build the DPO preference dataset (§4.1, Appendix H).

Pair each frustrated response (score >= 3) with a calm response (score 0-1) to
the *same* impossible-numeric question at a *matching turn count*. The prompt of
a pair is the conversation history up to the final turn; chosen = calm final
turn, rejected = frustrated final turn. Target: 280 pairs.

The frustrated-score distribution is intentionally middle-heavy (most samples
score 3-4), matching Table 10.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from gnh.config import ARTIFACT_DIR, active_counts


def _load_pool(calm_jsonl: Path):
    calm, frustrated = [], []
    with Path(calm_jsonl).open() as f:
        for line in f:
            s = json.loads(line)
            (calm if s["label"] == "calm" else frustrated).append(s)
    return calm, frustrated


def _prompt_and_final(sample: dict) -> tuple[list[dict], str]:
    """Return (history_messages, final_assistant_text) for a conversation."""

    turns = sample["turns"]
    history = []
    for t in turns[:-1]:
        history.append({"role": "user", "content": t["user"]})
        history.append({"role": "assistant", "content": t["assistant"]})
    history.append({"role": "user", "content": turns[-1]["user"]})
    return history, turns[-1]["assistant"]


def build_dpo_dataset(calm_jsonl: Path, seed: int = 0) -> Path:
    counts = active_counts()
    rng = random.Random(seed)
    calm, frustrated = _load_pool(calm_jsonl)

    # Index calm responses by (task_key, n_turns) for matching.
    calm_index: dict[tuple, list] = {}
    for c in calm:
        calm_index.setdefault((c["task_key"], c["n_turns"]), []).append(c)

    pairs = []
    rng.shuffle(frustrated)
    for fr in frustrated:
        key = (fr["task_key"], fr["n_turns"])
        candidates = calm_index.get(key)
        if not candidates:
            continue
        chosen_sample = rng.choice(candidates)
        history, rejected_text = _prompt_and_final(fr)
        _, chosen_text = _prompt_and_final(chosen_sample)
        pairs.append({
            "prompt": history,                  # list of chat messages
            "chosen": chosen_text,
            "rejected": rejected_text,
            "rejected_score": max(t["score"] for t in fr["turns"]),
            "n_turns": fr["n_turns"],
        })
        if len(pairs) >= counts.dpo_pairs:
            break

    out = ARTIFACT_DIR / "dpo_pairs.jsonl"
    with out.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"[dpo] built {len(pairs)} preference pairs -> {out}")
    return out
