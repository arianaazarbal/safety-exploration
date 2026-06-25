"""Build the SFT dataset (Section 4.1 / Appendix E).

650 calm conversations (1–3 turns), mixed with 500 standard instruct samples
from Dolci-Instruct-SFT to mitigate degeneration. Output records are full chat
conversations::

    {"messages": [{"role": "user"|"assistant", "content": ...}, ...]}

The calm conversations are reconstructed from the all-calm data by replaying
each conversation's plain context and its calm responses in order.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import config
from .generate_calm_data import CALM_PATH

SFT_PATH = config.DATA_DIR / "sft_conversations.jsonl"


def _load_calm() -> list[dict]:
    if not CALM_PATH.exists():
        raise FileNotFoundError("Run generate_calm_responses() first.")
    return [json.loads(l) for l in CALM_PATH.read_text().splitlines() if l]


def _reconstruct_calm_conversations(calm: list[dict]) -> list[dict]:
    """Group per-turn calm records back into full conversations.

    Records from the same conversation share an identical leading context; the
    final-turn record (max turn_index for that root) carries the full history.
    We rebuild messages from the deepest record's context + response.
    """
    # Group by the conversation's first user turn (the puzzle prompt) — unique
    # enough for the synthetic calm set.
    by_root: dict[str, list[dict]] = {}
    for rec in calm:
        root = rec["context"][0]["content"]
        by_root.setdefault(root, []).append(rec)

    conversations = []
    for recs in by_root.values():
        deepest = max(recs, key=lambda r: r["turn_index"])
        messages = list(deepest["context"]) + [
            {"role": "assistant", "content": deepest["response"]}]
        conversations.append({"messages": messages})
    return conversations


def _load_dolci(n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset(config.DOLCI_INSTRUCT_DATASET, split="train",
                          streaming=True, token=config.HF_TOKEN or None)
    except Exception:  # noqa: BLE001 — offline / gated
        print("[sft] Dolci-Instruct-SFT unavailable; skipping instruct mix.")
        return []
    rng = random.Random(seed)
    pool = []
    for i, row in enumerate(ds):
        if i >= n * 20 or len(pool) >= n * 5:
            break
        msgs = row.get("messages") or row.get("conversation")
        if msgs and isinstance(msgs, list):
            pool.append({"messages": [
                {"role": m.get("role"), "content": m.get("content", "")}
                for m in msgs if m.get("role") in ("user", "assistant")]})
    return rng.sample(pool, min(n, len(pool))) if pool else []


def build_sft_dataset(calm_n: int = config.SFTTrainConfig().calm_samples,
                      instruct_n: int = config.SFTTrainConfig().instruct_mix_samples,
                      seed: int = config.SEED) -> Path:
    rng = random.Random(seed)
    calm_convos = _reconstruct_calm_conversations(_load_calm())
    rng.shuffle(calm_convos)
    calm_convos = calm_convos[:calm_n]

    dolci = _load_dolci(instruct_n, seed)
    examples = calm_convos + dolci
    rng.shuffle(examples)

    with SFT_PATH.open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    print(f"[sft] wrote {len(calm_convos)} calm + {len(dolci)} instruct "
          f"= {len(examples)} examples -> {SFT_PATH}")
    return SFT_PATH
