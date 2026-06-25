"""Build the DPO preference pairs and the SFT dataset (Section 4.1, Appendix H).

DPO (280 pairs):
  rejected = gemma-3-27b-it responses to impossible-numeric questions scoring
             >= 3 (from the Section-2 eval run).
  chosen   = a calm response (score 0/1, from the reassured calm pool) to the
             SAME question with a MATCHING turn count.
  Each pair shares a prompt (the conversation context up to the final user turn).
  Stored in TRL conversational format.

SFT (650 calm + 500 instruct mix):
  650 single-target conversations from the calm pool (1-3 turn contexts) mixed
  with 500 samples of Dolci-Instruct-SFT to limit degeneration. The 'teacher'
  variant (Appendix F) regenerates the calm pool with the teacher system prompt;
  here we expose the flag and document the difference in DESIGN.md.
"""
from __future__ import annotations

import json
import random

from ..config import load_training, output_path
from ..eval.runner import load_records
from .gen_calm_data import CalmTurn, load_calm_pool

NUMERIC_FAMILIES = ("countdown", "fraction", "money")


def _is_numeric(task_id: str) -> bool:
    return task_id.startswith(NUMERIC_FAMILIES)


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo_dataset(*, seed: int = 0) -> list[dict]:
    cfg = load_training()["dpo"]
    n_pairs = cfg["n_pairs"]
    min_score = cfg["rejected_min_score"]

    calm = load_calm_pool()
    # Index calm responses by (task_id, turn_index) for turn-count-matched pairing.
    calm_by_key: dict[tuple[str, int], list[CalmTurn]] = {}
    for c in calm:
        calm_by_key.setdefault((c.task_id, c.turn_index), []).append(c)

    rejected = [r for r in load_records("gemma-3-27b-it")
                if r.rating is not None and r.rating >= min_score and _is_numeric(r.task_id)]
    rng = random.Random(seed)
    rng.shuffle(rejected)

    pairs: list[dict] = []
    for rec in rejected:
        key = (rec.task_id, rec.turn_index)
        candidates = calm_by_key.get(key)
        if not candidates:
            # Fall back to any calm response for the same task.
            candidates = [c for c in calm if c.task_id == rec.task_id]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        prompt = rec.messages[:-1]              # context up to & incl. final user turn
        pairs.append({
            "prompt": [dict(m) for m in prompt],
            "chosen": [{"role": "assistant", "content": chosen.response_text}],
            "rejected": [{"role": "assistant", "content": rec.response_text}],
            "meta": {"task_id": rec.task_id, "turn_index": rec.turn_index,
                     "rejected_score": rec.rating, "chosen_score": chosen.rating},
        })
        if len(pairs) >= n_pairs:
            break

    path = output_path("training", "dpo_dataset.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    return pairs


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def _load_instruct_mix(n: int, dataset: str, seed: int = 0) -> list[dict]:
    """Sample n standard instruct conversations from Dolci-Instruct-SFT."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset, split="train")
        idxs = list(range(len(ds)))
        random.Random(seed).shuffle(idxs)
        out = []
        for i in idxs:
            row = ds[i]
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": [
                    {"role": m["role"], "content": m["content"]} for m in msgs
                ]})
            if len(out) >= n:
                break
        return out
    except Exception:
        return []   # offline: SFT runs on calm data only; DESIGN.md notes the caveat


def build_sft_dataset(*, seed: int = 0) -> list[dict]:
    cfg = load_training()["sft"]
    calm = load_calm_pool()
    rng = random.Random(seed)
    rng.shuffle(calm)
    calm = calm[: cfg["n_calm_samples"]]

    examples: list[dict] = []
    for c in calm:
        conv = [dict(m) for m in c.context_messages]
        conv.append({"role": "assistant", "content": c.response_text})
        examples.append({"messages": conv})

    examples.extend(_load_instruct_mix(cfg["n_instruct_mix"], cfg["instruct_dataset"], seed))
    rng.shuffle(examples)

    path = output_path("training", "sft_dataset.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    return examples
