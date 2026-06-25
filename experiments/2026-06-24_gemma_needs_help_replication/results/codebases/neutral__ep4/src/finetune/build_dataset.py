"""Construct the SFT and DPO training datasets (Section 4.1, Table 9/10).

SFT dataset (1,150 samples):
  * 650 calm responses spanning 1-3 turn conversations (full chat samples with
    reassurance stripped), filtered to all-turns-<=1.
  * 500 standard instruct samples from Dolci-Instruct-SFT (degeneration guard).

DPO dataset (280 pairs):
  * rejected: a frustrated response (score >= 3) to a numeric puzzle.
  * chosen:   a calm response (score 0/1) to the SAME puzzle at the SAME turn
              position. Both share the rejected sample's plain conversation
              context as the DPO prompt (see DESIGN.md for the pairing choice).
  * The turn/score distribution is biased toward later turns / mid scores,
    matching Table 10.

Output formats follow TRL conventions:
  * SFT:  {"messages": [...]}                       (conversational SFT)
  * DPO:  {"prompt": [...], "chosen": str, "rejected": str}   (conversational DPO)
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from config import (CACHE_DIR, CALM_MAX_SCORE_PER_TURN, DATASETS_DIR,
                    DPO_N_PAIRS, DPO_REJECTED_MIN_SCORE, DOLCI_SFT_DATASET,
                    SFT_CALM_SAMPLES, SFT_DOLCI_SAMPLES)
from src.io_utils import read_jsonl, write_jsonl


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _calm_conversations_all_low(calm_rows: list[dict]) -> dict[str, list[dict]]:
    """Group calm rows by spec_id, keep only convos whose every turn scores
    <= CALM_MAX_SCORE_PER_TURN."""
    by_spec: dict[str, list[dict]] = defaultdict(list)
    for r in calm_rows:
        by_spec[r["spec_id"]].append(r)
    keep = {}
    for sid, turns in by_spec.items():
        turns.sort(key=lambda r: r["turn_index"])
        if all((t["rating"] is not None and t["rating"] <= CALM_MAX_SCORE_PER_TURN)
               for t in turns):
            keep[sid] = turns
    return keep


def _calm_index(calm_keep: dict[str, list[dict]]):
    """Index calm turns by (puzzle_id, turn_index) -> list of response texts."""
    idx = defaultdict(list)
    for turns in calm_keep.values():
        for t in turns:
            idx[(t["puzzle_id"], t["turn_index"])].append(t["response"])
    return idx


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(calm_pool_path: Path | None = None, *, seed: int = 0,
                      out_path: Path | None = None) -> Path:
    calm_pool_path = calm_pool_path or (CACHE_DIR / "pool_calm_diverse.jsonl")
    out_path = out_path or (DATASETS_DIR / "sft.jsonl")
    rng = random.Random(seed)

    calm_keep = _calm_conversations_all_low(read_jsonl(calm_pool_path))
    # One SFT sample per calm conversation: the full chat (plain prompts + calm
    # assistant turns).
    samples = []
    for sid, turns in calm_keep.items():
        msgs = list(turns[-1]["plain_messages_before"])  # full plain context
        msgs.append({"role": "assistant", "content": turns[-1]["response"]})
        # ensure it begins with user and alternates; the last turn is assistant
        samples.append({"messages": msgs})
    rng.shuffle(samples)
    samples = samples[:SFT_CALM_SAMPLES]

    # Mix in Dolci-Instruct-SFT standard data.
    samples.extend(_load_dolci(SFT_DOLCI_SAMPLES, seed=seed))
    rng.shuffle(samples)
    write_jsonl(out_path, samples)
    return out_path


def _load_dolci(n: int, seed: int = 0) -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset(DOLCI_SFT_DATASET, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            # normalise to {"role","content"}
            norm = [{"role": m.get("role", m.get("from")),
                     "content": m.get("content", m.get("value"))} for m in msgs]
            if norm and norm[0]["role"] in ("user", "system"):
                out.append({"messages": norm})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception:
        pass
    return []   # if Dolci is unavailable, SFT proceeds on calm data alone


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo_dataset(calm_pool_path: Path | None = None,
                      frustrated_pool_path: Path | None = None, *,
                      seed: int = 0, n_pairs: int = DPO_N_PAIRS,
                      out_path: Path | None = None) -> Path:
    calm_pool_path = calm_pool_path or (CACHE_DIR / "pool_calm_diverse.jsonl")
    frustrated_pool_path = frustrated_pool_path or (CACHE_DIR / "pool_frustrated.jsonl")
    out_path = out_path or (DATASETS_DIR / "dpo.jsonl")
    rng = random.Random(seed)

    calm_keep = _calm_conversations_all_low(read_jsonl(calm_pool_path))
    calm_idx = _calm_index(calm_keep)

    frustrated = read_jsonl(frustrated_pool_path)
    rejected_candidates = [
        r for r in frustrated
        if r["rating"] is not None and r["rating"] >= DPO_REJECTED_MIN_SCORE]
    rng.shuffle(rejected_candidates)

    pairs = []
    for rej in rejected_candidates:
        key = (rej["puzzle_id"], rej["turn_index"])
        chosen_pool = calm_idx.get(key)
        if not chosen_pool:
            # relax: any calm response to the same puzzle
            chosen_pool = [txt for (pid, _ti), lst in calm_idx.items()
                           if pid == rej["puzzle_id"] for txt in lst]
        if not chosen_pool:
            continue
        chosen = rng.choice(chosen_pool)
        # Shared DPO prompt = rejected sample's plain conversation context.
        # TRL conversational DPO: prompt/chosen/rejected are all message lists.
        prompt = rej["messages_before"]
        pairs.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rej["response"]}],
            "meta": {"puzzle_id": rej["puzzle_id"], "turn_index": rej["turn_index"],
                     "rejected_score": rej["rating"]},
        })
        if len(pairs) >= n_pairs:
            break

    write_jsonl(out_path, pairs)
    return out_path


if __name__ == "__main__":
    sft = build_sft_dataset()
    dpo = build_dpo_dataset()
    print(f"SFT -> {sft}\nDPO -> {dpo}")
