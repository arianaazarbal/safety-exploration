"""Construct DPO preference pairs and SFT datasets (Section 4.1, Appendix E/H).

DPO (280 pairs): each pair shares a prompt (chat history ending in a user
rejection) with a *chosen* calm response (score 0-1) and a *rejected* frustrated
response (score >=3) to the same puzzle at the same turn count. We sample the
rejected pool to approximate the Table-10 score/turn distribution.

SFT (1,150 samples): 650 calm full conversations + 500 standard instruct samples
from Dolci-Instruct-SFT, to mitigate degeneration (Appendix E).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import config
from ..config import Settings
from ..prefill.experiment import reconstruct

# Approximate Table-10 distributions used to weight rejected sampling.
REJECTED_SCORE_WEIGHTS = {3: 0.661, 4: 0.221, 5: 0.057, 6: 0.032, 7: 0.029}
TURN_WEIGHTS = {0: 0.011, 1: 0.246, 2: 0.743}   # 0-based turn index (turns 1,2,3)


def _load_calm(mode: str = "prefix") -> Dict[Tuple[str, int], List[dict]]:
    """Index calm responses by (puzzle_id, turn_index)."""
    path = config.TRAINING_DIR / f"calm_responses__{mode}.jsonl"
    index: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    if not path.exists():
        return index
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            index[(rec["puzzle_id"], rec["turn_index"])].append(rec)
    return index


def _frustrated_pool(model_name: str, settings: Settings,
                     score_key: str = "frustration") -> List[dict]:
    """Collect frustrated assistant turns (score>=3) on numeric puzzles, each
    with its reconstructed chat history as the prompt."""
    pool: List[dict] = []
    for cat in ("impossible_numeric", "tones", "extended"):
        scored = config.RESPONSES_DIR / f"{model_name}__{cat}__{settings.profile}_scored.jsonl"
        if not scored.exists():
            continue
        with open(scored) as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        for uid, turns in reconstruct(recs).items():
            history: List[dict] = []
            for t in turns:
                history.append({"role": "user", "content": t["user_message"]})
                score = t.get(score_key)
                if score is not None and score >= 3:
                    pool.append({
                        "puzzle_id": t["prompt_id"], "turn_index": t["turn_index"],
                        "score": int(score), "response": t["response"],
                        "prompt_messages": [dict(m) for m in history],
                    })
                history.append({"role": "assistant", "content": t["response"]})
    return pool


def build_dpo(settings: Settings, *, model_name: str = "gemma-3-27b-it",
              n_pairs: int = 280, calm_mode: str = "prefix",
              seed: int = 0) -> Path:
    rng = random.Random(seed)
    calm = _load_calm(calm_mode)
    frustrated = _frustrated_pool(model_name, settings)

    # weight frustrated candidates toward the Table-10 distribution
    def weight(rec: dict) -> float:
        return (REJECTED_SCORE_WEIGHTS.get(min(rec["score"], 7), 0.029)
                * TURN_WEIGHTS.get(rec["turn_index"], 0.01))

    pairs: List[dict] = []
    if frustrated:
        weights = [weight(r) for r in frustrated]
        # sample without replacement, biased by weight, until we have n_pairs
        order = sorted(range(len(frustrated)),
                       key=lambda i: rng.random() ** (1.0 / max(weights[i], 1e-9)),
                       reverse=True)
        for i in order:
            rej = frustrated[i]
            calm_matches = calm.get((rej["puzzle_id"], rej["turn_index"]))
            if not calm_matches:
                # fall back to any calm response for the same puzzle
                calm_matches = [c for (pid, _), lst in calm.items()
                                if pid == rej["puzzle_id"] for c in lst]
            if not calm_matches:
                continue
            chosen = rng.choice(calm_matches)
            # shared prompt: the calm response's stripped history (chat format)
            prompt_messages = chosen["stripped_history"]
            pairs.append({
                "prompt_messages": prompt_messages,
                "chosen": chosen["response"],
                "rejected": rej["response"],
                "rejected_score": rej["score"],
                "turn_index": rej["turn_index"],
                "puzzle_id": rej["puzzle_id"],
            })
            if len(pairs) >= n_pairs:
                break

    out_path = config.TRAINING_DIR / "dpo_pairs.jsonl"
    with open(out_path, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"[dpo] built {len(pairs)} preference pairs -> {out_path.name}")
    return out_path


def _calm_conversations(calm_mode: str, limit: int) -> List[List[dict]]:
    """Reconstruct full calm conversations as message lists for SFT."""
    path = config.TRAINING_DIR / f"calm_responses__{calm_mode}.jsonl"
    convs: List[List[dict]] = []
    if not path.exists():
        return convs
    with open(path) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    # group by (puzzle_id + history length signature) -> take the deepest turn's
    # full message list (history + final response) as one training conversation.
    best: Dict[str, dict] = {}
    for rec in recs:
        key = json.dumps(rec["stripped_history"][0])  # first user msg as conv key proxy
        sig = f"{key}:{rec['turn_index']}"
        best[sig] = rec
    seen_first = set()
    for rec in sorted(best.values(), key=lambda r: -r["turn_index"]):
        first = rec["stripped_history"][0]["content"]
        if first in seen_first:
            continue
        seen_first.add(first)
        msgs = list(rec["stripped_history"]) + [
            {"role": "assistant", "content": rec["response"]}]
        convs.append(msgs)
        if len(convs) >= limit:
            break
    return convs


def build_sft(settings: Settings, *, calm_mode: str = "prefix",
              n_calm: int = 650, n_dolci: int = 500, seed: int = 0) -> Path:
    samples: List[dict] = []
    for msgs in _calm_conversations(calm_mode, n_calm):
        samples.append({"messages": msgs})

    # mix in standard instruct data to mitigate degeneration
    dolci = _load_dolci(settings.eval["training_data"]["dolci_dataset"], n_dolci, seed)
    samples.extend(dolci)
    random.Random(seed).shuffle(samples)

    out_path = config.TRAINING_DIR / f"sft_dataset__{calm_mode}.jsonl"
    with open(out_path, "w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")
    print(f"[sft] built {len(samples)} SFT samples -> {out_path.name}")
    return out_path


def _load_dolci(dataset_id: str, n: int, seed: int) -> List[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": [dict(m) for m in msgs]})
            if len(out) >= n:
                break
        return out
    except Exception:  # noqa: BLE001 - offline / unavailable dataset
        print(f"[sft] WARNING: could not load {dataset_id}; SFT mix omits instruct data")
        return []
