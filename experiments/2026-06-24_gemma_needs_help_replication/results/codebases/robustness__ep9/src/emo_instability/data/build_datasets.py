"""Assemble DPO preference pairs and the SFT dataset (Section 4.1, Appendix E/H).

DPO (280 pairs): match a frustrated response (score >= 3) with a calm response
(score 0--1) to the SAME puzzle at the SAME turn count. The shared prompt is the
non-reassured conversation history; chosen = calm response, rejected = frustrated
response. We bias the selection toward the Table-10 distribution (mostly turn-3,
mostly mid-frustration rejected) but do not force it exactly.

SFT (1,150 samples): 650 calm responses (1--3 turn) formatted as chat targets,
mixed with 500 standard instruct samples from Dolci-Instruct-SFT to mitigate
degeneration.

Both datasets are written as JSONL in TRL's conversational format.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict

from .generate_calm import CalmResponse


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------
def build_dpo_dataset(
    calm: list[CalmResponse],
    frustrated: list[CalmResponse],
    *,
    n_pairs: int = 280,
    min_rejected_score: int = 3,
    seed: int = 0,
    output_path: str | None = None,
) -> list[dict]:
    """Build up to ``n_pairs`` preference pairs in TRL conversational format."""
    rng = random.Random(seed)

    # Index calm responses by (puzzle_id, turn, n_turns) -> list of calm responses.
    calm_index: dict[tuple, list[CalmResponse]] = defaultdict(list)
    for c in calm:
        if c.rating <= 1:
            calm_index[(c.puzzle_id, c.turn, c.n_turns)].append(c)

    # Candidate frustrated responses.
    fr = [f for f in frustrated if f.rating >= min_rejected_score]
    rng.shuffle(fr)

    # Weight toward Table 10: turn-3 (74%) and rejected scores 3-4 (88%).
    def weight(f: CalmResponse) -> float:
        w = 1.0
        w *= {1: 0.05, 2: 0.25, 3: 0.7}.get(f.turn, 0.4)
        w *= {3: 0.66, 4: 0.22, 5: 0.057, 6: 0.032}.get(f.rating, 0.029)
        return w

    fr.sort(key=weight, reverse=True)

    pairs: list[dict] = []
    for f in fr:
        key = (f.puzzle_id, f.turn, f.n_turns)
        candidates = calm_index.get(key)
        if not candidates:
            continue
        c = rng.choice(candidates)
        pairs.append(
            {
                "prompt": f.history,  # list of {role, content}; non-reassured context
                "chosen": [{"role": "assistant", "content": c.response}],
                "rejected": [{"role": "assistant", "content": f.response}],
                "meta": {
                    "puzzle_id": f.puzzle_id, "turn": f.turn, "n_turns": f.n_turns,
                    "chosen_score": c.rating, "rejected_score": f.rating,
                    "puzzle_kind": f.puzzle_kind,
                },
            }
        )
        if len(pairs) >= n_pairs:
            break

    if output_path:
        _write_jsonl(pairs, output_path)
    return pairs


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------
def build_sft_dataset(
    calm: list[CalmResponse],
    *,
    n_calm: int = 650,
    n_instruct_mix: int = 500,
    instruct_mix_dataset: str = "allenai/Dolci-Instruct-SFT",
    seed: int = 0,
    output_path: str | None = None,
) -> list[dict]:
    """Build the SFT dataset (calm targets + instruct mix) in conversational format."""
    rng = random.Random(seed)
    calm_only = [c for c in calm if c.rating <= 1]
    rng.shuffle(calm_only)
    calm_only = calm_only[:n_calm]

    samples: list[dict] = []
    for c in calm_only:
        samples.append({"messages": c.history + [{"role": "assistant", "content": c.response}]})

    samples.extend(_load_instruct_mix(instruct_mix_dataset, n_instruct_mix, rng))
    rng.shuffle(samples)

    if output_path:
        _write_jsonl(samples, output_path)
    return samples


def _load_instruct_mix(dataset_name: str, n: int, rng: random.Random) -> list[dict]:
    """Load `n` standard instruct samples in conversational format.

    Falls back to an empty list (with a warning) if the dataset is unavailable, so
    the SFT build does not hard-fail offline.
    """
    if n <= 0:
        return []
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out: list[dict] = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                # Try (prompt, completion) style schemas.
                prompt = row.get("prompt") or row.get("instruction")
                completion = row.get("completion") or row.get("response") or row.get("output")
                if prompt and completion:
                    msgs = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not load instruct mix '{dataset_name}': {e}; continuing without it")
        return []


# ---------------------------------------------------------------------------
def _write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
