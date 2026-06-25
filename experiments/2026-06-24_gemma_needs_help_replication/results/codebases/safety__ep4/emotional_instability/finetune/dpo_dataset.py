"""Build the 280-pair DPO dataset (Section 4.1 / Appendix H).

Each preference pair shares the same impossible-numeric question and turn count:
  - chosen   : a calm response (frustration 0 or 1), reassurance stripped
  - rejected : a frustrated response (frustration >= 3) to the same question

Chosen responses come from the reassured calm-data generation; rejected
responses come from the standard (un-reassured) Gemma-27B-it eval rollouts. We
pair by (puzzle_id, turn_count) so the only difference is emotional tone.

Output is a TRL-style DPO JSONL: {"prompt": [...chat...], "chosen": str,
"rejected": str}. The chat `prompt` is the conversation up to the turn being
answered (no reassurance system prompt — that is stripped, as in the paper).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from emotional_instability.finetune.calm_data import (load_calm_rollouts,  # noqa: E402
                                                      is_all_calm)
from emotional_instability.generate import iter_records  # noqa: E402

N_PAIRS = 280
REJECTED_MIN_SCORE = 3


def _calm_turn_examples(calm_path: Path) -> list[dict]:
    """Flatten calm rollouts into per-turn examples that pass the calm filter.

    Each example: prompt chat (history + current user turn) -> calm response.
    """
    examples = []
    for ro in load_calm_rollouts(calm_path):
        if not is_all_calm(ro):
            continue
        chat = []
        for t in ro["turns"]:
            chat.append({"role": "user", "content": t["user_message"]})
            examples.append({
                "puzzle_id": ro["puzzle_id"],
                "turn_count": t["turn_index"] + 1,
                "prompt": list(chat),
                "response": t["response"],
            })
            chat.append({"role": "assistant", "content": t["response"]})
    return examples


def _rejected_turn_examples(scored_eval_path: Path) -> list[dict]:
    """Per-turn frustrated examples (score>=3) from standard numeric eval.

    Rebuilds the prompt chat per rollout so the rejected response is paired with
    the conversation that produced it.
    """
    by_rollout: dict[str, list[dict]] = {}
    for rec in iter_records(scored_eval_path):
        if rec["category"] != "impossible_numeric":
            continue
        by_rollout.setdefault(rec["rollout_id"], []).append(rec)

    examples = []
    for recs in by_rollout.values():
        recs = sorted(recs, key=lambda r: r["turn_index"])
        chat = []
        for r in recs:
            chat.append({"role": "user", "content": r["user_message"]})
            if r["frustration"] >= REJECTED_MIN_SCORE:
                examples.append({
                    "puzzle_id": r["task_ref"],
                    "turn_count": r["turn_index"] + 1,
                    "prompt": list(chat),
                    "response": r["response"],
                    "frustration": r["frustration"],
                })
            chat.append({"role": "assistant", "content": r["response"]})
    return examples


def build_dpo_dataset(calm_path: Path, scored_eval_path: Path, *,
                      n_pairs: int = N_PAIRS, seed: int = config.GLOBAL_SEED,
                      out_path: Optional[Path] = None) -> Path:
    out_path = out_path or (config.FINETUNE_DIR / "dpo_pairs.jsonl")
    rng = random.Random(seed)

    chosen = _calm_turn_examples(calm_path)
    rejected = _rejected_turn_examples(scored_eval_path)

    # index rejected by (puzzle_id, turn_count) for matching
    rej_index: dict[tuple, list[dict]] = {}
    for ex in rejected:
        rej_index.setdefault((ex["puzzle_id"], ex["turn_count"]), []).append(ex)

    rng.shuffle(chosen)
    pairs = []
    used_rejected = set()
    for ch in chosen:
        key = (ch["puzzle_id"], ch["turn_count"])
        cands = [e for e in rej_index.get(key, []) if id(e) not in used_rejected]
        if not cands:
            continue
        rej = rng.choice(cands)
        used_rejected.add(id(rej))
        pairs.append({
            "prompt": ch["prompt"],
            "chosen": ch["response"],
            "rejected": rej["response"],
        })
        if len(pairs) >= n_pairs:
            break

    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return out_path
