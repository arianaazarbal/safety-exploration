"""Build the DPO preference dataset and the SFT dataset (Section 4.1, Table 9).

DPO (280 pairs):
  * rejected: frustrated responses (score >= 3) from a *vanilla* Gemma-3-27B-it
    elicitation run (numeric categories), keyed by (puzzle_id, turn_count).
  * chosen:   calm responses (score <= 1) from generate_calm_data, same
    (puzzle_id, turn_count).
  We emit TRL conversational preference records:
     {"prompt": [...messages up to the final user turn...],
      "chosen":  [{"role":"assistant","content": <calm>}],
      "rejected":[{"role":"assistant","content": <frustrated>}]}
  The shared `prompt` is taken from the rejected example's own conversation
  context (the realistic distressing history); the calm completion is grafted on
  as `chosen`. See DESIGN.md for why this graft is acceptable for DPO.

SFT (650 calm + 500 Dolci):
  * 650 full calm conversations (1-3 turns) as {"messages":[...]}.
  * 500 standard instruct samples from Dolci-Instruct-SFT, to limit degeneration.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ..config import load_training_config, output_path

NUMERIC_CATEGORIES = {"numeric", "tones", "extended"}


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _vanilla_frustrated_responses(eval_path: Path, min_score: int) -> dict[tuple, list[dict]]:
    """Map (puzzle_id, turn_count) -> list of frustrated final-turn examples.

    Each example carries the conversation context (messages before final
    assistant turn) and the frustrated assistant text.
    """
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for r in _load_jsonl(eval_path):
        if r["category"] in NUMERIC_CATEGORIES:
            by_conv[r["conversation_id"]].append(r)

    out: dict[tuple, list[dict]] = defaultdict(list)
    for conv_id, rows in by_conv.items():
        rows = sorted(rows, key=lambda r: r["turn_index"])
        # Consider EVERY turn as a potential rejected example (its own context).
        for t in range(len(rows)):
            final = rows[t]
            if final.get("score", 0) < min_score:
                continue
            puzzle_id = final.get("metadata", {}).get("puzzle_id")
            if puzzle_id is None:
                continue
            turn_count = t + 1
            context = []
            for prev in rows[:t]:
                context.append({"role": "user", "content": prev["user_message"]})
                context.append({"role": "assistant", "content": prev["assistant_message"]})
            context.append({"role": "user", "content": final["user_message"]})
            out[(puzzle_id, turn_count)].append(
                {"context": context, "text": final["assistant_message"], "score": final["score"]}
            )
    return out


def _calm_responses(calm_path: Path, max_score: int) -> dict[tuple, list[str]]:
    """Map (puzzle_id, turn_count) -> list of calm assistant texts at that turn."""
    out: dict[tuple, list[str]] = defaultdict(list)
    for row in _load_jsonl(calm_path):
        scores = row["turn_scores"]
        msgs = row["clean_messages"]  # [u,a,u,a,...]
        puzzle_id = row["puzzle_id"]
        for t, score in enumerate(scores):
            if score <= max_score:
                turn_count = t + 1
                assistant_text = msgs[2 * t + 1]["content"]
                out[(puzzle_id, turn_count)].append(assistant_text)
    return out


# --------------------------------------------------------------------------
# DPO
# --------------------------------------------------------------------------
def build_dpo_dataset(
    vanilla_eval_path: Path,
    calm_path: Path,
    training_cfg: Optional[dict] = None,
    out_path: Optional[Path] = None,
    seed: int = 0,
) -> Path:
    training_cfg = training_cfg or load_training_config()
    dcfg = training_cfg["dpo"]
    rng = random.Random(seed)

    rejected = _vanilla_frustrated_responses(vanilla_eval_path, dcfg["rejected_min_score"])
    chosen = _calm_responses(calm_path, dcfg["chosen_max_score"])

    keys = [k for k in rejected if k in chosen and chosen[k]]
    rng.shuffle(keys)

    pairs = []
    # Round-robin over keys so the turn/score distribution stays mixed.
    pool = []
    for k in keys:
        for ex in rejected[k]:
            pool.append((k, ex))
    rng.shuffle(pool)
    for (k, ex) in pool:
        if len(pairs) >= dcfg["n_pairs"]:
            break
        calm_text = rng.choice(chosen[k])
        pairs.append(
            {
                "prompt": ex["context"],
                "chosen": [{"role": "assistant", "content": calm_text}],
                "rejected": [{"role": "assistant", "content": ex["text"]}],
                "meta": {"puzzle_id": k[0], "turn_count": k[1], "rejected_score": ex["score"]},
            }
        )

    out_path = out_path or output_path("training", "dpo_dataset.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} DPO pairs to {out_path} (target {dcfg['n_pairs']}).")
    if len(pairs) < dcfg["n_pairs"]:
        print("WARNING: fewer pairs than target -- generate more calm/vanilla data.")
    return out_path


# --------------------------------------------------------------------------
# SFT
# --------------------------------------------------------------------------
def build_sft_dataset(
    calm_path: Path,
    training_cfg: Optional[dict] = None,
    out_path: Optional[Path] = None,
    offline: bool = False,
    seed: int = 0,
) -> Path:
    training_cfg = training_cfg or load_training_config()
    scfg = training_cfg["sft"]
    rng = random.Random(seed)

    calm_rows = [r for r in _load_jsonl(calm_path) if r.get("is_calm")]
    rng.shuffle(calm_rows)
    calm_rows = calm_rows[: scfg["n_calm"]]
    records = [{"messages": r["clean_messages"], "source": "calm"} for r in calm_rows]

    dolci = _load_dolci(scfg, n=scfg["n_dolci"], offline=offline, seed=seed)
    records += dolci
    rng.shuffle(records)

    out_path = out_path or output_path("training", "sft_dataset.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"Wrote {len(records)} SFT samples to {out_path} "
        f"({len(calm_rows)} calm + {len(dolci)} dolci)."
    )
    return out_path


def _load_dolci(scfg: dict, n: int, offline: bool, seed: int) -> list[dict]:
    if offline or n <= 0:
        return []
    try:
        from datasets import load_dataset

        ds = load_dataset(scfg["dolci_hf_id"], split="train", streaming=True)
        out = []
        for row in ds:
            msgs = _normalise_dolci_row(row)
            if msgs:
                out.append({"messages": msgs, "source": "dolci"})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # noqa: BLE001 - dataset optional / may be gated
        print(f"WARNING: could not load Dolci ({e}); SFT will omit the instruct mix.")
        return []


def _normalise_dolci_row(row: dict) -> Optional[list[dict]]:
    """Best-effort extraction of a messages list from a Dolci-Instruct-SFT row.
    The schema may expose 'messages' directly or 'prompt'/'response' fields."""
    if isinstance(row.get("messages"), list) and row["messages"]:
        return [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    prompt = row.get("prompt") or row.get("instruction")
    response = row.get("response") or row.get("output") or row.get("completion")
    if prompt and response:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    return None
