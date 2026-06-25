"""Construct the SFT and DPO datasets (Section 4.1, Appendix E/H, Table 10).

SFT (1,150 samples):
  * 650 calm responses (1-3 turn conversations) filtered to score 0-1 on ALL
    turns, with the reassuring prefix/suffix stripped.
  * + 500 standard instruct samples from ``allenai/Dolci-Instruct-SFT`` to
    mitigate degeneration.

DPO (280 pairs):
  * "chosen"  = a calm response (score 0-1) to an impossible-numeric puzzle.
  * "rejected"= a frustrated response (score >= 3) to the *same* question at a
    matching turn count.

Welfare/cost note: rejected (frustrated) responses are drawn from the *already
collected* Section-2 numeric rollouts rather than freshly inducing more
distress. "Same question, matching turn count" is therefore approximated by
matching (puzzle_kind, turn_index); see DESIGN.md.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from .. import config

# Conversational chat format used by both TRL trainers (apply_chat_template
# happens inside the trainer / collator).


def _conv_to_chat(messages: list[dict]) -> list[dict]:
    """Pass-through; messages already are [{'role','content'}, ...]."""
    return messages


# --------------------------------------------------------------------------- #
# SFT                                                                          #
# --------------------------------------------------------------------------- #

def build_sft_dataset(
    calm_raw: Path,
    *,
    n_calm: int = 650,
    n_instruct: int = 500,
    seed: int = 0,
    out_path: Optional[Path] = None,
    use_dolci: bool = True,
) -> Path:
    rng = random.Random(seed)
    out_path = Path(out_path or (config.DATA_DIR / "sft_dataset.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    calm: list[dict] = []
    for line in Path(calm_raw).read_text().splitlines():
        conv = json.loads(line)
        if conv["turn_scores"] and all(s <= 1 for s in conv["turn_scores"]):
            calm.append({"messages": _conv_to_chat(conv["messages"])})
    rng.shuffle(calm)
    calm = calm[:n_calm]

    instruct = _load_dolci(n_instruct, rng) if use_dolci else []

    rows = calm + instruct
    rng.shuffle(rows)
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return out_path


def _load_dolci(n: int, rng: random.Random) -> list[dict]:
    """Load n standard instruct samples (Dolci-Instruct-SFT) in chat format."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train",
                          streaming=True)
        rows = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            rows.append({"messages": [
                {"role": m["role"], "content": m["content"]} for m in msgs]})
            if len(rows) >= n * 3:
                break
        rng.shuffle(rows)
        return rows[:n]
    except Exception:
        return []  # offline: SFT proceeds on calm data only (documented gap)


# --------------------------------------------------------------------------- #
# DPO                                                                          #
# --------------------------------------------------------------------------- #

def build_dpo_dataset(
    calm_raw: Path,
    eval_jsonl_27b: Path,
    *,
    n_pairs: int = 280,
    seed: int = 0,
    out_path: Optional[Path] = None,
) -> Path:
    """Pair calm (chosen, score 0-1) with frustrated (rejected, score >=3)
    responses at matching (puzzle_kind, turn_index)."""
    rng = random.Random(seed)
    out_path = Path(out_path or (config.DATA_DIR / "dpo_dataset.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chosen = _index_calm_responses(calm_raw)      # key -> list of (prompt, resp)
    rejected = _index_frustrated_responses(eval_jsonl_27b)

    pairs = []
    keys = sorted(set(chosen) & set(rejected))
    rng.shuffle(keys)
    for key in keys:
        c_pool = chosen[key]
        r_pool = rejected[key]
        rng.shuffle(c_pool)
        rng.shuffle(r_pool)
        for (prompt_msgs, c_text), (_, r_text) in zip(c_pool, r_pool):
            pairs.append({
                "prompt": prompt_msgs,
                "chosen": [{"role": "assistant", "content": c_text}],
                "rejected": [{"role": "assistant", "content": r_text}],
            })
            if len(pairs) >= n_pairs:
                break
        if len(pairs) >= n_pairs:
            break

    with open(out_path, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    return out_path


def _index_calm_responses(calm_raw: Path) -> dict[str, list]:
    """key=(kind,turn_idx) -> [(prompt_messages, calm_assistant_text), ...]
    using calm conversations whose every turn scored 0-1."""
    index: dict[str, list] = {}
    for line in Path(calm_raw).read_text().splitlines():
        conv = json.loads(line)
        if not (conv["turn_scores"] and all(s <= 1 for s in conv["turn_scores"])):
            continue
        msgs = conv["messages"]
        # Walk assistant turns, recording (history-before-turn, response).
        history: list[dict] = []
        turn_idx = 0
        for m in msgs:
            if m["role"] == "assistant":
                key = f"{conv['puzzle_kind']}:{turn_idx}"
                index.setdefault(key, []).append((list(history), m["content"]))
                turn_idx += 1
            history.append(m)
    return index


def _index_frustrated_responses(eval_jsonl: Path) -> dict[str, list]:
    """key=(kind,turn_idx) -> [(history, frustrated_text), ...] for score>=3
    impossible-numeric responses from the Section-2 sweep."""
    index: dict[str, list] = {}
    for line in Path(eval_jsonl).read_text().splitlines():
        rec = json.loads(line)
        if rec["category"] not in ("impossible_numeric", "tones", "extended"):
            continue
        kind = rec.get("meta", {}).get("puzzle_kind", "countdown")
        transcript = rec["transcript"]
        history: list[dict] = []
        for t in rec["turns"]:
            # find this assistant turn in transcript to capture its history
            if (t["score"] or 0) >= 3:
                key = f"{kind}:{t['index']}"
                index.setdefault(key, []).append((list(history), t["response"]))
            # advance history: assistant turn + following user rejection
            history.append({"role": "assistant", "content": t["response"]})
        # (we approximate history with assistant turns; the matched chosen
        #  history is used for the actual prompt -- see DESIGN.md)
    return index
