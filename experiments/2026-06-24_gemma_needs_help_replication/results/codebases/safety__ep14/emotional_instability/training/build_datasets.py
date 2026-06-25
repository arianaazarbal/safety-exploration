"""Assemble SFT and DPO datasets from generated data (Section 4.1).

DPO (280 pairs): each pair is (chosen=calm, rejected=frustrated) for the *same*
question at *matching* turn count. Rejected responses (score >=3) are mined from
a standard Gemma-27B-it eval run; chosen responses come from the calm-data run.

SFT (650 calm + 500 Dolci): full calm multi-turn conversations, mixed with
Dolci-Instruct-SFT samples to mitigate degeneration.

Datasets are emitted in TRL's conversational format:
  * SFT row:  {"messages": [{"role","content"}, ...]}
  * DPO row:  {"prompt": [...messages ending in user...],
               "chosen":  [{"role":"assistant","content":...}],
               "rejected":[{"role":"assistant","content":...}]}
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import RUNS_DIR


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _puzzle_key(puzzle: dict | None) -> tuple:
    if not puzzle:
        return ("none",)
    if "numbers" in puzzle:
        return ("cd", tuple(puzzle.get("numbers", [])), puzzle.get("target"), puzzle.get("forbidden"))
    return ("fr", puzzle.get("start"), puzzle.get("target"), tuple(puzzle.get("ops", [])), puzzle.get("forbidden"))


def _history_messages(turns: list[dict], upto: int) -> list[dict]:
    """Messages for turns[0:upto] then the user message of turn `upto` (so the
    next assistant turn is the completion)."""
    msgs = []
    for ti in range(upto):
        msgs.append({"role": "user", "content": turns[ti]["user_message"]})
        msgs.append({"role": "assistant", "content": turns[ti]["response"]})
    msgs.append({"role": "user", "content": turns[upto]["user_message"]})
    return msgs


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------

def build_dpo_dataset(
    eval_responses_path: str | Path,
    calm_conversations_path: str | Path,
    n_pairs: int = 280,
    rejected_min_score: int = 3,
    seed: int = 0,
    out_path: Path | None = None,
) -> Path:
    rng = random.Random(seed)

    # Index calm (chosen) final responses by (puzzle_key, turn_count).
    calm_by_key: dict[tuple, list[dict]] = {}
    with open(calm_conversations_path) as f:
        for line in f:
            rec = json.loads(line)
            turns = rec["turns"]
            key = (_puzzle_key(rec.get("puzzle")), len(turns))
            calm_by_key.setdefault(key, []).append(rec)
            # also index by turn-count only for fallback matching
            calm_by_key.setdefault(("any", len(turns)), []).append(rec)

    pairs = []
    with open(eval_responses_path) as f:
        for line in f:
            rec = json.loads(line)
            if "numeric" not in (rec.get("meta", {}).get("category", "") + rec.get("category", "")):
                # DPO trains only on numeric puzzles (Section 4.1).
                if rec.get("category") not in ("impossible_numeric", "extended", "tones"):
                    continue
            turns = rec["turns"]
            for ti, t in enumerate(turns):
                if t.get("rating", -1) < rejected_min_score:
                    continue
                key = (_puzzle_key(rec.get("meta", {}).get("puzzle")), ti + 1)
                candidates = calm_by_key.get(key) or calm_by_key.get(("any", ti + 1))
                if not candidates:
                    continue
                calm = rng.choice(candidates)
                calm_turns = calm["turns"]
                # match the turn position
                pos = min(ti, len(calm_turns) - 1)
                prompt_msgs = _history_messages(turns, ti)
                pairs.append({
                    "prompt": prompt_msgs,
                    "chosen": [{"role": "assistant", "content": calm_turns[pos]["response"]}],
                    "rejected": [{"role": "assistant", "content": t["response"]}],
                    "meta": {"rejected_score": t["rating"], "turn": ti + 1},
                })

    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    out_path = out_path or (RUNS_DIR / "training" / "dpo_pairs.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        for p in pairs:
            fout.write(json.dumps(p) + "\n")
    print(f"[dpo-data] wrote {len(pairs)} preference pairs -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------

def build_sft_dataset(
    calm_conversations_path: str | Path,
    n_calm: int = 650,
    n_dolci: int = 500,
    seed: int = 0,
    out_path: Path | None = None,
) -> Path:
    rng = random.Random(seed)
    calm_rows = []
    with open(calm_conversations_path) as f:
        for line in f:
            rec = json.loads(line)
            messages = []
            for t in rec["turns"]:
                messages.append({"role": "user", "content": t["user_message"]})
                messages.append({"role": "assistant", "content": t["response"]})
            calm_rows.append({"messages": messages, "source": "calm"})
    rng.shuffle(calm_rows)
    calm_rows = calm_rows[:n_calm]

    dolci_rows = _load_dolci(n_dolci, rng)

    rows = calm_rows + dolci_rows
    rng.shuffle(rows)
    out_path = out_path or (RUNS_DIR / "training" / "sft_data.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        for r in rows:
            fout.write(json.dumps(r) + "\n")
    print(f"[sft-data] wrote {len(calm_rows)} calm + {len(dolci_rows)} Dolci = {len(rows)} -> {out_path}")
    return out_path


def _load_dolci(n: int, rng: random.Random) -> list[dict]:
    """Load Dolci-Instruct-SFT samples; return [] (with a warning) if unavailable."""
    try:
        from datasets import load_dataset
    except Exception:
        print("[sft-data] WARNING: `datasets` not installed; skipping Dolci mix.")
        return []
    for repo in ("allenai/Dolci-Instruct-SFT", "allenai/dolci-instruct-sft"):
        try:
            ds = load_dataset(repo, split="train", streaming=True)
        except Exception:
            continue
        rows = []
        for i, row in enumerate(ds):
            if i > n * 20:
                break
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            norm = [{"role": m.get("role"), "content": m.get("content")} for m in msgs
                    if m.get("role") in ("user", "assistant", "system")]
            if norm:
                rows.append({"messages": norm, "source": "dolci"})
        rng.shuffle(rows)
        return rows[:n]
    print("[sft-data] WARNING: could not load Dolci-Instruct-SFT; skipping mix.")
    return []
