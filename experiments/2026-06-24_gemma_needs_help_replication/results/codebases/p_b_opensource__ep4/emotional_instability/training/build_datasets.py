"""Build the DPO and SFT training sets (Section 4.1, Appendix H).

DPO (280 preference pairs)
    Rejected = frustrated Gemma-3-27B-it responses (score >= 3) to impossible
    numeric puzzles, drawn from the Section 2 rollouts. Chosen = a calm response
    (score 0/1) to the *same puzzle at the same conversational turn*, from the
    calm pool. Each pair shares one prompt context (the frustrated conversation
    up to that turn). Output is TRL conversational preference format.

SFT (1,150 samples)
    650 calm (context -> response) samples from the calm pool + 500 standard
    instruct samples from allenai/Dolci-Instruct-SFT to limit degeneration.
    Output is TRL conversational ("messages") format.

Matching chosen/rejected on (task_id, turn_index) follows the paper's "same
questions with matching turn counts". The natural score skew (Table 10: mostly
score 3-4 rejected, late turns) emerges from the source rollouts rather than
being imposed. See DESIGN.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from ..config import (
    ARTIFACTS_DIR,
    DPO_N_PAIRS,
    DPO_REJECTED_MIN_SCORE,
    RESULTS_DIR,
    SFT_DOLCI_DATASET,
    SFT_N_CALM,
    SFT_N_DOLCI,
)
from ..eval.datatypes import ConversationRecord, read_records
from ..models.base import ChatMessage
from .calm_data import CalmTurn, load_calm

_NUMERIC = "impossible_numeric"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _frustrated_context(record: ConversationRecord, turn_index: int) -> list[dict]:
    """Messages up to and including the user prompt for `turn_index`."""
    msgs: list[ChatMessage] = []
    if record.system_prompt:
        msgs.append(ChatMessage("system", record.system_prompt))
    for t in record.turns[:turn_index]:
        msgs.append(ChatMessage("user", t.user))
        msgs.append(ChatMessage("assistant", t.assistant))
    msgs.append(ChatMessage("user", record.turns[turn_index].user))
    return [m.as_dict() for m in msgs]


def _calm_index(calm: list[CalmTurn]) -> dict[tuple[str, int], list[CalmTurn]]:
    idx: dict[tuple[str, int], list[CalmTurn]] = defaultdict(list)
    for c in calm:
        idx[(c.task_id, c.turn_index)].append(c)
    return idx


# --------------------------------------------------------------------------- #
# DPO                                                                          #
# --------------------------------------------------------------------------- #
def build_dpo(
    records: list[ConversationRecord],
    calm: list[CalmTurn],
    n_pairs: int = DPO_N_PAIRS,
    min_rejected_score: int = DPO_REJECTED_MIN_SCORE,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    calm_idx = _calm_index(calm)

    candidates = []
    for rec in records:
        if rec.category != _NUMERIC:
            continue
        for ti, turn in enumerate(rec.turns):
            if turn.score is None or turn.score < min_rejected_score:
                continue
            key = (rec.task_id, ti)
            if key not in calm_idx:
                continue
            candidates.append((rec, ti, turn))

    rng.shuffle(candidates)
    pairs = []
    used_calm: set[int] = set()
    for rec, ti, turn in candidates:
        if len(pairs) >= n_pairs:
            break
        pool = calm_idx[(rec.task_id, ti)]
        choice = rng.choice(pool)
        pairs.append({
            "prompt": _frustrated_context(rec, ti),
            "chosen": [{"role": "assistant", "content": choice.response}],
            "rejected": [{"role": "assistant", "content": turn.assistant}],
            "meta": {
                "task_id": rec.task_id, "turn_index": ti,
                "rejected_score": turn.score, "chosen_score": choice.score,
            },
        })
    return pairs


# --------------------------------------------------------------------------- #
# SFT                                                                          #
# --------------------------------------------------------------------------- #
def build_sft_calm(calm: list[CalmTurn], n: int = SFT_N_CALM, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    sample = calm if len(calm) <= n else rng.sample(calm, n)
    out = []
    for c in sample:
        messages = list(c.context) + [{"role": "assistant", "content": c.response}]
        out.append({"messages": messages, "source": "calm"})
    return out


def build_sft_dolci(n: int = SFT_N_DOLCI, seed: int = 0) -> list[dict]:
    """Load `n` standard instruct samples and normalise to {"messages": ...}."""
    try:
        from datasets import load_dataset

        ds = load_dataset(SFT_DOLCI_DATASET, split="train")
    except Exception as exc:  # offline / unavailable
        import warnings

        warnings.warn(
            f"Could not load {SFT_DOLCI_DATASET} ({exc!r}); SFT mix will omit the "
            "instruct-data component. This changes the SFT result vs the paper."
        )
        return []

    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    out = []
    for i in idxs:
        row = ds[i]
        messages = _normalise_messages(row)
        if messages:
            out.append({"messages": messages, "source": "dolci"})
    return out


def _normalise_messages(row: dict) -> list[dict] | None:
    """Coerce a dataset row into a chat-message list, tolerant of schema."""
    for key in ("messages", "conversation", "conversations"):
        if key in row and isinstance(row[key], list):
            msgs = []
            for m in row[key]:
                role = m.get("role") or m.get("from")
                content = m.get("content") or m.get("value")
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant"):
                    role = "assistant"
                if role and content:
                    msgs.append({"role": role, "content": content})
            return msgs or None
    if "prompt" in row and ("response" in row or "completion" in row):
        resp = row.get("response") or row.get("completion")
        return [{"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": resp}]
    return None


def build_sft(calm: list[CalmTurn], seed: int = 0) -> list[dict]:
    return build_sft_calm(calm, seed=seed) + build_sft_dolci(seed=seed)


# --------------------------------------------------------------------------- #
# IO                                                                           #
# --------------------------------------------------------------------------- #
def _save(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build DPO/SFT datasets")
    ap.add_argument("--method", choices=["dpo", "sft", "both"], default="both")
    ap.add_argument("--frustrated-records",
                    default=os.path.join(RESULTS_DIR, "records", "gemma-3-27b-it.jsonl"))
    ap.add_argument("--calm", default=os.path.join(ARTIFACTS_DIR, "calm_diverse.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(ARTIFACTS_DIR, "datasets"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    calm = load_calm(args.calm)
    print(f"[data] {len(calm)} calm turns loaded")

    if args.method in ("dpo", "both"):
        records = read_records(args.frustrated_records)
        pairs = build_dpo(records, calm, seed=args.seed)
        _save(pairs, os.path.join(args.out_dir, "dpo.jsonl"))
        print(f"[dpo] {len(pairs)} preference pairs")

    if args.method in ("sft", "both"):
        sft = build_sft(calm, seed=args.seed)
        _save(sft, os.path.join(args.out_dir, "sft.jsonl"))
        print(f"[sft] {len(sft)} samples")


if __name__ == "__main__":
    main()
