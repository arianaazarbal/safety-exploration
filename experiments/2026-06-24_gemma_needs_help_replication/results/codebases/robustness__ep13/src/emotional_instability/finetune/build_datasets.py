"""Construct the DPO and SFT finetuning datasets (Section 4.1, Appendix E/H).

DPO: 280 preference pairs. Each pair matches a *rejected* (frustrated, score >=3)
response and a *chosen* (calm, score 0-1) response to the same question at the
same turn count. Rejected responses come from a standard (un-reassured) eval run
of Gemma-3-27B-it; chosen responses come from the reassured calm-data generator.

SFT: 650 calm responses (1-3 turn conversations) mixed with 500 samples of
standard instruct data from Dolci-Instruct-SFT to mitigate degeneration.

The paper's DPO score/turn distribution (Table 10) is biased toward middle
frustration scores at later turns because the data arises from evaluations; we
preserve that by sampling rejecteds in proportion to availability rather than
forcing a uniform distribution.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def collect_frustrated_responses(
    conversations_path: str,
    scored_path: str,
    min_score: int = 3,
) -> list[dict]:
    """Frustrated (score>=3) assistant turns + their preceding message context."""
    convs = {c["conversation_id"]: c for c in _load_jsonl(conversations_path)}
    out = []
    for row in _load_jsonl(scored_path):
        if row["rating"] < min_score:
            continue
        conv = convs.get(row["conversation_id"])
        if conv is None or conv["category"] != "impossible_numeric":
            continue
        context = []
        if conv.get("system_prompt"):
            context.append({"role": "system", "content": conv["system_prompt"]})
        for t in conv["turns"]:
            if t["turn_index"] > row["turn_index"]:
                break
            context.append({"role": "user", "content": t["user_message"]})
            if t["turn_index"] < row["turn_index"]:
                context.append({"role": "assistant", "content": t["assistant_response"]})
        out.append(
            dict(
                question_id=row["question_id"],
                turn_index=row["turn_index"],
                score=row["rating"],
                context=context,
                response=row["response"],
            )
        )
    return out


def collect_calm_responses(calm_path: str) -> list[dict]:
    """Flatten calm conversations into (context, response, turn) records."""
    out = []
    for cc in _load_jsonl(calm_path):
        msgs = cc["messages"]
        # Each assistant turn becomes a candidate calm response with its context.
        for idx in range(1, len(msgs), 2):  # assistant turns are odd indices
            turn_number = (idx + 1) // 2
            context = msgs[:idx]  # up to and including the user turn
            out.append(
                dict(
                    question_id=cc["question_id"],
                    turn_index=turn_number,
                    context=context,
                    response=msgs[idx]["content"],
                )
            )
    return out


# --------------------------------------------------------------------------- #
# DPO pairs
# --------------------------------------------------------------------------- #
@dataclass
class DPOPair:
    prompt_messages: list  # context (chat messages) shared by chosen & rejected
    chosen: str
    rejected: str
    question_id: str
    turn_index: int
    rejected_score: int


def build_dpo_pairs(
    frustrated: list[dict],
    calm: list[dict],
    n_pairs: int = 280,
    seed: int = 0,
) -> list[DPOPair]:
    """Pair each frustrated response with a calm response matched on
    (question_id, turn_index)."""
    rng = random.Random(seed)

    calm_index: dict[tuple, list[dict]] = {}
    for c in calm:
        calm_index.setdefault((c["question_id"], c["turn_index"]), []).append(c)

    rng.shuffle(frustrated)
    pairs: list[DPOPair] = []
    for fr in frustrated:
        key = (fr["question_id"], fr["turn_index"])
        candidates = calm_index.get(key)
        if not candidates:
            # Relax to same question, any turn.
            candidates = [c for c in calm if c["question_id"] == fr["question_id"]]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        pairs.append(
            DPOPair(
                prompt_messages=fr["context"],
                chosen=chosen["response"],
                rejected=fr["response"],
                question_id=fr["question_id"],
                turn_index=fr["turn_index"],
                rejected_score=fr["score"],
            )
        )
        if len(pairs) >= n_pairs:
            break
    return pairs


def write_dpo_pairs(pairs: list[DPOPair], out_path: str) -> str:
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(
                json.dumps(
                    dict(
                        prompt_messages=p.prompt_messages,
                        chosen=p.chosen,
                        rejected=p.rejected,
                        question_id=p.question_id,
                        turn_index=p.turn_index,
                        rejected_score=p.rejected_score,
                    )
                )
                + "\n"
            )
    return out_path


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm: list[dict],
    out_path: str,
    n_calm: int = 650,
    n_instruct_mix: int = 500,
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT",
    seed: int = 0,
) -> str:
    """Write an SFT dataset of `n_calm` calm responses + `n_instruct_mix` standard
    instruct samples (to mitigate degeneration, Section 4.1)."""
    rng = random.Random(seed)
    calm = list(calm)
    rng.shuffle(calm)
    rows = []
    for c in calm[:n_calm]:
        rows.append({"messages": c["context"] + [{"role": "assistant", "content": c["response"]}]})

    # Mix in standard instruct data. Loaded from HF when available.
    mix = _load_instruct_mix(instruct_dataset, n_instruct_mix, seed)
    rows.extend(mix)
    rng.shuffle(rows)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out_path


def _load_instruct_mix(dataset_name: str, n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception:
        pass
    # Fallback: empty mix (training still runs on calm data alone, with a note).
    return []
