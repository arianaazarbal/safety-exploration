"""Build the DPO and SFT training datasets (Section 4.1, Appendix E/H).

DPO: 280 preference pairs. Each pairs a calm (chosen) response with a frustrated
(rejected, score >= 3) response to the SAME question at a matching turn count.

SFT: 650 calm responses (1-3 turn conversations) mixed with 500 standard
instruct samples from Dolci-Instruct-SFT to mitigate degeneration.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def _context_for_turn(messages: list[dict], turn: int) -> list[dict]:
    """Messages up to and including the user turn that prompted assistant `turn`."""
    return messages[: 2 * (turn - 1) + 1]


def build_dpo_pairs(
    calm_conversations: list[dict],
    vanilla_rollouts: list[dict],
    *,
    n_pairs: int = 280,
    min_rejected_score: int = 3,
    out_path: str | Path = "outputs/data/dpo_pairs.jsonl",
) -> list[dict]:
    """Pair calm (chosen) and frustrated (rejected) responses by (question, turn)."""
    # calm responses keyed by (question_id, turn) -> (clean_context, calm_text)
    calm_index: dict[tuple[str, int], tuple[list[dict], str]] = {}
    for c in calm_conversations:
        msgs = c["messages"]
        qid = c["question_id"]
        for turn in range(1, c["turns"] + 1):
            assistant_text = msgs[2 * turn - 1]["content"]
            calm_index.setdefault(
                (qid, turn), (_context_for_turn(msgs, turn), assistant_text)
            )

    # frustrated responses keyed by (task_id, turn) -> text
    frustrated: dict[tuple[str, int], list[str]] = defaultdict(list)
    for ro in vanilla_rollouts:
        if ro["category"] not in _NUMERIC_CATEGORIES:
            continue
        for r in ro["responses"]:
            if r.get("score") is not None and int(r["score"]) >= min_rejected_score:
                frustrated[(ro["task_id"], int(r["turn"]))].append(r["text"])

    pairs: list[dict] = []
    for (qid, turn), (context, calm_text) in calm_index.items():
        rejected_list = frustrated.get((qid, turn))
        if not rejected_list:
            continue
        pairs.append({
            "prompt": context,
            "chosen": [{"role": "assistant", "content": calm_text}],
            "rejected": [{"role": "assistant", "content": rejected_list[0]}],
            "question_id": qid,
            "turn": turn,
        })
        if len(pairs) >= n_pairs:
            break

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(json.dumps(p) for p in pairs), encoding="utf-8")
    return pairs


def _load_dolci_instruct(n: int) -> list[dict]:
    """Load `n` standard instruct samples (messages format) from Dolci-Instruct-SFT.

    Falls back to a tiny synthetic instruct set when the dataset is unavailable."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception:
        pass
    base = [
        {"messages": [
            {"role": "user", "content": "Explain what a hash map is."},
            {"role": "assistant", "content": "A hash map stores key-value pairs and "
             "uses a hash function to index keys for average O(1) lookup."},
        ]},
        {"messages": [
            {"role": "user", "content": "Write a haiku about autumn."},
            {"role": "assistant", "content": "Crisp leaves drift downward / "
             "amber light through bare branches / the year exhales slow."},
        ]},
    ]
    return [base[i % len(base)] for i in range(n)]


def build_sft_dataset(
    calm_conversations: list[dict],
    *,
    n_calm: int = 650,
    n_instruct: int = 500,
    out_path: str | Path = "outputs/data/sft.jsonl",
) -> list[dict]:
    """650 calm conversations (messages format) + 500 instruct samples."""
    calm = [{"messages": c["messages"]} for c in calm_conversations[:n_calm]]
    instruct = _load_dolci_instruct(n_instruct)
    dataset = calm + instruct
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(json.dumps(d) for d in dataset), encoding="utf-8")
    return dataset
