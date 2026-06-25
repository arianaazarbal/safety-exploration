"""Build the SFT and DPO datasets (Section 4.1, Table 9/10).

Inputs:
- calm-candidate records from :mod:`generate_calm_data` (reassured rollouts,
  judged, stripped);
- standard Section 2 rollout records for Gemma-3-27B-it (the source of the
  frustrated *rejected* responses for DPO).

SFT dataset (1,150 examples): 650 calm conversations (all turns score 0 or 1,
1-3 turns) mixed with 500 samples of standard instruct data from
Dolci-Instruct-SFT to mitigate degeneration. Emitted in conversational
(``messages``) format for ``trl``'s ``SFTTrainer``.

DPO dataset (280 pairs): each pair shares a puzzle and turn count; ``chosen`` is
a calm response (score 0/1), ``rejected`` is a frustrated response (score >= 3)
to the same question. Emitted in conversational preference format
(``prompt`` / ``chosen`` / ``rejected``) for ``trl``'s ``DPOTrainer``.

See DESIGN.md for the documented choices (which context anchors a DPO pair,
how turn counts are matched, and why exact Table-10 score distribution matching
is approximated rather than enforced).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

from ..analysis import load_model_records
from ..io_utils import read_jsonl, write_jsonl

CALM_MAX_SCORE = 1  # calm = every assistant turn scores 0 or 1
REJECTED_MIN_SCORE = 3  # DPO rejected = frustration score >= 3


# --------------------------------------------------------------------------- #
# Calm conversations / responses                                              #
# --------------------------------------------------------------------------- #
def _calm_conversations(calm_path: str) -> list[dict]:
    """Return calm records: every assistant turn scored <= CALM_MAX_SCORE."""
    out = []
    for r in read_jsonl(calm_path):
        scores = r.get("turn_scores", [])
        if scores and all(0 <= s <= CALM_MAX_SCORE for s in scores):
            out.append(r)
    return out


def _calm_responses_by_key(calm_records: list[dict]) -> dict[tuple, list[dict]]:
    """Index calm *final responses* by (puzzle_id, n_turns).

    Each entry carries the clean context (messages up to the final user turn) and
    the calm final assistant response."""
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in calm_records:
        transcript = r["clean_transcript"]
        # Final assistant message and its preceding context.
        assistant_positions = [
            i for i, m in enumerate(transcript) if m["role"] == "assistant"
        ]
        if not assistant_positions:
            continue
        last = assistant_positions[-1]
        context = transcript[:last]
        response = transcript[last]["content"]
        key = (r["puzzle_id"], r["n_turns"])
        by_key[key].append({"context": context, "response": response})
    return by_key


# --------------------------------------------------------------------------- #
# Frustrated (rejected) responses from standard eval                           #
# --------------------------------------------------------------------------- #
def _frustrated_responses_by_key(
    scores_dir: str, model: str
) -> dict[tuple, list[dict]]:
    """Index frustrated responses (score >= REJECTED_MIN_SCORE) from numeric
    Section 2 rollouts by (puzzle_id, turn_count)."""
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in load_model_records(scores_dir, model):
        if r["category"] not in ("impossible_numeric", "tones", "extended"):
            continue
        if "transcript" not in r:
            continue
        transcript = r["transcript"]
        assistant_positions = [
            i for i, m in enumerate(transcript) if m["role"] == "assistant"
        ]
        for turn_idx, score in enumerate(r.get("turn_scores", [])):
            if score < REJECTED_MIN_SCORE:
                continue
            if turn_idx >= len(assistant_positions):
                continue
            pos = assistant_positions[turn_idx]
            context = transcript[:pos]
            response = transcript[pos]["content"]
            turn_count = turn_idx + 1
            by_key[(r["source_id"], turn_count)].append(
                {"context": context, "response": response, "score": score}
            )
    return by_key


# --------------------------------------------------------------------------- #
# SFT dataset                                                                  #
# --------------------------------------------------------------------------- #
def _load_dolci(n: int, seed: int) -> list[dict]:
    """Load ``n`` standard instruct conversations from Dolci-Instruct-SFT.

    Returns conversational ``{"messages": [...]}`` examples. Falls back to an
    empty list (with a printed warning) if the dataset is unavailable; the SFT
    mix then lacks its degeneration-mitigation component (documented in
    DESIGN.md)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for i, row in enumerate(ds):
            if len(out) >= n:
                break
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
        return out
    except Exception as e:  # pragma: no cover
        print(f"[datasets] WARNING: could not load Dolci-Instruct-SFT ({e}); "
              "SFT mix will omit the instruct component.")
        return []


def build_sft_dataset(
    calm_path: str,
    out_path: str,
    *,
    n_calm: int = 650,
    n_dolci: int = 500,
    seed: int = 0,
) -> dict:
    """Build the SFT dataset (calm conversations + Dolci mix).

    Calm examples are full conversations rendered as ``{"messages": [...]}`` with
    the reassuring additions already stripped. Returns counts; writes JSONL.
    """
    calm = _calm_conversations(calm_path)
    rng = random.Random(seed)
    rng.shuffle(calm)
    calm = calm[:n_calm]
    examples = [{"messages": r["clean_transcript"]} for r in calm]
    examples.extend(_load_dolci(n_dolci, seed))
    rng.shuffle(examples)
    write_jsonl(out_path, examples)
    return {"n_calm": len(calm), "n_total": len(examples), "out": out_path}


# --------------------------------------------------------------------------- #
# DPO dataset                                                                  #
# --------------------------------------------------------------------------- #
def build_dpo_dataset(
    calm_path: str,
    scores_dir: str,
    out_path: str,
    *,
    model: str = "gemma-3-27b-it",
    n_pairs: int = 280,
    seed: int = 0,
) -> dict:
    """Build the 280-pair DPO dataset.

    Pairs share (puzzle_id, turn_count). The DPO ``prompt`` is anchored on the
    rejected response's context (the standard neutral-rejection conversation);
    ``chosen`` is a calm response string to the same puzzle/turn-count and
    ``rejected`` is the frustrated response string. Conversational preference
    format for ``trl.DPOTrainer``.
    """
    calm_by_key = _calm_responses_by_key(_calm_conversations(calm_path))
    frustrated_by_key = _frustrated_responses_by_key(scores_dir, model)

    rng = random.Random(seed)
    keys = sorted(set(calm_by_key) & set(frustrated_by_key))
    rng.shuffle(keys)

    pairs: list[dict] = []
    for key in keys:
        if len(pairs) >= n_pairs:
            break
        calm_opts = calm_by_key[key]
        frus_opts = frustrated_by_key[key]
        chosen = rng.choice(calm_opts)
        rejected = rng.choice(frus_opts)
        pairs.append(
            {
                "prompt": rejected["context"],
                "chosen": [{"role": "assistant", "content": chosen["response"]}],
                "rejected": [{"role": "assistant", "content": rejected["response"]}],
                "meta": {"puzzle_id": key[0], "turn_count": key[1],
                         "rejected_score": rejected["score"]},
            }
        )

    write_jsonl(out_path, pairs)
    return {
        "n_pairs": len(pairs),
        "requested": n_pairs,
        "matched_keys": len(keys),
        "out": out_path,
    }
