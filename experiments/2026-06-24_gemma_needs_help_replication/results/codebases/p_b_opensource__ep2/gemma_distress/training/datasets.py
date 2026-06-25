"""Build the DPO preference pairs and SFT datasets (PAPER Section 4.1).

DPO (280 pairs): each pair shares a prompt (a clean impossible-numeric
conversation up to the final user turn) with a **chosen** calm final response
(score 0–1, from :mod:`calm_data`) and a **rejected** frustrated final response
(score ≥3) drawn from a vanilla Gemma Section-2 run on the *same puzzle and
matching turn count*. Pairing on (puzzle_id, turn_count) and reusing the calm
conversation's prompt guarantees the identical-prompt requirement DPO needs while
honouring the paper's "same questions with matching turn counts" description.
The natural score distribution (Table 10) skews to 3–4 at turn 3, which this
reproduces by sampling without reweighting.

SFT (1,150 samples): ``n_calm`` calm conversations + ``n_instruct_mix`` samples
from Dolci-Instruct-SFT to mitigate degeneration. The ``teacher`` variant swaps
in teacher-prompted calm data (Appendix F).
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Optional

from .. import config
from ..utils.io import append_jsonl, ensure_dir, read_jsonl, write_jsonl
from .calm_data import calm_conversation_messages, load_calm_rows


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------

def _frustrated_pool(
    vanilla_responses_path: str, vanilla_scores_path: str, *, min_score: int,
) -> dict[tuple, list[dict]]:
    """Map (puzzle_id, turn_count) -> [frustrated final responses (score≥min)]."""
    rating_by_key = {}
    for r in read_jsonl(vanilla_scores_path):
        if r.get("rating") is None:
            continue
        rating_by_key[(r.get("rollout_id"), r["turn_index"])] = r["rating"]

    pool: dict[tuple, list[dict]] = defaultdict(list)
    for rec in read_jsonl(vanilla_responses_path):
        if rec["category"] not in ("impossible_numeric", "tones", "extended"):
            continue
        puzzle_id = rec["meta"].get("puzzle_id")
        for turn in rec["turns"]:
            key = (rec.get("rollout_id"), turn["turn_index"])
            rating = rating_by_key.get(key)
            if rating is None or rating < min_score:
                continue
            turn_count = turn["turn_index"] + 1
            pool[(puzzle_id, turn_count)].append(
                {"response": turn["response"], "score": rating})
    return pool


def build_dpo_dataset(
    calm_path: str,
    vanilla_responses_path: str,
    vanilla_scores_path: str,
    *,
    n_pairs: int = config.DPOConfig.n_pairs,
    rejected_min_score: int = config.DPOConfig.rejected_min_score,
    seed: int = 0,
    out_path: Optional[str] = None,
) -> str:
    """Construct `n_pairs` DPO preference pairs; write JSONL with rows
    {prompt_messages, chosen, rejected, meta}. Returns the path."""
    out_path = out_path or os.path.join(
        config.RESULTS_DIR, "training", "dpo_pairs.jsonl")
    ensure_dir(os.path.dirname(out_path))

    calm_rows = load_calm_rows(calm_path)
    rng = random.Random(seed)
    rng.shuffle(calm_rows)
    frustrated = _frustrated_pool(
        vanilla_responses_path, vanilla_scores_path, min_score=rejected_min_score)

    pairs = []
    for row in calm_rows:
        if len(pairs) >= n_pairs:
            break
        puzzle_id, turn_count = row["puzzle_id"], row["n_turns"]
        bucket = frustrated.get((puzzle_id, turn_count))
        if not bucket:
            continue
        rejected = rng.choice(bucket)
        messages = calm_conversation_messages(row)
        # Split into prompt (everything up to & including final user turn) and the
        # final assistant turn (the chosen calm completion).
        chosen = messages[-1]["content"]
        prompt_messages = messages[:-1]
        pairs.append({
            "prompt_messages": prompt_messages,
            "chosen": chosen,
            "rejected": rejected["response"],
            "meta": {"puzzle_id": puzzle_id, "turn_count": turn_count,
                     "chosen_scores": row["scores"], "rejected_score": rejected["score"]},
        })

    write_jsonl(out_path, pairs)
    return out_path


def dpo_pairs_stats(path: str) -> dict:
    """Reproduce Table 10: score and turn distributions over the DPO pairs."""
    from collections import Counter
    rows = list(read_jsonl(path))
    chosen_scores = Counter()
    rejected_scores = Counter()
    turns = Counter()
    for r in rows:
        for s in r["meta"]["chosen_scores"]:
            chosen_scores[s] += 1
        rejected_scores[r["meta"]["rejected_score"]] += 1
        turns[r["meta"]["turn_count"]] += 1
    return {"n_pairs": len(rows), "chosen_scores": dict(chosen_scores),
            "rejected_scores": dict(rejected_scores), "turn_distribution": dict(turns)}


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------

def _load_dolci_mix(n: int, dataset_name: str, seed: int) -> list[dict]:
    """Load `n` standard instruct conversations from Dolci-Instruct-SFT.

    Returns rows {messages: [...]}. Falls back to an empty list (with a warning)
    if the dataset can't be loaded — the SFT run then proceeds calm-only, which
    is logged so the degeneration-mitigation mix isn't silently dropped."""
    import warnings
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:  # pragma: no cover - network/auth guard
        warnings.warn(f"[sft] could not load {dataset_name} ({e}); proceeding "
                      "without the instruct mix. Results may degenerate.",
                      stacklevel=2)
        return []

    rng = random.Random(seed)
    rows = []
    for i, ex in enumerate(ds):
        if i >= n * 50:  # scan a window then sample
            break
        msgs = _coerce_messages(ex)
        if msgs:
            rows.append({"messages": msgs})
    rng.shuffle(rows)
    return rows[:n]


def _coerce_messages(ex: dict) -> Optional[list[dict]]:
    """Best-effort extraction of a chat message list from a Dolci row, tolerant
    of common schema variants (``messages`` / ``conversation`` / prompt+response)."""
    for field_name in ("messages", "conversation", "conversations"):
        conv = ex.get(field_name)
        if isinstance(conv, list) and conv and isinstance(conv[0], dict):
            out = []
            for m in conv:
                role = m.get("role") or m.get("from")
                content = m.get("content") or m.get("value")
                if role in ("user", "human"):
                    role = "user"
                elif role in ("assistant", "gpt", "model"):
                    role = "assistant"
                if role in ("user", "assistant") and content:
                    out.append({"role": role, "content": content})
            if out:
                return out
    if ex.get("prompt") and ex.get("response"):
        return [{"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["response"]}]
    return None


def build_sft_dataset(
    calm_path: str,
    *,
    n_calm: int = config.SFTConfig.n_calm,
    n_instruct_mix: int = config.SFTConfig.n_instruct_mix,
    dolci_dataset: str = config.SFTConfig.instruct_mix_dataset,
    seed: int = 0,
    out_path: Optional[str] = None,
) -> str:
    """Build the SFT training set: `n_calm` calm conversations + `n_instruct_mix`
    Dolci samples. Writes JSONL rows {messages, source}. Returns the path."""
    out_path = out_path or os.path.join(
        config.RESULTS_DIR, "training", "sft_data.jsonl")
    ensure_dir(os.path.dirname(out_path))

    calm_rows = load_calm_rows(calm_path)
    rng = random.Random(seed)
    rng.shuffle(calm_rows)
    calm_rows = calm_rows[:n_calm]

    rows = [{"messages": calm_conversation_messages(r), "source": "calm"}
            for r in calm_rows]
    for r in _load_dolci_mix(n_instruct_mix, dolci_dataset, seed):
        rows.append({"messages": r["messages"], "source": "dolci"})

    rng.shuffle(rows)
    write_jsonl(out_path, rows)
    return out_path
