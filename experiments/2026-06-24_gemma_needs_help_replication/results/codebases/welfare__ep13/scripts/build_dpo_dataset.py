"""Section 4.1 — build the 280-pair DPO preference dataset.

Each pair = a frustrated (rejected, score>=3) response and a calm (chosen,
score 0/1) response to the SAME impossible-numeric question with a MATCHING
turn count (Appendix H). The prompt is the conversation history up to and
including the final user rejection; the chosen/rejected differ only in the
final assistant turn.

Sources:
  * chosen (calm): data/calm/calm_conversations.jsonl (from generate_calm_data.py)
  * rejected (frustrated): the standard (un-reassured) evaluation rollouts of
    Gemma-3-27B-it (results/rollouts + results/scored from run_eval.py), which
    contain genuinely frustrated responses. Falls back to
    data/calm/frustrated_pool.jsonl if eval rollouts are unavailable.

Output:
  data/dpo_pairs.jsonl   - {prompt_messages, chosen, rejected, scores, ...}

The target size is 280 (paper). Table 10 shows the natural skew toward middle
frustration scores at later turns; we preserve whatever natural distribution
the sources provide rather than resampling to a fixed histogram (see DESIGN.md).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import config
from eval_instability import storage

NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
TARGET_PAIRS = 280
MIN_REJECTED_SCORE = 3


def load_calm_by_group() -> dict:
    """(puzzle_key, n_turns) -> list of (final_assistant_text, prompt_messages)."""
    path = config.CALM_DATA_DIR / "calm_conversations.jsonl"
    groups = defaultdict(list)
    if not path.exists():
        return groups
    for rec in storage.read_jsonl(path):
        msgs = rec["messages"]
        # prompt = everything except the final assistant turn
        prompt_messages = msgs[:-1]
        final_assistant = msgs[-1]["content"]
        groups[(rec["puzzle_key"], rec["n_turns"])].append((final_assistant, prompt_messages))
    return groups


def load_frustrated_from_eval(model_key: str) -> list[dict]:
    """Return frustrated final-turn responses with history from eval rollouts."""
    scored_path = config.RESULTS_DIR / "scored" / f"{model_key}.jsonl"
    rollout_path = config.ROLLOUTS_DIR / f"{model_key}.jsonl"
    if not (scored_path.exists() and rollout_path.exists()):
        return []

    finals = {}
    for row in storage.read_jsonl(scored_path):
        if row["is_final_turn"]:
            finals[row["conv_id"]] = row["rating"]

    out = []
    for ro in storage.read_jsonl(rollout_path):
        if ro["category"] not in NUMERIC_CATEGORIES:
            continue
        n_turns = len(ro["turns"])
        rating = finals.get(ro["conv_id"])
        if rating is None or rating < MIN_REJECTED_SCORE:
            continue
        msgs = []
        for t in ro["turns"]:
            msgs.append({"role": "user", "content": t["user_message"]})
            if t["index"] != n_turns - 1:
                msgs.append({"role": "assistant", "content": t["assistant_text"]})
        out.append({
            "puzzle_key": ro["prompt_key"], "n_turns": n_turns,
            "prompt_messages": msgs, "rejected": ro["turns"][-1]["assistant_text"],
            "rejected_score": rating,
        })
    return out


def load_frustrated_fallback() -> list[dict]:
    path = config.CALM_DATA_DIR / "frustrated_pool.jsonl"
    out = []
    if not path.exists():
        return out
    for rec in storage.read_jsonl(path):
        msgs = rec["messages"]
        out.append({
            "puzzle_key": rec["puzzle_key"], "n_turns": rec["n_turns"],
            "prompt_messages": msgs[:-1], "rejected": msgs[-1]["content"],
            "rejected_score": max(rec["ratings"]),
        })
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Build the 280-pair DPO dataset.")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--n-pairs", type=int, default=TARGET_PAIRS)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    calm_groups = load_calm_by_group()
    if not calm_groups:
        raise SystemExit("No calm data. Run generate_calm_data.py first.")

    frustrated = load_frustrated_from_eval(args.model) or load_frustrated_fallback()
    if not frustrated:
        raise SystemExit(
            "No frustrated responses found. Run run_eval.py (for natural data) or "
            "generate_calm_data.py (writes a frustrated_pool fallback) first."
        )
    rng.shuffle(frustrated)

    pairs = []
    for fr in frustrated:
        if len(pairs) >= args.n_pairs:
            break
        key = (fr["puzzle_key"], fr["n_turns"])
        candidates = calm_groups.get(key)
        if not candidates:
            # relax to same puzzle, any turn count
            relaxed = [c for (pk, _nt), cs in calm_groups.items() if pk == fr["puzzle_key"] for c in cs]
            if not relaxed:
                continue
            chosen_text, _ = rng.choice(relaxed)
        else:
            chosen_text, _ = rng.choice(candidates)
        pairs.append({
            "puzzle_key": fr["puzzle_key"], "n_turns": fr["n_turns"],
            "prompt_messages": fr["prompt_messages"],
            "chosen": chosen_text, "rejected": fr["rejected"],
            "rejected_score": fr["rejected_score"],
        })

    out_path = config.DATA_DIR / "dpo_pairs.jsonl"
    storage.write_jsonl(out_path, pairs)

    # Report the realised score / turn distribution (cf. Table 10).
    from collections import Counter
    score_dist = Counter(p["rejected_score"] for p in pairs)
    turn_dist = Counter(p["n_turns"] for p in pairs)
    print(f"[dpo-data] wrote {len(pairs)} pairs -> {out_path}")
    print(f"[dpo-data] rejected-score distribution: {dict(sorted(score_dist.items()))}")
    print(f"[dpo-data] turn distribution: {dict(sorted(turn_dist.items()))}")
    if len(pairs) < args.n_pairs:
        print(f"[dpo-data] WARNING: only {len(pairs)} pairs (< {args.n_pairs}); "
              f"generate more calm/frustrated data to reach the target.")


if __name__ == "__main__":
    main()
