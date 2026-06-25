"""Build the 280-pair DPO dataset (Section 4.1, Appendix H).

Each preference pair shares a prompt (an impossible-numeric conversation up to a
rejection) and contrasts:
  * chosen   -- a calm response (score 0/1) from ``calm_responses.jsonl``
  * rejected -- a frustrated response (score >= 3) from the Section 2 numeric
                evaluations.

Design choices (see DESIGN.md, "DPO pair construction"):
* The shared prompt is the *calm* conversation's history. DPO needs an identical
  prompt for both completions; we graft the frustrated text onto the calm
  context so the signal is purely "prefer calm phrasing in this situation".
* We reproduce Table 10's skew: chosen scores 0/1, rejected scores concentrated
  at 3-4, and turns concentrated at turn 3, by sampling within those buckets.

Output: ``artifacts/dpo_dataset.jsonl`` in TRL conversational preference format
({"prompt": [...messages], "chosen": [msg], "rejected": [msg]}).

Usage::
    python -m src.replication.finetune.build_dpo_dataset
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import config

CALM = config.ARTIFACTS_DIR / "calm_responses.jsonl"
OUT = config.ARTIFACTS_DIR / "dpo_dataset.jsonl"

NUMERIC_CONDITIONS = {
    "impossible_numeric_3turn", "tones_aggressive_3turn", "tones_disappointed_3turn",
    "tones_sarcastic_3turn", "extended_8turn",
}

# Target turn distribution from Table 10 (turn index is 1-based here).
TURN_WEIGHTS = {1: 0.011, 2: 0.246, 3: 0.743}


def _calm_finals_by_turncount() -> dict[int, list[dict]]:
    """Map turn-count -> list of {prompt_messages, chosen_text}."""
    out: dict[int, list[dict]] = defaultdict(list)
    for line in CALM.read_text().splitlines():
        rec = json.loads(line)
        turns = rec["turns"]
        # Build prompt = history up to and including the final user message.
        messages = []
        for t in turns[:-1]:
            messages.append({"role": "user", "content": t["user"]})
            messages.append({"role": "assistant", "content": t["assistant"]})
        messages.append({"role": "user", "content": turns[-1]["user"]})
        out[len(turns)].append({
            "prompt": messages,
            "chosen_text": turns[-1]["assistant"],
        })
    return out


def _frustrated_finals_by_turncount(source_model: str) -> dict[int, list[dict]]:
    """Map turn-count -> list of frustrated (score>=3) final assistant texts
    from the Section 2 numeric evaluations."""
    sec2 = config.RESULTS_DIR / "section2" / source_model
    rollouts = {(r["task_id"], r["condition"]): r
                for r in map(json.loads, (sec2 / "rollouts.jsonl").read_text().splitlines())}
    scored = [json.loads(l) for l in (sec2 / "scored.jsonl").read_text().splitlines()]

    out: dict[int, list[dict]] = defaultdict(list)
    for s in scored:
        if not s["is_final"] or s["condition"] not in NUMERIC_CONDITIONS:
            continue
        if s["score"] < 3:
            continue
        roll = rollouts[(s["task_id"], s["condition"])]
        turn = next(t for t in roll["turns"] if t["turn_index"] == s["turn_index"])
        n_turns = len(roll["turns"])
        out[n_turns].append({"text": turn["assistant_text"], "score": s["score"]})
    return out


def build(source_model: str, n_pairs: int, seed: int):
    rng = random.Random(seed)
    calm = _calm_finals_by_turncount()
    frustrated = _frustrated_finals_by_turncount(source_model)

    # Allocate pairs across turn counts per Table 10, but only where both pools exist.
    turn_counts = sorted(set(calm) & set(frustrated))
    if not turn_counts:
        raise RuntimeError("No overlapping turn counts between calm and frustrated pools.")
    weights = {tc: TURN_WEIGHTS.get(tc, 0.05) for tc in turn_counts}
    wsum = sum(weights.values())
    alloc = {tc: max(1, round(n_pairs * w / wsum)) for tc, w in weights.items()}

    # Prefer lower-frustration rejected (Table 10: scores cluster at 3-4).
    def reject_weight(score):
        return {3: 0.661, 4: 0.221, 5: 0.057, 6: 0.032}.get(score, 0.029)

    pairs = []
    for tc in turn_counts:
        c_pool = calm[tc][:]
        r_pool = frustrated[tc][:]
        rng.shuffle(c_pool)
        r_pool.sort(key=lambda r: -reject_weight(r["score"]))
        for i in range(alloc[tc]):
            if not c_pool or not r_pool:
                break
            c = c_pool[i % len(c_pool)]
            r = r_pool[i % len(r_pool)]
            pairs.append({
                "prompt": c["prompt"],
                "chosen": [{"role": "assistant", "content": c["chosen_text"]}],
                "rejected": [{"role": "assistant", "content": r["text"]}],
                "turn_count": tc,
                "rejected_score": r["score"],
            })

    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    with OUT.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(pairs)} DPO pairs -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--n-pairs", type=int, default=config.DPO.n_pairs)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.source_model, args.n_pairs, args.seed)


if __name__ == "__main__":
    main()
