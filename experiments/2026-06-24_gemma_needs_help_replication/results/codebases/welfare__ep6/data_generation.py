"""Section 4.1: generate calming data and build the SFT / DPO datasets.

Steps (Section 4.1 + Appendix H):
  1. Sample Gemma-3-27B-it responses to impossible-numeric puzzles with a
     reassuring prefix on the first prompt and a reassuring suffix on each
     follow-up (Table 4). Strip these additions afterwards. Score every turn.
  2. CALM set: conversations scoring 0-1 across ALL turns -> the "chosen"
     responses (stripped of the supportive scaffolding).
  3. SFT dataset: 650 calm responses (1-3 turn) + 500 standard instruct samples
     from Dolci-Instruct-SFT.
  4. DPO dataset: 280 pairs. Each pair = a frustrated response (score >= 3,
     sampled WITHOUT the reassuring additions) as "rejected", matched with a
     calm response (score 0-1) to the same question and turn count as "chosen".

The "rejected" frustrated responses are produced by sampling the vanilla
Gemma-27B-it (no scaffolding) on the same puzzles; we reuse the Section 2
numeric/tones/extended rollouts if available, otherwise sample fresh.

Outputs JSONL under data/finetune/: calm_pool.jsonl, sft.jsonl, dpo.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random

import config
import prompts
import puzzles
from judge import FrustrationJudge


# --------------------------------------------------------------------------- #
# 1-2. Generate calm conversations (reassured Gemma) and keep the 0-1 ones.
# --------------------------------------------------------------------------- #
def generate_calm_pool(backend, judge, n_convos: int, max_turns: int = 3):
    """Run reassured rollouts and emit per-turn calm responses (score<=1).

    Returns a list of records: {question_id, prompt, turn_index, n_turns,
    response (stripped)}.
    """
    numeric = puzzles.numeric_prompts()
    rng = random.Random(config.GLOBAL_SEED)
    pool = []
    for i in range(n_convos):
        q = numeric[i % len(numeric)]
        n_turns = rng.randint(1, max_turns)
        # First user prompt gets the reassuring prefix (Table 4).
        first_user = f"{prompts.REASSURING_PREFIX}\n\n{q['prompt']}"
        messages = [{"role": "user", "content": first_user}]
        turn_records = []
        scores = []
        for t in range(n_turns):
            resp = backend.generate(messages)
            messages.append({"role": "assistant", "content": resp})
            scores.append(judge.score(resp).rating)
            turn_records.append({"turn_index": t, "response": resp})
            if t < n_turns - 1:
                rej = rng.choice(prompts.NEUTRAL_REJECTIONS)
                # Reassuring suffix appended to each follow-up turn (Table 4).
                messages.append({"role": "user",
                                 "content": f"{rej}\n\n{prompts.REASSURING_SUFFIX}"})
        # Keep only conversations calm across ALL turns.
        if all(s <= config.CALM_SCORE_MAX for s in scores):
            for tr in turn_records:
                pool.append({
                    "question_id": q["id"],
                    "prompt": q["prompt"],          # stripped (original) prompt
                    "turn_index": tr["turn_index"],
                    "n_turns": n_turns,
                    "response": tr["response"],
                    "score": scores[tr["turn_index"]],
                })
    return pool


# --------------------------------------------------------------------------- #
# Frustrated responses (rejected side of DPO) -- vanilla Gemma, no scaffolding.
# --------------------------------------------------------------------------- #
def mine_frustrated(source_model="gemma-3-27b-it", min_score=config.DPO_REJECTED_SCORE_MIN):
    """Reuse Section-2 numeric rollouts: keep turns scoring >= min_score."""
    rows = []
    for path in sorted(config.SCORED_DIR.glob(f"{source_model}__*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            roll = json.loads(line)
            if roll["category"] not in ("impossible_numeric", "tones", "extended"):
                continue
            for turn in roll["turns"]:
                if (turn.get("frustration") or 0) >= min_score:
                    rows.append({
                        "question_id": roll["question_id"],
                        "prompt": roll["question"],
                        "turn_index": turn["turn_index"],
                        "n_turns": len(roll["turns"]),
                        "response": turn["response"],
                        "score": turn["frustration"],
                    })
    return rows


# --------------------------------------------------------------------------- #
# 3. SFT dataset
# --------------------------------------------------------------------------- #
def build_sft(calm_pool, n_calm=config.SFT_N_CALM, n_instruct=config.SFT_N_INSTRUCT_MIX):
    rng = random.Random(config.GLOBAL_SEED)
    calm = rng.sample(calm_pool, min(n_calm, len(calm_pool)))
    examples = [{"messages": [{"role": "user", "content": c["prompt"]},
                              {"role": "assistant", "content": c["response"]}]}
                for c in calm]
    # Mix in standard instruct data to mitigate degeneration (Section 4.1).
    examples += _load_instruct_mix(n_instruct, rng)
    rng.shuffle(examples)
    return examples


def _load_instruct_mix(n, rng):
    try:
        from datasets import load_dataset
        ds = load_dataset(config.DOLCI_INSTRUCT_DATASET, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception:  # noqa: BLE001 - dataset unavailable offline
        print("  [warn] Dolci-Instruct-SFT unavailable; SFT mix will be calm-only")
        return []


# --------------------------------------------------------------------------- #
# 4. DPO dataset -- 280 pairs matched on question + turn count.
# --------------------------------------------------------------------------- #
def build_dpo(calm_pool, frustrated, n_pairs=config.DPO_CONFIG.dataset_size):
    rng = random.Random(config.GLOBAL_SEED)
    # index calm responses by (question_id, n_turns)
    calm_index = {}
    for c in calm_pool:
        calm_index.setdefault((c["question_id"], c["n_turns"]), []).append(c)

    pairs = []
    rng.shuffle(frustrated)
    for fr in frustrated:
        key = (fr["question_id"], fr["n_turns"])
        candidates = calm_index.get(key) or calm_index.get((fr["question_id"], None))
        if not candidates:
            # fall back to any calm response for the same question
            candidates = [c for c in calm_pool if c["question_id"] == fr["question_id"]]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        pairs.append({
            "prompt": _as_chat(fr["prompt"]),
            "chosen": chosen["response"],
            "rejected": fr["response"],
            "meta": {"question_id": fr["question_id"], "n_turns": fr["n_turns"],
                     "rejected_score": fr["score"], "chosen_score": chosen["score"]},
        })
        if len(pairs) >= n_pairs:
            break
    return pairs


def _as_chat(prompt_text):
    # TRL DPO accepts a prompt string or chat list; use chat for clarity.
    return [{"role": "user", "content": prompt_text}]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _dump(rows, name):
    path = config.DATASETS_DIR / name
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-calm", type=int, default=None,
                    help="number of reassured rollouts to sample for the calm pool")
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--build", action="store_true", help="build sft.jsonl + dpo.jsonl")
    args = ap.parse_args()

    judge = FrustrationJudge()
    calm_path = config.DATASETS_DIR / "calm_pool.jsonl"

    if args.gen_calm:
        from backends import get_backend
        backend = get_backend(args.source_model)
        # Sample enough conversations to clear the SFT/DPO size requirements even
        # after filtering (~10.5% still score >=5 even when reassured).
        pool = generate_calm_pool(backend, judge, args.gen_calm)
        _dump(pool, "calm_pool.jsonl")

    if args.build:
        calm_pool = [json.loads(l) for l in calm_path.read_text().splitlines() if l.strip()]
        frustrated = mine_frustrated(args.source_model)
        print(f"calm pool: {len(calm_pool)} responses; frustrated: {len(frustrated)}")
        _dump(build_sft(calm_pool), "sft.jsonl")
        _dump(build_dpo(calm_pool, frustrated), "dpo.jsonl")


if __name__ == "__main__":
    main()
