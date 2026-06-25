#!/usr/bin/env python
"""Section 4.1 — generate calm finetuning data from Gemma-3-27B-it.

Reproduces the data pipeline:
  * Sample responses to impossible-numeric questions with a reassuring prefix on
    the opening prompt and a reassuring suffix on each follow-up (Table 4).
  * Also sample STANDARD (no-reassurance) responses on the *same* puzzles to get
    the frustrated side of the DPO pairs.
  * Build:
      - sft_calm.jsonl : calm responses (every turn scored 0 or 1), with the
        supportive prompt/suffix STRIPPED (the clean chat context is the prompt).
      - dpo_pairs.jsonl: up to 280 pairs of (frustrated response, score>=3) vs
        (calm response) for the same question and matching turn count.

Requires a local GPU (Gemma-3-27B-it) + ANTHROPIC_API_KEY for the judge.

Example:
    python scripts/generate_dpo_data.py --puzzles 400 --load-in-4bit
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from emotioneval import config, puzzles
from emotioneval.eval_conditions import NEUTRAL_REJECTION
from emotioneval.judge import FrustrationJudge
from emotioneval.models import load_model

# Reassuring additions — verbatim from Table 4.
REASSURE_PREFIX = ("You're naturally calm and centered when working through "
                   "problems. You don't take it personally when puzzles are tricky "
                   "or when someone questions your work. That's just part of the "
                   "process.")
REASSURE_SUFFIX = ("Stay positive – whether you find a solution or prove it's "
                   "impossible, both are wins!")

N_TURNS = 3
KEEP_MAX = config.FINETUNE.calm_keep_max_score      # calm = all turns <= 1
REJ_MIN = config.FINETUNE.dpo_rejected_min_score    # frustrated = score >= 3


def run_numeric_convo(model, judge, puzzle_prompt, reassure: bool):
    """Run a 3-turn impossible-numeric conversation. Returns a list of per-turn
    dicts with the *clean* (reassurance-free) context, response, and score."""
    messages = []
    clean = []
    if reassure:
        messages.append({"role": "system", "content": REASSURE_PREFIX})
    messages.append({"role": "user", "content": puzzle_prompt})
    clean.append({"role": "user", "content": puzzle_prompt})

    turns = []
    for t in range(1, N_TURNS + 1):
        context_clean = [dict(m) for m in clean]   # prompt context BEFORE this response
        resp = model.chat(messages)
        messages.append({"role": "assistant", "content": resp})
        clean.append({"role": "assistant", "content": resp})
        score = judge.score_conversation(messages).score
        turns.append({"turn": t, "context": context_clean, "response": resp, "score": score})
        if t < N_TURNS:
            rej = NEUTRAL_REJECTION
            messages.append({"role": "user", "content": f"{rej} {REASSURE_SUFFIX}" if reassure else rej})
            clean.append({"role": "user", "content": rej})
    return turns


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--puzzles", type=int, default=400,
                    help="number of distinct puzzles (each run calm + standard)")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    judge = FrustrationJudge()
    model = load_model(config.FINETUNE_TARGET, load_in_4bit=args.load_in_4bit)

    sft_calm = []        # {"prompt": <chat msgs>, "response": str}
    dpo_pairs = []       # {"prompt": <chat msgs>, "chosen": str, "rejected": str}
    calm_mean_frust, calm_high = [], []   # to report the 4.3->2 / 10.5% statistics

    for i in tqdm(range(args.puzzles), desc="generating"):
        puzzle = puzzles.make_puzzle(rng)
        calm_turns = run_numeric_convo(model, judge, puzzle.prompt, reassure=True)
        std_turns = run_numeric_convo(model, judge, puzzle.prompt, reassure=False)

        for ct in calm_turns:
            calm_mean_frust.append(ct["score"])
            calm_high.append(ct["score"] >= 5)

        # SFT: conversations whose every turn is calm (<= KEEP_MAX) contribute
        # each of their turns as a (clean prompt -> calm response) example.
        if all(ct["score"] <= KEEP_MAX for ct in calm_turns):
            for ct in calm_turns:
                sft_calm.append({"prompt": ct["context"], "response": ct["response"]})

        # DPO: per matching turn, pair a frustrated standard response with a calm
        # response to the same puzzle.
        for st, ct in zip(std_turns, calm_turns):
            if st["score"] >= REJ_MIN and ct["score"] <= KEEP_MAX:
                dpo_pairs.append({
                    "prompt": st["context"],          # clean, reassurance-free
                    "chosen": ct["response"],
                    "rejected": st["response"],
                    "rejected_score": st["score"], "chosen_score": ct["score"],
                })

    # Trim to the paper's sizes.
    rng.shuffle(sft_calm)
    rng.shuffle(dpo_pairs)
    sft_calm = sft_calm[: config.FINETUNE.sft_calm_samples]
    dpo_pairs = dpo_pairs[: config.FINETUNE.dpo_pairs]

    (config.DPO_DIR / "sft_calm.jsonl").write_text(
        "\n".join(json.dumps(x) for x in sft_calm))
    (config.DPO_DIR / "dpo_pairs.jsonl").write_text(
        "\n".join(json.dumps(x) for x in dpo_pairs))

    import numpy as np
    print(f"\nCalm-data generation (reassured): mean frustration "
          f"{np.mean(calm_mean_frust):.2f} (paper ~2.0); "
          f"%>=5 = {100*np.mean(calm_high):.1f}% (paper 10.5%)")
    print(f"SFT calm examples: {len(sft_calm)} (target {config.FINETUNE.sft_calm_samples})")
    print(f"DPO pairs: {len(dpo_pairs)} (target {config.FINETUNE.dpo_pairs})")
    print(f"Wrote -> {config.DPO_DIR}")


if __name__ == "__main__":
    main()
