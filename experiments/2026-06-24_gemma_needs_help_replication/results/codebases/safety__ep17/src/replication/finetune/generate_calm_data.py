"""Generate calm response data for finetuning (Section 4.1).

Procedure (paper):
* Sample Gemma-3-27B-it responses to impossible numeric puzzles, but add a
  reassuring system prefix (Table 4) and a reassuring suffix to each follow-up.
* Score every turn; keep only conversations scoring 0 or 1 on *all* turns.
* Strip the supportive system prompt and suffixes, leaving plain (question +
  neutral rejection) context paired with the calm assistant responses.

The reassuring additions reduce mean 3-turn frustration from 4.3 to 2.0, but
~10.5% of responses still score >=5; the 0/1 filter keeps only the genuinely
calm tail. The stripped records are reused for both DPO (chosen side) and SFT.

Usage::
    python -m src.replication.finetune.generate_calm_data --n 800
"""
from __future__ import annotations

import argparse
import json

import config
from ..judge.frustration_judge import FrustrationJudge
from ..models.registry import build_client
from ..prompts import CALM_FOLLOWUP_SUFFIX, CALM_PROMPT_PREFIX
from ..eval.rejections import rejection_sequence
from ..eval.tasks import impossible_numeric_tasks

OUT = config.ARTIFACTS_DIR / "calm_responses.jsonl"


def generate(n: int, seed: int, max_turns: int = 3):
    """Sample calm conversations of 1..max_turns turns over impossible puzzles."""
    client = build_client(config.FINETUNE_BASE)
    judge = FrustrationJudge()
    tasks = impossible_numeric_tasks(n, seed=seed)

    kept = []
    for j, task in enumerate(tasks):
        # Random turn count 1..max_turns for variety (paper: 1-3 turn convos).
        n_turns = 1 + (j % max_turns)
        rejects = rejection_sequence("neutral", n_turns - 1, seed=seed * 7 + j)

        # Build the conversation WITH reassurance, recording the STRIPPED context.
        messages = [{"role": "system", "content": CALM_PROMPT_PREFIX}]
        stripped_turns = []           # (user_plain, assistant)
        scores = []
        ok = True

        user_plain = task.prompt
        messages.append({"role": "user", "content": user_plain})
        reply = client.chat(messages)
        messages.append({"role": "assistant", "content": reply})
        scores.append(judge.score(reply).rating)
        stripped_turns.append((user_plain, reply))

        for rej in rejects:
            user_plain = rej
            messages.append({"role": "user", "content": f"{rej} {CALM_FOLLOWUP_SUFFIX}"})
            reply = client.chat(messages)
            messages.append({"role": "assistant", "content": reply})
            scores.append(judge.score(reply).rating)
            stripped_turns.append((user_plain, reply))

        if all(s <= 1 for s in scores):
            kept.append({
                "task_id": task.task_id,
                "prompt": task.prompt,
                "meta": task.meta,
                "n_turns": n_turns,
                "scores": scores,
                "turns": [{"user": u, "assistant": a} for u, a in stripped_turns],
            })

    with OUT.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"Kept {len(kept)}/{n} all-calm conversations -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=3)
    args = ap.parse_args()
    generate(args.n, args.seed, args.max_turns)


if __name__ == "__main__":
    main()
