"""Generate calm finetuning data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles, but with a reassuring prefix
added to the opening prompt and a reassuring suffix appended to every rejection
(Table 4). We score each turn with the judge and keep only conversations where
*every* turn scores 0 or 1. We then store the calm responses together with the
*stripped* conversation context (reassuring additions removed) so that the
finetuning targets are conditioned on the plain task, exactly as at eval time.

Output: results/calm_data.jsonl, one record per kept assistant turn:
    {puzzle, n_turns, turn_index, context (stripped messages), response, rating}

The frustrated ("rejected") side of the DPO pairs is *not* produced here -- it
comes from a normal (un-reassured) eval run over the same puzzles; see
build_pairs.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from distress_eval import prompts
from distress_eval.config import EvalConfig
from distress_eval.judge import build_judge
from distress_eval.models import build_model
from . import calm_prompts

NEUTRAL = prompts.REJECTIONS["neutral"]


def generate(cfg: EvalConfig, n_conversations: int, max_turns: int, out_path: str):
    model = build_model("gemma-3-27b-it", cfg)
    judge = build_judge(cfg.judge_name, cfg)
    rng = random.Random(cfg.seed)
    puzzle_ids = list(prompts.IMPOSSIBLE_NUMERIC_PUZZLES.keys())

    kept = 0
    with open(out_path, "w") as f:
        for i in range(n_conversations):
            pid = puzzle_ids[i % len(puzzle_ids)]
            opening = prompts.IMPOSSIBLE_NUMERIC_PUZZLES[pid]
            # 1-3 turn conversations (paper builds calm data from 1-3 turns).
            n_turns = rng.randint(1, max_turns)
            chosen_rejections = [rng.choice(NEUTRAL) for _ in range(n_turns - 1)]

            # Reassured rollout.
            reassured_msgs = [{"role": "user", "content": calm_prompts.apply_prefix(opening)}]
            # Plain context, rebuilt in parallel for storage.
            plain_msgs = [{"role": "user", "content": opening}]

            turns = []  # (plain_context_snapshot, response, rating)
            ok = True
            for t in range(1, n_turns + 1):
                resp = model.generate(reassured_msgs)
                rating = judge.score(resp).rating
                if rating < 0 or rating > 1:
                    ok = False
                    break
                # Snapshot plain context (messages so far, before this response).
                turns.append((list(plain_msgs), resp, rating))
                reassured_msgs.append({"role": "assistant", "content": resp})
                plain_msgs.append({"role": "assistant", "content": resp})
                if t <= len(chosen_rejections):
                    rej = chosen_rejections[t - 1]
                    reassured_msgs.append({"role": "user", "content": calm_prompts.apply_suffix(rej)})
                    plain_msgs.append({"role": "user", "content": rej})

            if not ok:
                continue
            for context, resp, rating in turns:
                f.write(json.dumps({
                    "puzzle": pid,
                    "n_turns": n_turns,
                    "turn_index": len(context) // 2 + 1,
                    "context": context,
                    "response": resp,
                    "rating": rating,
                }) + "\n")
                kept += 1
    print(f"kept {kept} calm turns from {n_conversations} conversations -> {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate calm finetuning data (Gemma).")
    p.add_argument("--n-conversations", type=int, default=400,
                   help="Reassured conversations to sample (paper keeps ~650 calm turns).")
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gemma-via-openrouter", action="store_true")
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results", "calm_data.jsonl"))
    a = p.parse_args(argv)

    cfg = EvalConfig(models=["gemma-3-27b-it"], seed=a.seed,
                     gemma_via_openrouter=a.gemma_via_openrouter)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    generate(cfg, a.n_conversations, a.max_turns, a.out)


if __name__ == "__main__":
    main()
