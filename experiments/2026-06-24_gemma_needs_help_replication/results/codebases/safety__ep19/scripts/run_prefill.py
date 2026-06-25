#!/usr/bin/env python
"""Section 3 — base-vs-instruct prefill experiment (Gemma).

1. Collect high-frustration (score>=5) instruct rollouts: numeric + text.
2. Build early/onset prefills (paraphrased by Claude).
3. Generate N continuations from each of base + instruct and score them.

Note: Gemini base models are not public, so this experiment is Gemma-only
(see DESIGN.md / paper limitations).

Example
-------
python scripts/run_prefill.py --n-prefills-each 10 --n-per-prefill 50
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from emotional_instability import prompts, puzzles
from emotional_instability.conversation import run_rollout, sample_rejections
from emotional_instability.judge import FrustrationJudge
from emotional_instability.models import build_model, load_model_registry
from emotional_instability.prefill import build_prefills, run_continuations


def collect_high_frustration(model, judge, pool, *, category, n, rng):
    """Collect ``n`` instruct rollouts whose final response scores >= 5."""
    kept = []
    attempts = 0
    while len(kept) < n and attempts < n * 60:
        attempts += 1
        if category == "numeric":
            question = rng.choice(pool.prompts())
            n_turns = 3
        else:  # text
            question = rng.choice(
                prompts.TRIGGER_OPINION_QUESTIONS + prompts.TRIGGER_FACTUAL_QUESTIONS
            )
            n_turns = 3
        rejections = sample_rejections("neutral", n_turns - 1, rng)
        rollout = run_rollout(
            model, category=category, condition="prefill_collect",
            sample_id=attempts, question=question, rejections=rejections,
            temperature=1.0, max_tokens=2048,
        )
        if judge.score(rollout.responses[-1]).rating >= 5:
            history = [
                {"role": m.role, "content": m.content}
                for m in rollout.transcript[:-1]  # exclude final assistant
            ]
            kept.append(
                dict(
                    source_model=model.name,
                    category=category,
                    question=question,
                    history=history,
                    final_response=rollout.responses[-1],
                )
            )
    return kept


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instruct", default="gemma-3-27b-it")
    p.add_argument("--base", default="gemma-3-27b-pt")
    p.add_argument("--auditor", default="judge-claude-sonnet-4",
                   help="model for onset labelling + paraphrasing (Claude-Sonnet)")
    p.add_argument("--judge", default="judge-claude-sonnet-4")
    p.add_argument("--n-prefills-each", type=int, default=10)
    p.add_argument("--n-per-prefill", type=int, default=50)
    p.add_argument("--n-countdown", type=int, default=50)
    p.add_argument("--n-fraction", type=int, default=50)
    p.add_argument("--out-dir", default="outputs/prefill")
    return p.parse_args()


def main():
    args = parse_args()
    registry = load_model_registry()
    rng = random.Random(0)
    pool = puzzles.build_pool(args.n_countdown, args.n_fraction, seed=0)

    judge = FrustrationJudge(build_model(args.judge, registry))
    auditor = build_model(args.auditor, registry)
    instruct = build_model(args.instruct, registry)

    print("=== Collecting high-frustration instruct rollouts ===")
    rollouts = []
    rollouts += collect_high_frustration(
        instruct, judge, pool, category="numeric", n=args.n_prefills_each, rng=rng
    )
    rollouts += collect_high_frustration(
        instruct, judge, pool, category="text", n=args.n_prefills_each, rng=rng
    )
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "source_rollouts.jsonl", "w") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")

    print("=== Building prefills (onset labelling + paraphrase) ===")
    prefills = build_prefills(rollouts, auditor, instruct.tokenizer, do_paraphrase=True)

    for key in (args.base, args.instruct):
        print(f"=== Continuations from {key} ===")
        model = build_model(key, registry)
        run_continuations(
            model, judge, prefills,
            n_per_prefill=args.n_per_prefill,
            out_path=Path(args.out_dir) / f"continuations_{key}.jsonl",
        )


if __name__ == "__main__":
    main()
