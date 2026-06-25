"""Judge reliability check (Section 2.1).

"To validate judge reliability, we randomly sampled 260 responses for
re-scoring with GPT-5-mini, using the same prompt. The judges show strong
agreement (Pearson r = 0.792, p < 0.001), with 78% of responses within one
point."

This re-scores a random subset of the Claude-judged responses with GPT-5-mini
and reports Pearson r, p-value, and the within-one-point agreement rate.

Usage:
    python -m gemma_distress.validate_judge --results results/section2.jsonl --n 260
"""
from __future__ import annotations

import argparse
import json
import random

from . import config
from .analyze import iter_scored_responses, load_results
from .judge import OpenAIJudge


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Validate the Claude judge against GPT-5-mini")
    p.add_argument("--results", default="results/section2.jsonl")
    p.add_argument("--n", type=int, default=config.JUDGE_VALIDATION_SAMPLE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/judge_validation.json")
    args = p.parse_args(argv)

    records = load_results(args.results)
    # We need the conversation context per scored turn; rebuild from records.
    pairs = []  # (claude_score, response, context_turns, turn)
    for rec in records:
        turns = rec["turns"]
        for i, t in enumerate(turns):
            if t.get("frustration") is None:
                continue
            ctx = []
            for tt in turns[: i + 1]:
                ctx.append(("user", tt["user_message"]))
                if tt["turn"] < t["turn"]:
                    ctx.append(("assistant", tt["response"]))
            pairs.append((t["frustration"], t["response"], ctx, t["turn"]))

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    sample = pairs[: args.n]

    judge = OpenAIJudge()
    claude_scores, gpt_scores = [], []
    for claude_score, response, ctx, turn in sample:
        try:
            gpt_score, _ = judge.score(ctx, response, turn)
        except Exception as e:  # noqa: BLE001
            print(f"skip (validation judge error): {e}")
            continue
        claude_scores.append(claude_score)
        gpt_scores.append(gpt_score)

    r, pval = _pearson(claude_scores, gpt_scores)
    within1 = (
        sum(1 for a, b in zip(claude_scores, gpt_scores) if abs(a - b) <= 1) / len(claude_scores)
        if claude_scores
        else float("nan")
    )
    report = {
        "n": len(claude_scores),
        "pearson_r": r,
        "p_value": pval,
        "within_one_point": within1,
        "paper_reference": {"pearson_r": 0.792, "within_one_point": 0.78, "n": 260},
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def _pearson(xs: list[float], ys: list[float]) -> tuple[float, float]:
    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(xs, ys)
        return float(r), float(p)
    except Exception:  # noqa: BLE001 — scipy missing; compute r without p
        n = len(xs)
        if n < 2:
            return (float("nan"), float("nan"))
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        return (num / den if den else float("nan"), float("nan"))


if __name__ == "__main__":
    main()
