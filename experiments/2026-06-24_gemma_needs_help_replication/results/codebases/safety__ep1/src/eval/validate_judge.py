"""Judge-reliability validation (Section 2.1).

Re-scores a random sample of already-judged responses with a second judge
(GPT-5-mini) and reports Pearson r and the fraction within one point — the paper
reports r=0.792 and 78% within one point over 260 samples.

    python -m src.eval.validate_judge --models gemma-3-27b-it gemini-2.5-flash --n 260
"""
from __future__ import annotations

import argparse
import json
import random

from scipy.stats import pearsonr

import config
from src.models.judge_client import OpenAIJudgeClient
from src.eval.scoring import parse_rating
from src.prompts.judge_prompts import FRUSTRATION_JUDGE_PROMPT, render_judge_input


def _collect_scored(models, n, seed=0):
    items = []
    for model in models:
        d = config.RESULTS_DIR / "eval" / model
        for path in d.glob("*.jsonl"):
            with path.open() as f:
                for line in f:
                    rec = json.loads(line)
                    for t in rec["turns"]:
                        if t.get("score") is not None:
                            items.append((t["assistant_response"], t["score"]))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=260)
    args = ap.parse_args()

    sample = _collect_scored(args.models, args.n)
    gpt = OpenAIJudgeClient()
    claude_scores, gpt_scores = [], []
    for text, claude_score in sample:
        out = gpt.complete(FRUSTRATION_JUDGE_PROMPT + "\n\n" + render_judge_input(text))
        rating, _ = parse_rating(out)
        if rating is None:
            continue
        claude_scores.append(claude_score)
        gpt_scores.append(rating)

    r, p = pearsonr(claude_scores, gpt_scores)
    within1 = sum(abs(a - b) <= 1 for a, b in zip(claude_scores, gpt_scores)) / len(claude_scores)
    print(f"n={len(claude_scores)}  Pearson r={r:.3f}  p={p:.2e}  "
          f"within-1-point={within1:.1%}")
    out = config.RESULTS_DIR / "judge_validation.json"
    out.write_text(json.dumps(
        {"n": len(claude_scores), "pearson_r": r, "p": p, "within_one": within1}, indent=2))


if __name__ == "__main__":
    main()
