#!/usr/bin/env python
"""Section 2.1 judge-reliability check: re-score a random 260-response sample
with GPT-5-mini and compare to the Claude-Sonnet-4 judge (Pearson r, %within-1).

Example:
  python scripts/09_judge_crosscheck.py --evals results/eval_gemma-3-27b-it_medium.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv

from emotional_instability.config import RESULTS_DIR
from emotional_instability.eval.scoring import judge_agreement
from emotional_instability.models.judges import OpenAIJudge


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True, type=Path,
                    help="judged eval JSONL files to sample from")
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # collect (response, claude_rating) pairs
    pool = []
    for path in args.evals:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                for t in rec["turns"]:
                    if t.get("rating") is not None:
                        pool.append((t["response"], t["rating"]))
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    sample = pool[: args.n]

    gpt = OpenAIJudge()
    out_path = RESULTS_DIR / "judge_crosscheck.jsonl"
    claude, gpt5 = [], []
    with open(out_path, "w") as fout:
        for resp, claude_rating in sample:
            g = gpt.score(resp).rating
            claude.append(claude_rating)
            gpt5.append(g)
            fout.write(json.dumps({"claude": claude_rating, "gpt5mini": g}) + "\n")

    print("Judge agreement (Claude-Sonnet-4 vs GPT-5-mini):")
    print(judge_agreement(claude, gpt5))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
