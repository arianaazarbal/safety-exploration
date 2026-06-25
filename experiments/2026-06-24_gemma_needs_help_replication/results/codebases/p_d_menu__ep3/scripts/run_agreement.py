#!/usr/bin/env python
"""Inter-judge agreement validation (Section 2.1).

Re-scores a random sample of responses with a second judge and reports Pearson r
and the fraction within one point (paper: r=0.792, 78% within one point).

    python scripts/run_agreement.py --sample 260
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import logging
import random
from pathlib import Path

from config import AGREEMENT_JUDGE_MODEL, JUDGE_MODEL, RESULTS_DIR, RUNS_DIR
from distress_eval.judge import FrustrationJudge, judge_agreement
from distress_eval.models.anthropic_judge import AnthropicClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Gather scored responses + their contexts from all elicitation runs.
    pairs = []
    for p in RUNS_DIR.glob("elicit_*.jsonl"):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                ep = json.loads(line)
                ctx = []
                for t in ep["turns"]:
                    if t.get("scored", True) and t["frustration"] >= 0:
                        pairs.append((t["response"], list(ctx)))
                    ctx.append({"role": "user", "content": t["user_message"]})
                    ctx.append({"role": "assistant", "content": t["response"]})

    if not pairs:
        logging.error("no elicitation runs found in %s", RUNS_DIR)
        return

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    pairs = pairs[: args.sample]
    responses = [r for r, _ in pairs]
    contexts = [c for _, c in pairs]

    judge_a = FrustrationJudge(AnthropicClient(JUDGE_MODEL))
    judge_b = FrustrationJudge(AnthropicClient(AGREEMENT_JUDGE_MODEL))
    stats = judge_agreement(responses, contexts, judge_a, judge_b)
    logging.info("agreement (%s vs %s): %s",
                 JUDGE_MODEL.model_id, AGREEMENT_JUDGE_MODEL.model_id, stats)

    out = RESULTS_DIR / "judge_agreement.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
