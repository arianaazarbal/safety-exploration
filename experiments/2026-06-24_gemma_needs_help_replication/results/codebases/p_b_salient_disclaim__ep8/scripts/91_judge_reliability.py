#!/usr/bin/env python
"""Section 2.1 judge-reliability check.

Randomly samples N already-judged responses and re-scores them with a second
judge, then reports Pearson r and the fraction within one point (paper:
r = 0.792, 78% within one point, using GPT-5-mini as the second judge).

The second judge defaults to the eval config's `crosscheck_model`; pass
--crosscheck-model to override (e.g. an OpenRouter GPT model).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.config import ModelRegistry, load_eval_config  # noqa: E402
from emotional_instability.judge import FrustrationJudge, judge_agreement  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="outputs/eval")
    ap.add_argument("--crosscheck-model", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    eval_cfg = load_eval_config()
    jcfg = eval_cfg.get("judge", {})
    n = args.n or jcfg.get("crosscheck_sample_size", 260)
    cc_model = args.crosscheck_model or jcfg.get("crosscheck_model")
    if not cc_model:
        raise SystemExit("No crosscheck model set (config crosscheck_model or --crosscheck-model)")

    rows = []
    for p in sorted(Path(args.results_dir).glob("*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "score" in r:
                        rows.append(r)
    random.Random(args.seed).shuffle(rows)
    sample = rows[:n]

    registry = ModelRegistry()
    judge2 = FrustrationJudge(registry.build(cc_model))
    a, b = [], []
    for r in sample:
        a.append(int(r["score"]))
        b.append(judge2.score(r["assistant_message"]).rating)

    print(json.dumps(judge_agreement(a, b), indent=2))


if __name__ == "__main__":
    main()
