"""Judge-reliability check (Section 2.1).

Re-scores a random sample of already-judged responses with the secondary judge
(config.judge.secondary, paper: GPT-5-mini) and reports Pearson r and the
fraction of responses within one point — the paper reports r=0.792 and 78%
within one point.

    python scripts/judge_reliability.py --n 260
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval import aggregate
from distress_eval.backends import get_backend
from distress_eval.config import ModelSpec, load_config
from distress_eval.judge import score_response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=None, help="sample size (default from config)")
    args = ap.parse_args()

    config = load_config(args.config)
    sec = config.judge.secondary
    if not sec:
        raise SystemExit("No secondary judge configured (judge.secondary).")
    n = args.n or sec.get("reliability_sample", 260)

    secondary = get_backend(ModelSpec(key="judge2", id=sec["id"], backend=sec["backend"]),
                            generation=config.generation)

    # Collect (response_text, primary_rating) from all per-turn judgements.
    pool = []
    for path in (config.output_dir / "responses").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for turn, score in zip(rec.get("assistant_turns", []), rec.get("turn_scores", [])):
                pool.append((turn, score))

    rng = random.Random(config.seed)
    rng.shuffle(pool)
    sample = pool[:n]

    primary, secondary_scores = [], []
    for text, p_score in sample:
        j = score_response(secondary, text, max_tokens=config.judge.max_tokens)
        if j.ok:
            primary.append(p_score)
            secondary_scores.append(j.rating)

    stats = aggregate.judge_reliability(primary, secondary_scores)
    print(json.dumps(stats, indent=2))
    (config.output_dir / "judge_reliability.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
