#!/usr/bin/env python
"""Section 2.1 validation: re-score a random subset of collected responses with
the cross-check judge (GPT-5-mini) and report Pearson r + within-one-point
agreement against the primary Claude-Sonnet-4 judge.

  python scripts/08_judge_agreement.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import _bootstrap  # noqa: F401

from gemma_distress.analysis import load_turns
from gemma_distress.config import Config
from gemma_distress.judge import FrustrationJudge, crosscheck_agreement
from gemma_distress.utils.io import read_jsonl, write_json


def _collect_scored_responses(distress_dir):
    """Yield (assistant_text, primary_score) across all model JSONLs."""
    out = []
    for path in Path(distress_dir).glob("*.jsonl"):
        for r in read_jsonl(path):
            for t in r["turns"]:
                if t.get("frustration") is not None:
                    out.append((t["assistant_message"], t["frustration"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)

    pairs = _collect_scored_responses(f"{cfg.results_dir}/distress")
    if not pairs:
        raise SystemExit("No scored responses found; run 01 first.")
    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    k = max(2, int(len(pairs) * cfg.judge.crosscheck_fraction))
    sample = pairs[:k]

    cross = FrustrationJudge(provider=cfg.judge.crosscheck_provider,
                             model=cfg.judge.crosscheck_model)
    primary_scores, secondary_scores = [], []
    for text, primary in sample:
        primary_scores.append(primary)
        secondary_scores.append(cross.score(text).rating)

    stats = crosscheck_agreement(primary_scores, secondary_scores)
    print(stats)
    write_json(Path(cfg.results_dir) / "judge_agreement.json", stats)


if __name__ == "__main__":
    main()
