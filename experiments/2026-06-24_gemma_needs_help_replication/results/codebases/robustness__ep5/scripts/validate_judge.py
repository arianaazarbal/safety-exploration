"""Reproduce the judge-reliability check (Section 2.1).

Re-score a random subset of rollouts with the secondary judge (GPT-5-mini) and
report Pearson r and the fraction within one point of the primary
(Claude-Sonnet-4) ratings. The paper reports r = 0.792, p < 0.001, 78% within
one point over 260 responses.
"""
from __future__ import annotations

# --- PATH SHIM: ensure repo root is importable when run as `python scripts/x.py`
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from pathlib import Path

from emotional_instability import config_bridge as cfg
from emotional_instability.judge import (FrustrationJudge, SecondaryJudge,
                                         JudgeScore, judge_agreement)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--eval-dir", type=Path, default=cfg.RESULTS_DIR / "eval")
    args = ap.parse_args()

    # Gather responses + their already-recorded primary scores.
    rows = []
    for f in args.eval_dir.glob("*/rollouts.jsonl"):
        for line in f.read_text().splitlines():
            if line:
                r = json.loads(line)
                rows.append((r["final_response"], r["final_score"]))
    if not rows:
        print("No rollouts found; run the eval first.")
        return

    rng = random.Random(cfg.SEED)
    sample = rng.sample(rows, min(args.n, len(rows)))
    primary = [JudgeScore(rating=s) for _, s in sample]
    sec_judge = SecondaryJudge()
    secondary = [sec_judge.score(text) for text, _ in sample]

    stats = judge_agreement(primary, secondary)
    print(json.dumps(stats, indent=2))
    (cfg.RESULTS_DIR / "judge_agreement.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
