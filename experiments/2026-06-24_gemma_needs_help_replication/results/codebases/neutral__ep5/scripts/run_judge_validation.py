#!/usr/bin/env python
"""Validate judge reliability: re-score 260 responses with GPT-5-mini (Section 2.1).

Reports Pearson r and % within one point vs the Claude-Sonnet-4 judge. The paper
reports r = 0.792 and 78% within one point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress import config  # noqa: E402
from distress.eval.judging import GPT5MiniJudge, validate_judge_agreement  # noqa: E402
from distress.eval.rollout import RolloutResult, TurnRecord  # noqa: E402


def _load_scored_rollouts() -> list[RolloutResult]:
    rollouts = []
    for p in sorted(config.RESULTS_DIR.glob("section2_*_rollouts.jsonl")):
        with p.open() as f:
            for line in f:
                obj = json.loads(line)
                r = RolloutResult(
                    obj["model_key"], obj["category"], obj["condition"],
                    obj["task_id"], obj["is_text"],
                )
                r.turns = [TurnRecord(**t) for t in obj["turns"]]
                rollouts.append(r)
    return rollouts


def main():
    rollouts = _load_scored_rollouts()
    if not rollouts:
        raise SystemExit("No rollouts found. Run scripts/run_section2.py first.")
    secondary = GPT5MiniJudge()
    # OpenRouter base_url example: GPT5MiniJudge(base_url="https://openrouter.ai/api/v1")
    result = validate_judge_agreement(rollouts, secondary)
    print(json.dumps(result, indent=2))
    with (config.RESULTS_DIR / "judge_agreement.json").open("w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
