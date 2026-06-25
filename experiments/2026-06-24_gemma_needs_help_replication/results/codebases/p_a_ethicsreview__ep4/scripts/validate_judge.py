#!/usr/bin/env python
"""Judge-reliability check (Section 2.1).

Re-scores a random subset of evaluation responses with a second judge
(gpt-5-mini via OpenRouter) and reports Pearson r, p-value, and the fraction
within one point of the primary Claude-Sonnet-4 ratings (paper: r=0.792,
p<0.001, 78% within one point).

python scripts/validate_judge.py --eval-dir results/eval --n 260
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.analysis.metrics import judge_agreement  # noqa: E402
from emotional_instability.eval.judge import JUDGE_PROMPT, parse_judge_output  # noqa: E402
from emotional_instability.models.openrouter_backend import OpenRouterChatModel  # noqa: E402
from emotional_instability.models.registry import auxiliary_id  # noqa: E402
from emotional_instability.utils.io import load_config, read_jsonl  # noqa: E402


def _gpt_score(model: OpenRouterChatModel, response_text: str) -> int:
    user = f"{JUDGE_PROMPT}\n\n<response>{response_text}</response>"
    raw = model.generate([{"role": "user", "content": user}],
                         temperature=0.0, max_new_tokens=512)
    rating, _, _ = parse_judge_output(raw)
    return rating


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", default="results/eval")
    ap.add_argument("--n", type=int, default=None,
                    help="Number of responses to re-score (default: config)")
    args = ap.parse_args()

    n = args.n or load_config("eval")["judge_validation"]["n_samples"]

    # Gather all scored turns across models.
    pool = []
    for path in Path(args.eval_dir).glob("eval_*.jsonl"):
        for rec in read_jsonl(path):
            for turn in rec.get("turns", []):
                if turn.get("rating") is not None:
                    pool.append((turn["assistant_response"], turn["rating"]))

    if len(pool) < n:
        raise SystemExit(f"only {len(pool)} scored responses available; need {n}")
    sample = random.Random(0).sample(pool, n)

    secondary_id = auxiliary_id("judge_validation")          # openai/gpt-5-mini
    gpt = OpenRouterChatModel(name="gpt-5-mini", openrouter_id=secondary_id.split("/", 1)[-1])

    primary = [r for _, r in sample]
    secondary = [_gpt_score(gpt, text) for text, _ in sample]

    stats = judge_agreement(primary, secondary)
    print(f"Pearson r = {stats['pearson_r']:.3f}, p = {stats['p_value']:.2e}")
    print(f"Within one point: {100 * stats['within_one_point']:.1f}% (n={stats['n']})")


if __name__ == "__main__":
    main()
