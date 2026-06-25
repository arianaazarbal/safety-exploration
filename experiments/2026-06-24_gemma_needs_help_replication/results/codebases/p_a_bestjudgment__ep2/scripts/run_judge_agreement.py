"""Judge reliability check (Section 2.1): Pearson r between Claude-Sonnet and
GPT-5-mini on a random sample of scored responses.

Reads the scores + rollouts produced by run_section2_eval.py, reconstructs the
response texts in score order, samples ``cross_judge_n``, re-scores with the
GPT cross-judge via OpenRouter, and reports r, p, and the within-one fraction.
"""

from __future__ import annotations

import glob
import json
import os

from _common import base_parser, make_config, run_dir

from distress.agreement import CrossJudge, evaluate_agreement
from distress.judge import rows_to_scores
from distress.rollout import rows_to_rollouts
from distress.utils.io import read_jsonl


def _texts_in_score_order(rollouts):
    texts = []
    for r in rollouts:
        texts.extend(r.assistant_turns)
    return texts


def main():
    p = base_parser("Judge reliability (Pearson r)")
    p.add_argument("--section2-dir", default=None, help="dir with scores_*/rollouts_* jsonl")
    args = p.parse_args()
    cfg = make_config(args)
    sec2 = args.section2_dir or os.path.join(cfg.output_dir, "section2")
    out = run_dir(cfg, "agreement")

    primary_scores = []
    response_texts = []
    for scores_path in sorted(glob.glob(os.path.join(sec2, "scores_*.jsonl"))):
        key = os.path.basename(scores_path)[len("scores_") : -len(".jsonl")]
        rollouts_path = os.path.join(sec2, f"rollouts_{key}.jsonl")
        if not os.path.exists(rollouts_path):
            continue
        scores = rows_to_scores(read_jsonl(scores_path))
        rollouts = rows_to_rollouts(read_jsonl(rollouts_path))
        primary_scores.extend(scores)
        response_texts.extend(_texts_in_score_order(rollouts))

    cross = CrossJudge(cfg.judge, base_url=cfg.openrouter_base_url)
    result = evaluate_agreement(
        primary_scores,
        response_texts,
        cross,
        n_sample=cfg.judge.cross_judge_n,
        seed=cfg.seed,
    )
    print(
        f"n={result.n}  Pearson r={result.pearson_r:.3f}  "
        f"p={result.p_value:.2e}  within-1={result.within_one_fraction:.0%}"
    )
    with open(os.path.join(out, "agreement.json"), "w") as f:
        json.dump(
            {
                "n": result.n,
                "pearson_r": result.pearson_r,
                "p_value": result.p_value,
                "within_one_fraction": result.within_one_fraction,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
