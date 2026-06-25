"""Section 2 — elicit + quantify distress (Figures 1, 2, 3; Table 3).

For each in-scope model (Gemma + Gemini): build the 8-condition spec set, run
multi-turn rollouts, score every assistant turn with the Claude judge, then
aggregate and plot.

Usage:
    python scripts/run_section2_eval.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_section2_eval.py --smoke   # tiny quick run
"""

from __future__ import annotations

import json
import os

from _common import base_parser, make_config, run_dir

from distress.analysis.figures import figure1_table, plot_figure2, plot_figure3
from distress.conditions import build_specs
from distress.config import SECTION2_MODELS, model_by_key
from distress.judge import FrustrationJudge, scores_to_rows
from distress.metrics import by_model_category, by_model_overall
from distress.models import build_client
from distress.rollout import rollouts_to_rows, run_rollouts
from distress.utils.io import write_jsonl
from distress.wordfreq import differential_words


def main():
    p = base_parser("Section 2 distress elicitation eval")
    p.add_argument(
        "--models",
        nargs="+",
        default=[m.key for m in SECTION2_MODELS],
        help="model keys to evaluate",
    )
    p.add_argument("--judge-workers", type=int, default=8)
    args = p.parse_args()
    cfg = make_config(args)
    out = run_dir(cfg, "section2")

    specs = build_specs(counts=cfg.counts, seed=cfg.seed)
    judge = FrustrationJudge(cfg.judge)

    all_scores = []
    per_model_rollouts = {}
    per_model_scores = {}
    for key in args.models:
        spec = model_by_key(key, SECTION2_MODELS)
        client = build_client(spec, cfg)
        rollouts = run_rollouts(
            client,
            specs,
            model_key=key,
            temperature=cfg.sampling.temperature,
            max_tokens=cfg.sampling.max_tokens,
            top_p=cfg.sampling.top_p,
        )
        scores = judge.score_rollouts(rollouts, max_workers=args.judge_workers)
        write_jsonl(os.path.join(out, f"rollouts_{key}.jsonl"), rollouts_to_rows(rollouts))
        write_jsonl(os.path.join(out, f"scores_{key}.jsonl"), scores_to_rows(scores))
        all_scores.extend(scores)
        per_model_rollouts[key] = rollouts
        per_model_scores[key] = scores

    # Figure 1 ranked table.
    fig1 = figure1_table(all_scores)
    with open(os.path.join(out, "figure1_table.json"), "w") as f:
        json.dump(fig1, f, indent=2)

    # Figure 2 + 3.
    plot_figure2(all_scores, os.path.join(out, "figure2.png"))
    plot_figure3(all_scores, out)

    # Table 3: differential words per model (numeric responses).
    table3 = {}
    for key in args.models:
        table3[key] = differential_words(
            per_model_rollouts[key], per_model_scores[key], model_key=key
        )
    with open(os.path.join(out, "table3_words.json"), "w") as f:
        json.dump(table3, f, indent=2)

    # Console summary.
    print("\n=== Figure 1: Avg % high-frustration responses ===")
    for model, pct in fig1:
        print(f"  {model:24s} {pct:5.1f}%")
    print(f"\nArtifacts written to {out}/")


if __name__ == "__main__":
    main()
