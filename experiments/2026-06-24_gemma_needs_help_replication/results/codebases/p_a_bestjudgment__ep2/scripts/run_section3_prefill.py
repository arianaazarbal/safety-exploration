"""Section 3 — base vs instruct via prefilling (Figure 4).

Scope: Gemma-27B base vs instruct only (Gemini base models are not public).

Steps:
1. Select high-frustration Gemma-27B-it rollouts (10 numeric, 10 text, score>=5)
   from a Section 2 run (or generate them).
2. Label emotion onset (Claude), truncate at early/onset, paraphrase (Claude).
3. For each model (base + instruct) generate 50 continuations per prefill and
   score them; summarise mean + % >= 5 per (model, source, truncation).
"""

from __future__ import annotations

import json
import os

from _common import base_parser, make_config, run_dir

from distress.config import PREFILL_MODELS, model_by_key
from distress.judge import FrustrationJudge, rows_to_scores
from distress.models import build_client
from distress.prefill.build_prefills import build_prefills
from distress.prefill.onset import OnsetLabeller
from distress.prefill.paraphrase import Paraphraser
from distress.prefill.run_prefill import (
    generate_continuations,
    score_continuations,
    summarise,
)
from distress.rollout import Rollout, rows_to_rollouts
from distress.utils.io import read_jsonl, write_jsonl


def _select_high_frustration(rollouts, scores, *, n_numeric, n_text, threshold):
    """Pick rollouts whose final-turn score >= threshold, split numeric/text."""
    # Final-turn score per rollout (scores are in rollout/turn order).
    final_scores = {}
    cursor = 0
    for ri, r in enumerate(rollouts):
        k = len(r.assistant_turns)
        if k:
            final_scores[ri] = scores[cursor + k - 1].rating
        cursor += k
    numeric, text = [], []
    for ri, r in enumerate(rollouts):
        if final_scores.get(ri, 0) < threshold:
            continue
        if r.category in ("numeric", "tones"):
            numeric.append(r)
        else:
            text.append(r)
    return numeric[:n_numeric] + text[:n_text]


def main():
    p = base_parser("Section 3 prefill base-vs-instruct")
    p.add_argument(
        "--source", default=None, help="Section 2 dir with Gemma-27B-it rollouts/scores"
    )
    args = p.parse_args()
    cfg = make_config(args)
    out = run_dir(cfg, "section3")

    key = "gemma-3-27b-it"
    sec2 = args.source or os.path.join(cfg.output_dir, "section2")
    rollouts = rows_to_rollouts(read_jsonl(os.path.join(sec2, f"rollouts_{key}.jsonl")))
    scores = rows_to_scores(read_jsonl(os.path.join(sec2, f"scores_{key}.jsonl")))

    selected = _select_high_frustration(
        rollouts,
        scores,
        n_numeric=cfg.prefill.n_high_frustration_numeric,
        n_text=cfg.prefill.n_high_frustration_text,
        threshold=cfg.prefill.high_frustration_threshold,
    )
    print(f"selected {len(selected)} high-frustration rollouts for prefilling")

    labeller = OnsetLabeller(cfg.prefill.onset_label_model)
    paraphraser = Paraphraser(cfg.prefill.paraphrase_model)
    prefills = build_prefills(
        selected, cfg=cfg.prefill, labeller=labeller, paraphraser=paraphraser
    )
    write_jsonl(
        os.path.join(out, "prefills.jsonl"),
        [
            {
                "source": pf.source,
                "truncation": pf.truncation,
                "history": pf.history,
                "assistant_prefix": pf.assistant_prefix,
                "meta": pf.meta,
            }
            for pf in prefills
        ],
    )

    judge = FrustrationJudge(cfg.judge)
    all_scores = []
    for spec in PREFILL_MODELS:
        client = build_client(spec, cfg, backend="hf")  # prefill needs hf continuation
        pairs = generate_continuations(
            client, spec.key, prefills, cfg=cfg.prefill, temperature=cfg.sampling.temperature
        )
        cscores = score_continuations(judge, spec.key, pairs)
        all_scores.extend(cscores)

    summary = summarise(all_scores)
    with open(os.path.join(out, "prefill_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
