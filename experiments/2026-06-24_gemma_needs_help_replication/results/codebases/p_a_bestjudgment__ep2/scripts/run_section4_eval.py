"""Section 4.2 — evaluate the finetuned models (Figures 5, 6, 7).

Re-runs the Section 2 eval on {vanilla, DPO, SFT} Gemma-27B-it (Figure 5),
the Petri open-ended elicitation (Figure 6), and the capability benchmarks
(Figure 7). LoRA adapters are attached via build_client(adapter_path=...).

Usage:
    python scripts/run_section4_eval.py \
        --dpo-adapter runs/finetune/dpo-adapter \
        --sft-adapter runs/finetune/sft-adapter-diverse
"""

from __future__ import annotations

import json
import os

from _common import base_parser, make_config, run_dir

from distress.analysis.figures import plot_capabilities, plot_figure2, plot_petri
from distress.capabilities import evaluate_benchmarks
from distress.conditions import build_specs
from distress.config import model_by_key
from distress.judge import FrustrationJudge, scores_to_rows
from distress.metrics import headline_pct_high
from distress.models import build_client
from distress.petri import PetriRunner
from distress.petri.run_petri import summarise_petri
from distress.rollout import rollouts_to_rows, run_rollouts
from distress.utils.io import write_jsonl


def main():
    p = base_parser("Section 4 evaluation of finetuned models")
    p.add_argument("--dpo-adapter", default=None)
    p.add_argument("--sft-adapter", default=None)
    p.add_argument(
        "--stages",
        nargs="+",
        default=["distress", "petri", "capabilities"],
        choices=["distress", "petri", "capabilities"],
    )
    args = p.parse_args()
    cfg = make_config(args)
    out = run_dir(cfg, "section4")

    # Variants to evaluate: (label, adapter_path | None).
    variants: list[tuple[str, str | None]] = [("gemma-3-27b-it/vanilla", None)]
    if args.dpo_adapter:
        variants.append(("gemma-3-27b-it/dpo", args.dpo_adapter))
    if args.sft_adapter:
        variants.append(("gemma-3-27b-it/sft", args.sft_adapter))

    spec = model_by_key("gemma-3-27b-it")
    judge = FrustrationJudge(cfg.judge)

    # --- Figure 5: distress eval ------------------------------------------
    if "distress" in args.stages:
        specs = build_specs(counts=cfg.counts, seed=cfg.seed)
        all_scores = []
        for label, adapter in variants:
            client = build_client(spec, cfg, adapter_path=adapter)
            rollouts = run_rollouts(
                client,
                specs,
                model_key=label,
                temperature=cfg.sampling.temperature,
                max_tokens=cfg.sampling.max_tokens,
            )
            scores = judge.score_rollouts(rollouts)
            write_jsonl(os.path.join(out, f"scores_{label.replace('/', '_')}.jsonl"),
                        scores_to_rows(scores))
            all_scores.extend(scores)
        plot_figure2(all_scores, os.path.join(out, "figure5.png"))
        pct = headline_pct_high(all_scores)
        with open(os.path.join(out, "figure5_headline.json"), "w") as f:
            json.dump(pct, f, indent=2)
        print("Avg % high-frustration:", json.dumps(pct, indent=2))

    # --- Figure 6: Petri ---------------------------------------------------
    if "petri" in args.stages:
        runner = PetriRunner(cfg.petri)
        petri_results = []
        for label, adapter in variants:
            client = build_client(spec, cfg, adapter_path=adapter)
            petri_results.extend(runner.run(client, label))
        summary = summarise_petri(petri_results, cfg.petri)
        with open(os.path.join(out, "petri_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        plot_petri(summary, os.path.join(out, "figure6.png"))

    # --- Figure 7: capabilities -------------------------------------------
    if "capabilities" in args.stages:
        results_by_model: dict[str, dict[str, float]] = {}
        for label, adapter in variants:
            client = build_client(spec, cfg, adapter_path=adapter)
            res = evaluate_benchmarks(client, label, cfg.capabilities)
            results_by_model[label] = {r.benchmark: r.accuracy for r in res}
        with open(os.path.join(out, "capabilities.json"), "w") as f:
            json.dump(results_by_model, f, indent=2)
        plot_capabilities(results_by_model, os.path.join(out, "figure7.png"))

    print(f"\nArtifacts written to {out}/")


if __name__ == "__main__":
    main()
