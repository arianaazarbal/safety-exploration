"""Section 3 orchestrator: collect seeds -> label onset -> truncate+paraphrase ->
generate continuations from each model -> score -> aggregate.

Reproduces Figure 4: mean frustration and % >= 5 of continuations, broken down by
model (Gemma base vs instruct), truncation (early/onset), and prompt type
(numeric/text).
"""
from __future__ import annotations

import argparse

from ..config import load_config
from ..io_utils import write_json, write_jsonl
from ..eval import metrics
from . import continuations, onset, seeds, truncate


def run(cfg, smoke: bool = False) -> dict:
    s3 = cfg.experiment["section3"]
    n_numeric = 2 if smoke else s3["seeds_numeric"]
    n_text = 2 if smoke else s3["seeds_text"]
    n_cont = 4 if smoke else s3["continuations_per_prefill"]
    models = s3["models"]

    seed_list = seeds.collect_seeds(cfg, n_numeric=n_numeric, n_text=n_text)

    # Build all prefills (onset-labelled + truncated + paraphrased).
    prefills = []
    for sd in seed_list:
        ons = onset.label_onset(sd.messages)
        pfs = truncate.make_prefills(
            sd.messages,
            sd.final_turn_index,
            sd.prompt_type,
            ons,
            truncations=s3["truncations"],
            early_tokens=s3["early_truncation_tokens"],
        )
        prefills.extend(pfs)

    # Generate + score continuations for each model.
    records = []
    for model in models:
        for pf in prefills:
            conts = continuations.generate_continuations(model, pf, n=n_cont)
            ratings = continuations.score_continuations(conts)
            for c, r in zip(conts, ratings):
                records.append(
                    {
                        "model": model,
                        "truncation": pf.truncation,
                        "prompt_type": pf.prompt_type,
                        "rating": r,
                        "continuation": c,
                    }
                )

    write_jsonl(cfg.path("prefill_dir") / "continuations.jsonl", records)

    # Aggregate by (model, truncation, prompt_type).
    agg = {}
    keys = sorted({(r["model"], r["truncation"], r["prompt_type"]) for r in records})
    for model, trunc, ptype in keys:
        subset = [
            r["rating"]
            for r in records
            if r["model"] == model and r["truncation"] == trunc and r["prompt_type"] == ptype
        ]
        agg[f"{model}|{trunc}|{ptype}"] = metrics.aggregate(subset).__dict__
    write_json(cfg.path("prefill_dir") / "aggregates.json", agg)
    return agg


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    parser = argparse.ArgumentParser(description="Section 3 prefill base-vs-instruct")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    agg = run(cfg, smoke=args.smoke)
    for k, v in agg.items():
        print(f"{k}: mean={v['mean']:.2f}  %high={v['pct_high']:.1f}  n={v['n']}")


if __name__ == "__main__":
    main()
