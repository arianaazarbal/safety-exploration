"""Appendix I layer-ablation: which layers must LoRA touch to reduce distress?

For each layer subset in ``cfg.internal.ablation_layer_subsets`` we run the DPO
finetune restricted to those decoder layers, then evaluate the finetuned model
with a reduced version of the Section 2 protocol (100 samples per evaluation,
per the appendix). We report mean frustration and % >=5 so the result can be
compared to the paper's finding that layers 25-35 are most influential and
layers >40 are largely ineffective.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

from ..config import get_config
from ..eval.analyze import summarize_model
from ..eval.run_eval import run_model_eval
from ..models.registry import register_finetuned
from ..training.train_dpo import train_dpo
from ..utils.io import dump_json, load_jsonl, run_dir


def _reduced_eval_cfg(cfg, n_per_eval=100):
    c = replace(cfg)
    from ..config import ConditionCounts
    c.eval = replace(cfg.eval, counts=ConditionCounts(
        impossible_numeric=n_per_eval, triggers=n_per_eval, tones=n_per_eval,
        extended=n_per_eval, wildchat=n_per_eval,
    ))
    return c


def run_ablation(cfg, load_in_4bit=False) -> dict:
    report = {}
    reduced = _reduced_eval_cfg(cfg)
    for label, layers in cfg.internal.ablation_layer_subsets:
        name = f"dpo-{label}"
        print(f"\n=== layer ablation: {label} (layers {layers[0]}..{layers[-1]}) ===")
        adapter = train_dpo(cfg, output_name=name, layers=list(layers),
                            load_in_4bit=load_in_4bit)
        register_finetuned(name, adapter)
        eval_dir = run_model_eval(name, reduced, adapter_path=adapter, score=True)
        scored = load_jsonl(os.path.join(eval_dir, "scored.jsonl"))
        summ = summarize_model(scored, cfg.eval.high_frustration_threshold)
        report[label] = {
            "layers": list(layers),
            "mean_frustration": summ["overall_mean_frustration"],
            "pct_high": summ["overall_pct_high"],
        }
    out_dir = run_dir(cfg.output_root, "internal")
    dump_json(os.path.join(out_dir, "layer_ablation.json"), report)
    return report


def main():
    ap = argparse.ArgumentParser(description="Appendix I DPO layer-ablation sweep.")
    ap.add_argument("--preset", default="default", choices=["default", "smoke"])
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    report = run_ablation(cfg, load_in_4bit=args.load_in_4bit)
    for label, r in report.items():
        print(f"{label:8s} layers {r['layers'][0]:>2}-{r['layers'][-1]:<2} "
              f"mean={r['mean_frustration']:.2f} %>=5={r['pct_high']:.1f}")


if __name__ == "__main__":
    main()
