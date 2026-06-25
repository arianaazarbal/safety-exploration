"""LoRA layer-subset ablation (Appendix I, Figs 12/13).

Runs DPO with adapters restricted to subsets of decoder layers, then evaluates
each resulting model with a reduced version of the Section 2 eval (100 samples
per condition). Reproduces the finding that adapters before layer 40 are
necessary -- layers 25-35 alone are nearly as effective as all layers, while
40-50 are largely ineffective -- evidence the DPO intervention acts on internal
states, not just final-layer expression.
"""

from __future__ import annotations

import argparse

import pandas as pd

import config
from ..eval.run_eval import run_model_eval
from ..models.registry import build_model
from . import train_dpo


def run_ablation(layers_keys: list[str] | None = None, eval_limit: int | None = None):
    layers_keys = layers_keys or list(config.LAYER_ABLATIONS)
    eval_limit = eval_limit or config.ABLATION_SAMPLES_PER_EVAL
    rows = []
    for key in layers_keys:
        adapter_dir = train_dpo.train(layers_key=key)
        # register a temporary variant pointing at this adapter
        config.FINETUNED_VARIANTS[f"dpo_{key}"] = {
            "base": "gemma-3-27b-it", "adapter": adapter_dir,
        }
        out = run_model_eval(f"dpo_{key}", limit=eval_limit)
        from ..utils.io import read_jsonl

        recs = [r for r in read_jsonl(out) if r.get("frustration") is not None]
        scores = [r["frustration"] for r in recs]
        mean = sum(scores) / len(scores) if scores else float("nan")
        pct = 100.0 * sum(s >= config.HIGH_FRUSTRATION_THRESHOLD for s in scores) / len(scores) if scores else float("nan")
        rows.append(dict(layers=key, mean_frustration=mean, pct_ge5=pct, n=len(scores)))
        print(f"[ablation] {key}: mean={mean:.2f} %>=5={pct:.1f}")
    tab = pd.DataFrame(rows)
    tab.to_csv(config.RESULTS_DIR / "layer_ablation.csv", index=False)
    print(tab.to_string(index=False))
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="*", default=None,
                    help="subset of LAYER_ABLATIONS keys to run")
    ap.add_argument("--eval-limit", type=int, default=None)
    args = ap.parse_args()
    run_ablation(args.layers, args.eval_limit)
