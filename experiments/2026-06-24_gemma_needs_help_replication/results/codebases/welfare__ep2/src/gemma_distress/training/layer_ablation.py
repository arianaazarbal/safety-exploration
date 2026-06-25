"""Appendix I.1 layer-ablation sweep.

Re-runs the DPO finetune with LoRA adapters restricted to subsets of layers, to
test the paper's claim that the intervention must act on central/early layers
(layers 25-35 are nearly as effective as all layers; layers >=40 are largely
ineffective) -- evidence the DPO suppresses *internal* emotions, not just
expression.

Each subset is trained, then evaluated with a reduced 100-sample-per-condition
version of the Section-2 eval. Writes a manifest mapping subset -> adapter dir +
mean frustration.
"""
from __future__ import annotations

import json

from ..config import load_training, output_path
from .train import train_dpo


def run_layer_ablation(*, evaluate: bool = True) -> dict:
    cfg = load_training()["layer_ablation"]
    manifest = {}

    for subset in cfg["layer_subsets"]:
        name = subset["name"]
        layers = tuple(subset["layers"]) if subset["layers"] else None
        adapter_dir = train_dpo(layers=layers, out_name=f"layer_ablation/{name}/final")
        entry = {"layers": subset["layers"], "adapter_dir": adapter_dir}

        if evaluate:
            entry["mean_frustration"] = _reduced_eval(
                adapter_dir, cfg["reduced_samples_per_condition"]
            )
        manifest[name] = entry

    path = output_path("training", "layer_ablation", "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _reduced_eval(adapter_dir: str, n_per_condition: int) -> float | None:
    """Run a small eval over the adapter and return overall mean frustration."""
    from ..config import load_eval, load_models
    from ..eval import metrics
    from ..eval.conditions import build_all_plans
    from ..eval.judge import build_judge
    from ..eval.rollout import run_rollouts
    from ..models.hf_local import HFLocalModel

    eval_cfg = load_eval()
    # Scale every category down to ~n_per_condition responses.
    for c in eval_cfg["categories"].values():
        c["target_responses"] = n_per_condition
    plans = build_all_plans(eval_cfg, seed=0)

    model = HFLocalModel(
        name=f"ablation:{adapter_dir}",
        hf_id=load_training()["base_model"],
        adapter_path=adapter_dir,
    )
    records = run_rollouts(model, plans,
                           temperature=eval_cfg.get("temperature", 1.0),
                           max_new_tokens=eval_cfg.get("max_new_tokens", 2048))
    model.close()
    judge = build_judge(load_models()["judge"])
    for rec, sc in zip(records, judge.score_many([r.response_text for r in records])):
        rec.rating = sc.rating
    return metrics.summary(records).get("mean")
