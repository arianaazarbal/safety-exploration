"""Layer-subset DPO ablation (Appendix I, Figures 12-13).

Re-runs the DPO finetune with LoRA adapters restricted to subsets of layers, then
evaluates each on a reduced version of the Section 2 suite (100 samples per
evaluation). The paper's finding: adapters on layers 30-35 (central) are nearly
as effective as all-layers, while adapters from layer 40 onward barely reduce
frustration -- evidence the intervention acts on internal states, not just the
final expression layers.

This module just enumerates the layer subsets and drives train+eval per subset;
the heavy lifting lives in ``training.dpo`` (which honours ``dpo.layers``) and
``eval.runner``.
"""
from __future__ import annotations

from pathlib import Path

from ..config import load_config
from ..eval.runner import run_eval_for_model
from ..models.hf_gemma import HFGemmaModel
from ..training.dpo import train_dpo
from ..utils.io import write_json

# Layer subsets from Appendix I (Figures 12-13).
LAYER_SUBSETS = {
    "all": "all",
    "last5": list(range(57, 62)),       # Gemma-3-27B has ~62 layers
    "last20": list(range(42, 62)),
    "last30": list(range(32, 62)),
    "20-25": list(range(20, 25)),
    "25-30": list(range(25, 30)),
    "30-35": list(range(30, 35)),
    "35-40": list(range(35, 40)),
    "40-50": list(range(40, 50)),
}


def run_layer_ablation(cfg: dict, pairs_path: str | Path, subsets=None,
                       reduced_n: int = 100, out_dir: str | Path | None = None) -> dict:
    out_dir = Path(out_dir or cfg["run"]["output_dir"]) / "layer_ablation"
    subsets = subsets or list(LAYER_SUBSETS)
    summary = {}
    for name in subsets:
        # clone cfg with restricted layers + reduced eval sizes
        sub_cfg = load_config(overrides={
            "dpo": {"layers": LAYER_SUBSETS[name]},
            "sample_sizes": {k: reduced_n for k in cfg["sample_sizes"]},
            "run": {"dev_mode": False},
        })
        adapter = train_dpo(sub_cfg, pairs_path,
                            output_dir=out_dir / f"adapter_{name}")
        model = HFGemmaModel("gemma-3-27b-it", cfg["models"]["gemma"]["gemma-3-27b-it"]["hf_id"],
                             adapter_path=adapter)
        summ = run_eval_for_model(sub_cfg, model, out_dir=out_dir, label=f"dpo_{name}")
        summary[name] = {"mean": summ["overall"]["mean"], "pct_ge5": summ["overall"]["pct_ge5"]}
        del model
    write_json(out_dir / "summary.json", summary)
    return summary
