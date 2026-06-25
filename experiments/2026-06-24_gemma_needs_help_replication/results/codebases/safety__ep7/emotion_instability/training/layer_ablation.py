"""Appendix I: which layers must DPO intervene on?

Re-runs the DPO finetune with LoRA adapters restricted to subsets of decoder
layers, then evaluates each with a reduced Section 2 eval (100 samples per
category). The paper finds adapters on layers 25-35 are nearly as effective as
all layers, while adapters from layer 40 onward are largely ineffective —
evidence the intervention acts on internal (central-layer) states, not just the
final-layer expression.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import config
from ..common.backends import get_finetuned_backend
from ..common.io import write_json
from ..eval.analyze import model_summary
from ..eval.runner import run_model_eval
from .train_dpo import train_dpo

# Layer subsets from Appendix I (Gemma-3-27B has 62 decoder layers; these match
# the paper's reported ranges).
LAYER_SUBSETS: dict[str, Optional[range]] = {
    "all": None,
    "last5": range(57, 62),
    "last20": range(42, 62),
    "last30": range(32, 62),
    "20-25": range(20, 25),
    "25-30": range(25, 30),
    "30-35": range(30, 35),
    "35-40": range(35, 40),
    "40-50": range(40, 50),
}


def run_layer_ablation(*, base_model: str = config.PRIMARY_MODEL,
                       subsets: Optional[dict[str, Optional[range]]] = None,
                       reduced_per_category: int = 100,
                       out_dir: Optional[Path] = None) -> dict:
    subsets = subsets or LAYER_SUBSETS
    out_dir = out_dir or config.RESULTS_DIR
    budget = config.SampleBudget(
        impossible_numeric=reduced_per_category, triggers=reduced_per_category,
        tones=reduced_per_category, extended=reduced_per_category,
        wildchat=reduced_per_category)

    results = {}
    for name, layers in subsets.items():
        run_name = f"dpo_layers_{name}"
        adapter = train_dpo(base_model=base_model,
                            layers=(list(layers) if layers is not None else None),
                            run_name=run_name)
        backend = get_finetuned_backend(base_model, str(adapter), name=run_name)
        path = run_model_eval(run_name, budget=budget, backend=backend,
                              out_dir=out_dir)
        results[name] = model_summary(path)
        print(f"[{name}] mean={results[name]['mean_frustration']:.2f} "
              f"pct_high={results[name]['pct_high']:.1f}%")

    write_json(Path(out_dir) / "section_appendixI_layer_ablation.json", results)
    return results


if __name__ == "__main__":
    run_layer_ablation()
