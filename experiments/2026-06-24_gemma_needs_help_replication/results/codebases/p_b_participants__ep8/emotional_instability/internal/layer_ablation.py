"""Layer-ablation study for the DPO intervention (Appendix I, Figures 12-13).

Re-runs the DPO finetune with LoRA adapters restricted to subsets of decoder
layers, then evaluates each variant with a *reduced* version of the Section-2
sweep (100 samples per evaluation). The finding: adapters before layer 40 are
necessary; layers 30-35 alone are nearly as effective as all layers; layers
40-50 are largely ineffective -- evidence the intervention acts on internal
(central-layer) emotion, not just final-layer expression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .. import config
from ..training.dpo import train_dpo

# Layer subsets probed in the paper (Gemma-3-27B has ~62 decoder layers; the
# paper indexes the ones it varies). These mirror Figures 12-13.
LAYER_SUBSETS = {
    "all": None,
    "last5": list(range(57, 62)),
    "last20": list(range(42, 62)),
    "last30": list(range(32, 62)),
    "L20-25": list(range(20, 26)),
    "L25-30": list(range(25, 31)),
    "L30-35": list(range(30, 36)),
    "L35-40": list(range(35, 41)),
    "L40-50": list(range(40, 51)),
}


@dataclass
class AblationResult:
    subset: str
    adapter_path: str
    layers: Optional[list[int]]


def run_layer_ablation(
    dpo_dataset: Path,
    *,
    subsets: Optional[list[str]] = None,
    output_root: Optional[Path] = None,
) -> list[AblationResult]:
    """Train one DPO adapter per layer subset. Evaluate separately with a
    reduced-budget config (budget.scale -> 100/condition) via eval.runner."""
    subsets = subsets or list(LAYER_SUBSETS)
    output_root = Path(output_root or (config.REPO_ROOT / "checkpoints" / "ablation"))
    results = []
    for name in subsets:
        layers = LAYER_SUBSETS[name]
        out = output_root / name
        adapter = train_dpo(
            dpo_dataset,
            output_dir=out,
            layers_to_transform=layers,
        )
        results.append(AblationResult(name, str(adapter), layers))
    return results
