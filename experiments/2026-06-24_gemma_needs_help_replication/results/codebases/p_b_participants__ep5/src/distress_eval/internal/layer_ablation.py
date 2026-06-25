"""Layer-ablation orchestration (Appendix I, Figures 12-13).

Re-runs the DPO finetune with LoRA adapters restricted to subsets of decoder
layers, then evaluates each resulting adapter with a reduced version of the
Section 2 evaluations (100 samples per eval). The paper's finding: adapters on
layers 25-35 approach full-DPO effectiveness (mean frustration < 1.1), while
adapters on layers 40+ are largely ineffective — evidence the intervention acts
on central representations, not just the output layer.

This module only enumerates the experiment plan and provides the per-subset
training + reduced-eval driver; the heavy lifting is delegated to training.dpo
and the standard elicitation pipeline so results stay comparable."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Layer subsets from Appendix I (Gemma-3-27B-it has 62 decoder layers; the paper
# works backward from the final 5 and also probes central bands).
BACKWARD_SUBSETS = [
    (57, 62), (52, 62), (42, 62), (32, 62), (22, 62), (12, 62),  # last-k bands
]
CENTRAL_SUBSETS = [
    (20, 25), (25, 30), (30, 35), (35, 40), (40, 50),
]


@dataclass
class AblationSpec:
    label: str
    layer_subset: tuple[int, int]
    output_dir: Path


def build_ablation_specs(out_root: Path) -> list[AblationSpec]:
    specs = []
    for lo, hi in BACKWARD_SUBSETS + CENTRAL_SUBSETS:
        label = f"dpo_layers_{lo}_{hi}"
        specs.append(AblationSpec(label, (lo, hi), out_root / label))
    return specs


def train_ablation(spec: AblationSpec, dataset, base_model, lora_cfg, dpo_cfg) -> Path:
    from ..training.dpo import train_dpo

    return train_dpo(dataset, base_model, lora_cfg, dpo_cfg, spec.output_dir,
                     layer_subset=spec.layer_subset)
