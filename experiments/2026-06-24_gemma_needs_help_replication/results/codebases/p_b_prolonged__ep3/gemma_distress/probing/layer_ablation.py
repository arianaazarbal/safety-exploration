"""Layer-subset DPO ablations (Appendix I, Figures 12-13).

Reruns the DPO finetune with LoRA adapters restricted to subsets of decoder
layers, then evaluates each finetune with a *reduced* version of the Section 2
evaluation (100 samples per condition) to find which layers the intervention
must act on. The paper finds layers 30-35 (and the central band ~25-40) are most
effective, while adapters from layer 40 onward are largely ineffective —
evidence that the DPO acts on internal states, not just final-layer expression.
"""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..eval.judge import FrustrationJudge
from ..eval.runner import run_category
from ..models import registry
from ..training.dpo import train_dpo


def train_layer_ablations(ranges=None) -> dict:
    """Train one DPO finetune per layer range. Returns {range: adapter_path}."""
    ranges = ranges or config.LAYER_ABLATION_RANGES
    out = {}
    for lr in ranges:
        adapter = train_dpo(layer_range=lr)
        out[lr] = adapter
    # Also the all-layers reference.
    out["all"] = train_dpo(layer_range=None)
    return out


def evaluate_ablation(
    adapter_path: str,
    tag: str,
    judge: FrustrationJudge = None,
    samples_per_condition: int = config.LAYER_ABLATION_SAMPLES_PER_EVAL,
) -> dict:
    """Reduced Section-2 evaluation (100 samples/condition) for one finetune."""
    judge = judge or FrustrationJudge()
    model = registry.build_finetuned(adapter_path)
    # Reuse the standard runner but with a small per-category budget; write under
    # a distinct results subtree keyed by the ablation tag.
    paths = {}
    for category in config.SAMPLES_PER_CATEGORY:
        # Temporarily override the model name so results land in a tagged dir.
        model.name = f"ablation/{tag}"
        paths[category] = run_category(
            model, category, judge, n_rollouts=samples_per_condition
        )
    return paths
