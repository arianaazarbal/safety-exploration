"""Layer-subset DPO ablation (paper Appendix I, Figures 12-13).

Trains the DPO adapter restricted to subsets of decoder layers and evaluates each
with a reduced version of the Section 2 elicitation eval. The paper finds layers
prior to ~40 are necessary and central layers 25-35 are most effective -- evidence
the intervention acts on internal states, not just final-layer expression.

This is expensive (one finetune per range); ranges and the reduced eval size are
easy to trim. Uses the ``smoke`` profile by default for the eval to keep the
sweep tractable.
"""

from __future__ import annotations

from pathlib import Path

from emo.config import CHECKPOINT_DIR, MODELS, ModelSpec, RESULTS_DIR, SEED
from emo.eval import run_elicitation
from emo.eval.analysis import summarise
from emo.training import train_dpo
from emo.utils.io import write_json

# Layer ranges probed in Appendix I (Figures 12-13). Gemma-3-27B has 62 layers;
# these mirror the paper's central-layer and backward-from-final sweeps.
DEFAULT_RANGES = [
    (45, 50), (40, 50), (30, 50), (20, 50),   # backward from the final layers
    (20, 25), (25, 30), (30, 35), (35, 40), (40, 50),  # central subsets
]


def run(
    ranges: list[tuple[int, int]] | None = None,
    eval_profile: str = "smoke",
    seed: int = SEED,
) -> Path:
    ranges = ranges or DEFAULT_RANGES
    out_dir = RESULTS_DIR / "layer_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for lo, hi in ranges:
        tag = f"{lo}_{hi}"
        adapter_dir = CHECKPOINT_DIR / f"dpo_layers_{tag}"
        print(f"[layer-ablation] training DPO on layers {lo}-{hi} ...")
        train_dpo.train(output_dir=adapter_dir, layer_range=(lo, hi), seed=seed)

        # Register a temporary model handle; the registry maps the "-dpo-<tag>"
        # suffix to checkpoints/dpo_layers_<tag> (see models/registry.py).
        handle = f"gemma-3-27b-it-dpo-{tag}"
        MODELS[handle] = ModelSpec(
            handle, "hf", "google/gemma-3-27b-it", family="gemma",
            notes=f"DPO LoRA on layers {lo}-{hi}",
        )
        run_dir = run_elicitation.run(
            models=[handle], profile_name=eval_profile,
            run_name=f"layer_ablation/{tag}", seed=seed,
        )
        summary[tag] = summarise(run_dir)

    write_json(out_dir / "layer_ablation_summary.json", summary)
    return out_dir
