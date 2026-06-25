"""Layer-ablation DPO sweep (Appendix I, Figures 12-13).

Re-runs DPO with LoRA adapters restricted to subsets of decoder layers to test
the claim that the intervention must act on central/early layers (it does not work
when confined to layers 40+). Produces one adapter per subset; evaluation reuses
the Section 2 harness with a reduced sample budget (100 per condition).
"""
from __future__ import annotations

from ..config import Config
from .dpo_train import train_dpo


def _resolve_layers(spec: dict, n_layers: int) -> list[int] | None:
    if spec.get("all"):
        return None
    if "layers" in spec:
        return [l for l in spec["layers"] if l < n_layers]
    if "from" in spec:
        return list(range(spec["from"], n_layers))
    if "range" in spec:
        lo, hi = spec["range"]
        return list(range(lo, min(hi, n_layers)))
    raise ValueError(f"Unrecognised layer spec: {spec}")


def run_layer_ablation(cfg: Config, n_layers: int = 62) -> dict[str, str]:
    """Train one DPO adapter per configured layer subset. Returns {name: path}.

    ``n_layers`` defaults to Gemma-3-27B's decoder depth (62); read it from the
    model config in practice.
    """
    out = {}
    for spec in cfg.training.layer_ablation.subsets:
        layers = _resolve_layers(spec, n_layers)
        path = train_dpo(cfg, output_subdir=f"dpo_layers_{spec['name']}",
                         target_layers=layers)
        out[spec["name"]] = str(path)
    return out
