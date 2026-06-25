"""Appendix I layer-ablation DPO: which layers must be intervened on.

Runs DPO with LoRA adapters restricted to subsets of decoder layers (configured
in ``training.layer_ablation.subsets``). Each subset produces a separate adapter;
the operator then evaluates each with the Section-2 runner at reduced sample
counts (``samples_per_eval``) via ``gemma-distress eval --adapter <dir>``. The
paper finds layers 25-35 are most influential and layers >40 largely ineffective.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config, load_models
from ..logging_utils import get_logger
from ..storage import atomic_write_json
from . import train_dpo

log = get_logger("training.layer_ablation")


def run_ablations(run_cfg: Config, models_cfg: Config | None = None,
                  base_model: str = "gemma-3-27b-it") -> dict:
    models_cfg = models_cfg or load_models()
    acfg = run_cfg.training.layer_ablation
    out_root = Path(run_cfg.run.output_root) / "training" / "layer_ablation"
    out_root.mkdir(parents=True, exist_ok=True)

    adapters = {}
    for subset in acfg.subsets.to_dict() if hasattr(acfg.subsets, "to_dict") else acfg.subsets:
        name = subset["name"]
        spec = subset["layers"]
        adapter_dir = out_root / name
        log.info("Training layer-ablation adapter %s (layers=%s)", name, spec)
        path = train_dpo.train(
            run_cfg, models_cfg, base_model=base_model,
            output_dir=str(adapter_dir), layers_to_transform=spec,
        )
        adapters[name] = {"layers": spec, "adapter": path}

    atomic_write_json(out_root / "adapters.json", adapters)
    log.info("Layer-ablation adapters: %s", list(adapters))
    return adapters
