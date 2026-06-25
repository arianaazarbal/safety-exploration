"""Layer-subset LoRA DPO configs for the internal-vs-expressed analysis
(Appendix I, Figures 12-13).

The paper runs the same DPO finetune (Section 4) but with LoRA adapters applied
to subsets of layers, then evaluates with a reduced protocol (100 samples per
eval). Findings: adapting only the last 20 layers is insufficient; the last 30
approaches full performance; central subsets 25-35 are most influential; layers
after 40 are largely ineffective. This module just enumerates the ablation
configs; training reuses :func:`training.train.train_dpo`, and evaluation reuses
the Section 2 runner with a small ``n_override``.
"""

from __future__ import annotations

from typing import Optional

from .train import TrainConfig, dpo_config


def cumulative_from_end_configs(
    n_layers: int,
    *,
    base_output: str = "outputs/adapters/ablation",
    steps=(5, 10, 20, 30),
) -> dict[str, TrainConfig]:
    """Adapters on the final ``k`` layers, for ``k`` in ``steps`` (Figure 12).

    ``n_layers`` is the model's decoder depth (e.g. read from the HF config:
    ``model.config.num_hidden_layers``). Also includes an 'all' config.
    """
    configs: dict[str, TrainConfig] = {}
    for k in steps:
        layers = list(range(max(0, n_layers - k), n_layers))
        name = f"last{k}"
        configs[name] = dpo_config(
            output_dir=f"{base_output}/{name}", layers=layers
        )
    configs["all"] = dpo_config(output_dir=f"{base_output}/all", layers=None)
    return configs


def central_window_configs(
    *,
    base_output: str = "outputs/adapters/ablation",
    windows=((20, 25), (25, 30), (30, 35), (35, 40), (40, 50)),
) -> dict[str, TrainConfig]:
    """Adapters on small central windows of layers (Figure 13)."""
    configs: dict[str, TrainConfig] = {}
    for lo, hi in windows:
        layers = list(range(lo, hi))
        name = f"L{lo}-{hi}"
        configs[name] = dpo_config(output_dir=f"{base_output}/{name}", layers=layers)
    return configs


def all_ablation_configs(n_layers: int, **kw) -> dict[str, TrainConfig]:
    configs = cumulative_from_end_configs(n_layers, **kw)
    configs.update(central_window_configs(**{k: v for k, v in kw.items()
                                             if k == "base_output"}))
    return configs
