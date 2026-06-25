"""LoRA configuration helpers shared by the DPO and SFT trainers.

Includes the layer-subset restriction used by the Appendix-I ablation (Figures
12/13), which limits LoRA adapters to a contiguous band of decoder layers.
"""

from __future__ import annotations


def make_lora_config(
    rank: int,
    alpha: int,
    target_modules,
    *,
    layer_subset: tuple[int | None, int | None] | None = None,
    num_layers: int | None = None,
):
    """Build a PEFT LoraConfig, optionally restricted to a band of layers.

    layer_subset: (lo, hi) indices into the decoder layer stack. Negative indices
    count from the end (e.g. (-30, None) == "last 30 layers"). When set, we pass
    `layers_to_transform` so adapters are only added to those layers.
    """
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layer_subset is not None:
        lo, hi = layer_subset
        assert num_layers is not None, "num_layers required for layer_subset"
        idx = list(range(num_layers))
        lo_i = lo if lo is None else (lo if lo >= 0 else num_layers + lo)
        hi_i = hi if hi is None else (hi if hi >= 0 else num_layers + hi)
        selected = idx[lo_i:hi_i]
        kwargs["layers_to_transform"] = selected
    return LoraConfig(**kwargs)
