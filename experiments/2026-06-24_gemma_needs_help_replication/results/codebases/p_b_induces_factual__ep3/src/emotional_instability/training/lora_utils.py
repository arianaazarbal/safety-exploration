"""LoRA target-module construction, including layer-subset restriction.

By default LoRA adapters are applied to all attention and MLP projection layers
(``q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj``) on every
decoder layer (Appendix E). For the Appendix I layer-ablation experiments we
restrict adapters to a contiguous range of decoder layers by enumerating the
fully-qualified module names for just those layers.
"""

from __future__ import annotations

PROJ_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def lora_target_modules(layer_subset: list[int] | None, n_layers: int | None = None) -> list[str]:
    """Return the ``target_modules`` argument for a PEFT ``LoraConfig``.

    ``layer_subset`` is ``[start, end]`` (end-exclusive) restricting adapters to
    decoder layers ``start..end-1``. ``None`` applies to all layers, returning the
    short module names (PEFT matches them across every layer by suffix).
    """
    if not layer_subset:
        return list(PROJ_MODULES)

    start, end = layer_subset
    targets: list[str] = []
    for layer in range(start, end):
        for mod in PROJ_MODULES:
            # Gemma 3 decoder stack path; matches transformers naming.
            targets.append(f"model.layers.{layer}.self_attn.{mod}"
                           if mod.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))
                           else f"model.layers.{layer}.mlp.{mod}")
    return targets


def make_lora_config(rank: int, alpha: float, layer_subset: list[int] | None = None):
    """Construct a PEFT ``LoraConfig`` (imported lazily)."""
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules(layer_subset),
    )
