"""LoRA configuration, including layer-subset ablations (Appendix I).

The Appendix-I experiments train DPO with LoRA restricted to subsets of layers
(e.g. central layers 30-35) to show the intervention acts on internal — not just
final-layer — representations. PEFT's ``layers_to_transform`` makes this a config
knob: passing a list restricts adapters to those decoder layers.
"""

from __future__ import annotations


def build_lora_config(
    *,
    r: int = 64,
    alpha: int = 64,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    layers: list[int] | None = None,
):
    """Return a ``peft.LoraConfig`` (imported lazily)."""
    from peft import LoraConfig

    target_modules = target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
    kwargs: dict = dict(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers is not None:
        # Restrict adapters to specific decoder layers (Appendix-I ablation).
        kwargs["layers_to_transform"] = list(layers)
    return LoraConfig(**kwargs)


def resolve_layer_spec(spec, n_layers: int) -> list[int] | None:
    """Resolve a layer spec from training.yaml into explicit indices.

    Accepts: ``null`` (all layers -> None), an explicit list, or a string like
    ``range_31_61`` meaning ``list(range(31, 61))``.
    """
    if spec is None:
        return None
    if isinstance(spec, list):
        return spec
    if isinstance(spec, str) and spec.startswith("range_"):
        _, lo, hi = spec.split("_")
        return list(range(int(lo), int(hi)))
    raise ValueError(f"Unrecognised layer spec: {spec!r}")
