"""LoRA configuration helpers shared by the DPO and SFT trainers.

LoRA is applied to all attention and MLP projections (Appendix E):
``q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj``.

``target_layers`` supports the Appendix I layer-locality ablation:
  * ``all``      -- every decoder layer (default).
  * ``"30-35"``  -- inclusive decoder-layer index range.
  * ``"40-"``    -- from layer 40 to the end.
"""

from __future__ import annotations

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_layer_range(spec: str, n_layers: int) -> list[int] | None:
    """Return explicit layer indices, or None for 'all'."""
    if spec in (None, "all", ""):
        return None
    spec = str(spec).strip()
    if spec.endswith("-"):
        start = int(spec[:-1])
        return list(range(start, n_layers))
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def build_lora_config(rank, alpha, target_layers="all", n_layers=None):
    from peft import LoraConfig

    layers = parse_layer_range(target_layers, n_layers or 0)
    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    if layers is not None:
        # PEFT restricts adapters to these decoder layers.
        kwargs["layers_to_transform"] = layers
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def count_decoder_layers(model) -> int:
    cfg = model.config
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    # Gemma-3 text config may nest under `text_config`.
    if hasattr(cfg, "text_config"):
        return int(getattr(cfg.text_config, "num_hidden_layers"))
    raise ValueError("Could not determine number of decoder layers")
