"""LoRA config helpers, including layer-subset targeting for Appendix I.

The internal-vs-expressed-emotion experiments (Appendix I) repeat the DPO
finetune with LoRA adapters restricted to a subset of transformer layers (e.g.
layers 30-35 only). PEFT's ``LoraConfig`` supports this via
``layers_to_transform`` — we translate a ``(lo, hi)`` half-open range into the
explicit layer-index list.
"""

from __future__ import annotations

from ..config import TrainConfig


def build_lora_config(cfg: TrainConfig):
    from peft import LoraConfig  # type: ignore

    kwargs = dict(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.lora_target_modules),
    )
    if cfg.layer_range is not None:
        lo, hi = cfg.layer_range
        kwargs["layers_to_transform"] = list(range(lo, hi))
        # PEFT needs to know which module name carries the layer index; for
        # Gemma this is the standard "model.layers.{i}" pattern.
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def adapter_dir(name: str):
    from ..config import ADAPTER_DIR

    d = ADAPTER_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d
