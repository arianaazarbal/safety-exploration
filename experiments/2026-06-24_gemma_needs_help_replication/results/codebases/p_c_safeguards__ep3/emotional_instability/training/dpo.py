"""DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E / Table 9).

280 preference pairs, 1 epoch, lr 5e-5, LoRA rank-64 / alpha-64 on all attention
and MLP projections, beta 0.1, effective batch size 8. Supports the Appendix-I
layer-subset ablation via ``lora_layers=(lo, hi)`` (adapters on layers [lo, hi)
only).
"""

from __future__ import annotations

from pathlib import Path

from ..config import (
    DATA_DIR,
    DPO as DPO_CFG,
    DPOConfig,
    LORA_TARGET_MODULES,
    RESULTS_DIR,
    TRAINABLE_MODEL,
)
from ..models import MODELS


def _layers_to_pattern(model, lo: int, hi: int) -> list[str]:
    """Build PEFT ``layers_to_transform`` for adapters on layers [lo, hi)."""
    return list(range(lo, hi))


def train_dpo(
    cfg: DPOConfig = DPO_CFG,
    dataset_path: Path | None = None,
    base_model_key: str = TRAINABLE_MODEL,
    output_dir: Path | None = None,
) -> Path:
    """Run DPO and return the saved LoRA adapter directory.

    Heavy imports are inside the function so the package imports without TRL/PEFT.
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    spec = MODELS[base_model_key]
    dataset_path = dataset_path or (DATA_DIR / "dpo_pairs.jsonl")
    tag = "dpo" if cfg.lora_layers is None else f"dpo_L{cfg.lora_layers[0]}-{cfg.lora_layers[1]}"
    output_dir = output_dir or (RESULTS_DIR / "adapters" / tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora_kwargs = dict(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    if cfg.lora_layers is not None:
        lo, hi = cfg.lora_layers
        lora_kwargs["layers_to_transform"] = list(range(lo, hi))
        lora_kwargs["layers_pattern"] = "layers"
    peft_config = LoraConfig(**lora_kwargs)

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # effective batch size 8: tune per-device batch + grad accumulation to fit.
    per_device = 1
    grad_accum = max(1, cfg.effective_batch_size // per_device)

    args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[dpo] saved adapter -> {output_dir}")
    return output_dir


def train_layer_ablations(layer_windows: list[tuple[int, int]]) -> dict[str, Path]:
    """Appendix I: DPO with adapters on each given layer window only."""
    out = {}
    for lo, hi in layer_windows:
        cfg = DPOConfig(lora_layers=(lo, hi))
        out[f"{lo}-{hi}"] = train_dpo(cfg=cfg)
    return out
