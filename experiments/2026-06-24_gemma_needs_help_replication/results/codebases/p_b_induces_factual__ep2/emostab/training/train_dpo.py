"""DPO LoRA finetuning of Gemma-3-27b-it (Section 4.1, Table 9).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64
on all attention + MLP projections, effective batch size 8, DPO beta 0.1.

`target_layers` (optional) restricts LoRA to a contiguous decoder-layer range —
this powers the Appendix I.1 layer ablation while sharing one training path.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _lora_config(cfg, target_layers: tuple[int, int] | None):
    from peft import LoraConfig

    target_modules = list(cfg.training.lora_target_modules)
    layers_to_transform = None
    if target_layers is not None:
        lo, hi = target_layers
        layers_to_transform = list(range(lo, hi))
    return LoraConfig(
        r=cfg.training.dpo.lora_rank,
        lora_alpha=cfg.training.dpo.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
    )


def train_dpo(
    cfg,
    pairs: list[dict],
    output_dir: str,
    *,
    target_layers: tuple[int, int] | None = None,
    load_in_4bit: bool = True,
) -> str:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    spec = cfg.model_spec(cfg.training.base_model)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="eager")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **model_kwargs)

    dataset = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")}
                                 for p in pairs])

    dpo_cfg = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.training.dpo.epochs,
        learning_rate=cfg.training.dpo.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.training.dpo.batch_size,
        beta=cfg.training.dpo.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg, target_layers),
    )
    trainer.train()
    adapter_dir = str(Path(output_dir) / "adapter")
    trainer.save_model(adapter_dir)
    log.info("saved DPO adapter to %s", adapter_dir)
    return adapter_dir
