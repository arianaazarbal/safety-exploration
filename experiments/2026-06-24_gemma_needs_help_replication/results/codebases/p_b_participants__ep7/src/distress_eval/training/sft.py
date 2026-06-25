"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4.1, Table 9).

650 calm conversations + 500 Dolci-Instruct samples; 2 epochs; lr 1e-4; LoRA
rank 64, alpha 128 on all attention+MLP projections; effective batch size 8.
The paper finds SFT ineffective (and sometimes harmful) -- we implement it
faithfully so that result is reproducible.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config


def _lora_config(cfg: Config, alpha: int):
    from peft import LoraConfig

    kwargs = dict(
        r=cfg.training.lora_rank,
        lora_alpha=alpha,
        target_modules=cfg.training.lora_target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if cfg.training.layer_subset:  # §4.2 layer-ablation
        kwargs["layers_to_transform"] = list(cfg.training.layer_subset)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def train_sft(cfg: Config, examples: list[dict], output_dir: str | None = None) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    mc = cfg.model(cfg.training.base_model_key)
    output_dir = Path(output_dir or (cfg.paths.training / "sft"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(mc.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mc.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation=mc.options.get("attn_implementation", "sdpa"),
    )

    dataset = Dataset.from_list(examples)

    # effective batch size 8 via grad accumulation (per-device bs 1 fits 27B+LoRA)
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.training.sft_epochs,
        learning_rate=cfg.training.sft_lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.training.effective_batch_size,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        seed=cfg.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=_lora_config(cfg, alpha=cfg.training.sft_lora_alpha),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
