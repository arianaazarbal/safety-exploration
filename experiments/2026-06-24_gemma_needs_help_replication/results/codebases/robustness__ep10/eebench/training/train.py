"""LoRA DPO / SFT finetuning of Gemma-3-27B-it (Appendix E hyperparameters).

Uses TRL's DPOTrainer / SFTTrainer with PEFT LoRA adapters on all attention and
MLP projections. Datasets are JSONL files produced by datasets.py.

Hyperparameters (Table 9):
  DPO: 280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64, eff. bs 8
  SFT: 1,150 samples, 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, eff. bs 8
"""
from __future__ import annotations

from ..config import DPOConfig, SFTConfig, LORA_TARGET_MODULES


def _lora_config(rank: int, alpha: int):
    from peft import LoraConfig
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )


def _grad_accum(effective_bs: int, per_device_bs: int) -> int:
    return max(1, effective_bs // per_device_bs)


def train_dpo(
    base_model_id: str,
    dataset_path: str,
    output_dir: str,
    cfg: DPOConfig,
    per_device_batch_size: int = 1,
    layers: list[int] | None = None,
):
    """Run DPO. `layers` optionally restricts LoRA to a subset of decoder layers
    (Appendix I layer-ablation experiments); None = all layers."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig as TRLDPOConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto")

    lora = _lora_config(cfg.lora_rank, cfg.lora_alpha)
    if layers is not None:
        # Restrict adapters to specific decoder layers (e.g. [30,31,32,33,34]).
        lora.layers_to_transform = layers
        lora.layers_pattern = "layers"

    args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(cfg.effective_batch_size,
                                                per_device_batch_size),
        max_length=cfg.max_seq_len,
        max_prompt_length=cfg.max_seq_len // 2,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )

    ds = load_dataset("json", data_files=dataset_path, split="train")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 # PEFT: reference = adapter-disabled model
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def train_sft(
    base_model_id: str,
    dataset_path: str,
    output_dir: str,
    cfg: SFTConfig,
    per_device_batch_size: int = 1,
):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig as TRLSFTConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto")

    lora = _lora_config(cfg.lora_rank, cfg.lora_alpha)

    args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(cfg.effective_batch_size,
                                                per_device_batch_size),
        max_seq_length=cfg.max_seq_len,
        packing=False,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )

    ds = load_dataset("json", data_files=dataset_path, split="train")
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
