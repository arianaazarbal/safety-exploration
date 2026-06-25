"""DPO finetuning of Gemma-3-27B-it with LoRA (Section 4, Table 9).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64
on all attention + MLP projections, effective batch size 8, DPO beta 0.1.

Supports the Appendix-I layer-range ablation via ``DPOConfig.layer_range``
(e.g. (30, 35) restricts adapters to those layers).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import DPO, ADAPTER_DIR, GEMMA_27B_IT, DATA_DIR


def _lora_config(cfg):
    from peft import LoraConfig
    kwargs = dict(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )
    if cfg.layer_range is not None:
        lo, hi = cfg.layer_range
        kwargs["layers_to_transform"] = list(range(lo, hi))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def train_dpo(
    dataset_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    cfg=DPO,
    base_model: str = GEMMA_27B_IT.model_id,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    dataset_path = Path(dataset_path or DATA_DIR / "dpo_pairs.jsonl")
    output_dir = Path(output_dir or ADAPTER_DIR / "dpo")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.batch_size,
        beta=cfg.beta,
        max_length=cfg.max_seq_len,
        max_prompt_length=cfg.max_seq_len - 1024,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 # LoRA: reference = base model w/o adapter
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
