"""DPO finetuning of Gemma-3-27B-it with LoRA (Section 4 / Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank-64 alpha-64 on all attn+MLP projections,
effective batch size 8. Trains on the 280 preference pairs from datasets.py.
"""

from __future__ import annotations

from pathlib import Path

from ..config import FinetuneConfig
from ..clients.factory import model_by_name
from .datasets import DPO_PATH


def _lora_config(cfg: FinetuneConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if cfg.lora.layers_to_transform is not None:
        kwargs["layers_to_transform"] = cfg.lora.layers_to_transform
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _format_pairs(dataset, tokenizer):
    """Convert {prompt:[msgs], chosen:str, rejected:str} -> TRL 'standard' format
    {prompt:str, chosen:str, rejected:str} using the chat template."""

    def _map(ex):
        prompt = tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "chosen": ex["chosen"], "rejected": ex["rejected"]}

    return dataset.map(_map, remove_columns=dataset.column_names)


def train(cfg: FinetuneConfig, data_path: Path = DPO_PATH) -> str:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    base_id = model_by_name(cfg.base_model).model_id
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=str(data_path), split="train")
    ds = _format_pairs(ds, tokenizer)

    grad_accum = max(1, cfg.effective_batch_size // cfg.per_device_batch_size)
    args = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        beta=cfg.dpo_beta,
        max_length=cfg.max_seq_len,
        max_prompt_length=cfg.max_seq_len // 2,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=cfg.seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[done] DPO adapter saved -> {cfg.output_dir}")
    return cfg.output_dir
