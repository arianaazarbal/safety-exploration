"""DPO finetuning of Gemma-3-27B-it (Section 4 / Appendix E, Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank-64 (alpha 64) on all attention + MLP
projection layers. The Appendix-I layer-subset ablation is supported via
``DPOConfig.layer_range`` -> PEFT ``layers_to_transform``.
"""

from __future__ import annotations

from pathlib import Path

import config
from config import DPOConfig


def _lora_config(cfg: DPOConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if cfg.layer_range is not None:
        lo, hi = cfg.layer_range
        kwargs["layers_to_transform"] = list(range(lo, hi))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _format_pairs(pairs: list[dict], tokenizer) -> "Dataset":
    from datasets import Dataset

    rows = {"prompt": [], "chosen": [], "rejected": []}
    for p in pairs:
        prompt = tokenizer.apply_chat_template(
            p["prompt_messages"], add_generation_prompt=True, tokenize=False
        )
        rows["prompt"].append(prompt)
        rows["chosen"].append(p["chosen"])
        rows["rejected"].append(p["rejected"])
    return Dataset.from_dict(rows)


def train_dpo(
    dpo_pairs: list[dict],
    *,
    base_model_id: str = config.FINETUNE_BASE.model_id,
    cfg: DPOConfig | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    cfg = cfg or DPOConfig()
    output_dir = Path(output_dir or (config.CHECKPOINT_DIR / "gemma-3-27b-dpo"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    dataset = _format_pairs(dpo_pairs, tokenizer)

    training_args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.effective_batch_size,
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
