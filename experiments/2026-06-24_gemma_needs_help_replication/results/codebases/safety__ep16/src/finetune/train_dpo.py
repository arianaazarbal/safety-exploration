"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4.1, App. E Table 9).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64, alpha 64,
effective batch size 8, DPO beta 0.1, LoRA on all attention + MLP projections
(q,k,v,o,gate,up,down). Trained adapter saved to checkpoints/gemma-3-27b-dpo.
"""

from __future__ import annotations

from pathlib import Path

from config import API, CHECKPOINTS_DIR, DATASETS_DIR

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _to_dpo_record(tokenizer, row: dict) -> dict:
    """Render a stored pair into TRL's {prompt, chosen, rejected} text format
    using the chat template. The prompt is the shared conversation context."""
    from src.models.hf_model import _fold_system

    msgs = _fold_system(row["prompt_messages"])
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}


def train_dpo(
    *,
    base_model: str = "google/gemma-3-27b-it",
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    epochs: int = 1,
    learning_rate: float = 5e-5,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    effective_batch_size: int = 8,
    per_device_batch_size: int = 1,
    beta: float = 0.1,
    load_in_4bit: bool = True,
):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dataset_path = dataset_path or (DATASETS_DIR / "dpo_dataset.jsonl")
    output_dir = output_dir or (CHECKPOINTS_DIR / "gemma-3-27b-dpo")

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=API.hf_token)

    model_kwargs: dict = {"torch_dtype": torch.bfloat16, "token": API.hf_token, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    peft_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
    )

    raw = load_dataset("json", data_files=str(dataset_path), split="train")
    ds = raw.map(lambda row: _to_dpo_record(tokenizer, row), remove_columns=raw.column_names)

    grad_accum = max(1, effective_batch_size // per_device_batch_size)
    cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        beta=beta,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        max_length=2048,
        max_prompt_length=1536,
    )

    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train_dpo] saved adapter -> {output_dir}")
    return output_dir
