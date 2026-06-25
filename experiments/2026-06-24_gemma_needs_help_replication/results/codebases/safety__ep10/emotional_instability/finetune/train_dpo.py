"""LoRA DPO finetuning of Gemma-3-27B-it (App. E, Table 9).

Hyper-parameters (DPO column): 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64,
effective batch size 8, beta 0.1, adapters on all attn+MLP projections.

`layers_to_transform` is exposed for the Appendix I layer-ablation experiments
(e.g. restrict adapters to layers 30-35).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import ARTIFACTS_DIR
from .build_dataset import load_hf_dataset
from .lora import lora_config

# Table 9 (DPO)
DPO_HPARAMS = dict(
    base_model="google/gemma-3-27b-it",
    epochs=1, learning_rate=5e-5, lora_rank=64, lora_alpha=64,
    beta=0.1, effective_batch_size=8,
)


def train_dpo(dataset_path: "str | Path" = ARTIFACTS_DIR / "dpo_dataset.jsonl",
              output_dir: "str | Path" = ARTIFACTS_DIR / "adapters" / "dpo",
              base_model: str = DPO_HPARAMS["base_model"],
              per_device_batch_size: int = 1,
              grad_accum: int = 8,
              load_in_4bit: bool = True,
              layers_to_transform: Optional[list[int]] = None,
              max_length: int = 4096) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant, attn_implementation="eager")

    peft_cfg = lora_config(DPO_HPARAMS["lora_rank"], DPO_HPARAMS["lora_alpha"],
                           layers_to_transform=layers_to_transform)

    dataset = load_hf_dataset(dataset_path)
    # Keep only the columns TRL needs; drop our bookkeeping 'meta' column.
    keep = [c for c in ("prompt", "chosen", "rejected") if c in dataset.column_names]
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in keep])

    cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=DPO_HPARAMS["epochs"],
        learning_rate=DPO_HPARAMS["learning_rate"],
        beta=DPO_HPARAMS["beta"],
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,   # -> effective batch 8
        gradient_checkpointing=True,
        bf16=True, logging_steps=10, save_strategy="epoch",
        max_length=max_length, max_prompt_length=max_length // 2,
    )

    trainer = DPOTrainer(
        model=model, args=cfg, train_dataset=dataset,
        processing_class=tokenizer, peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


if __name__ == "__main__":
    train_dpo()
