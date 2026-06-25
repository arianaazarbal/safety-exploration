"""LoRA SFT finetuning of Gemma-3-27B-it (App. E, Table 9).

Hyper-parameters (SFT column): 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128,
effective batch size 8, adapters on all attn+MLP projections. The paper reports
SFT is ineffective (and the 'teacher' variant slightly worsens distress); we
include it to reproduce that negative result.
"""

from __future__ import annotations

from pathlib import Path

from ..config import ARTIFACTS_DIR
from .build_dataset import load_hf_dataset
from .lora import lora_config

# Table 9 (SFT)
SFT_HPARAMS = dict(
    base_model="google/gemma-3-27b-it",
    epochs=2, learning_rate=1e-4, lora_rank=64, lora_alpha=128,
    effective_batch_size=8,
)


def train_sft(dataset_path: "str | Path" = ARTIFACTS_DIR / "sft_dataset.jsonl",
              output_dir: "str | Path" = ARTIFACTS_DIR / "adapters" / "sft",
              base_model: str = SFT_HPARAMS["base_model"],
              per_device_batch_size: int = 1,
              grad_accum: int = 8,
              load_in_4bit: bool = True,
              max_length: int = 4096) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

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

    peft_cfg = lora_config(SFT_HPARAMS["lora_rank"], SFT_HPARAMS["lora_alpha"])
    dataset = load_hf_dataset(dataset_path)  # conversational {"messages": [...]}

    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=SFT_HPARAMS["epochs"],
        learning_rate=SFT_HPARAMS["learning_rate"],
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,   # -> effective batch 8
        gradient_checkpointing=True,
        bf16=True, logging_steps=10, save_strategy="epoch",
        max_length=max_length, packing=False,
        # NB: training on the full conversation text. Masking to
        # assistant-only loss needs a chat template that emits an assistant
        # token mask (TRL's assistant_only_loss); Gemma 3's template does not
        # reliably, so we leave it off to keep training robust.
    )

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=dataset,
        processing_class=tokenizer, peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


if __name__ == "__main__":
    train_sft()
