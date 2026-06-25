"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4, Table 9).

Hyperparameters (Appendix E): 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all
attention + MLP projection layers, effective batch size 8, on 1,150 samples
(650 calm + 500 instruct mix). Trains on the multi-turn conversations directly
via TRL's SFTTrainer with the Gemma chat template.

The paper finds SFT ineffective (and the 'teacher' variant counterproductive);
this trainer supports both the 'diverse' and 'teacher' datasets so that
negative result can be reproduced.
"""

from __future__ import annotations

import json
from pathlib import Path

import config
from config import SFTConfig


def _load_messages_dataset(path: Path):
    from datasets import Dataset

    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return Dataset.from_list(rows)


def train(
    sft_path: Path | None = None,
    *,
    cfg: SFTConfig | None = None,
    base_model: str = config.FINETUNE_BASE_MODEL,
    output_dir: Path | None = None,
    tag: str = "diverse",
    load_in_4bit: bool = True,
):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    cfg = cfg or SFTConfig()
    sft_path = sft_path or (config.FINETUNE_DIR / f"sft_{tag}.jsonl")
    output_dir = Path(output_dir or (config.ADAPTER_DIR / f"sft_{tag}"))
    model_id = config.MODELS[base_model].model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    lora = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM", bias="none")

    dataset = _load_messages_dataset(sft_path)
    per_device = 1
    grad_accum = max(1, cfg.effective_batch_size // per_device)

    args = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        # TRL applies the chat template to the "messages" field automatically.
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[sft] saved adapter ({tag}) to {output_dir}")
    return output_dir


if __name__ == "__main__":
    train()
