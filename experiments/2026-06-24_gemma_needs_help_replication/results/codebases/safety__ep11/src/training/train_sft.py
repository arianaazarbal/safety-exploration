"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4, Appendix E Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all proj layers, effective batch
size 8. Trains on calm conversations mixed with Dolci-Instruct-SFT. The paper
finds SFT ineffective (and the 'teacher' variant counterproductive); this script
reproduces the 'diverse' SFT run that shares its calm data with DPO.
"""
from __future__ import annotations

import json
from pathlib import Path

import config


def _load_examples(path: Path) -> "list[dict]":
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def _lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=config.SFT.lora_rank,
        lora_alpha=config.SFT.lora_alpha,
        target_modules=list(config.SFT.target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def train_sft(
    sft_path: Path,
    *,
    base_model: str = config.FINETUNE_BASE_MODEL,
    output_dir: Path | None = None,
) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TrlSFTConfig, SFTTrainer

    output_dir = output_dir or (config.ARTIFACT_DIR / "gemma-sft")
    repo = config.HF_MODELS[base_model]
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map="auto")

    # SFTTrainer applies the chat template to the `messages` field automatically.
    dataset = Dataset.from_list(
        [{"messages": ex["messages"]} for ex in _load_examples(sft_path)])

    args = TrlSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.SFT.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=2048,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"[sft] saved adapter -> {output_dir}")
    return output_dir
