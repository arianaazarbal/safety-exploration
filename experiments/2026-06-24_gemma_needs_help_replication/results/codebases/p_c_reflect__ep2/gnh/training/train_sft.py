"""SFT finetuning of Gemma-3-27B-it with LoRA (Table 9, Appendix E/F).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all attention+MLP projections,
effective batch size 8. Trains on chat-formatted conversations, masking loss to
the assistant turns only (TRL's completion-only collator behaviour).
"""

from __future__ import annotations

import json
from pathlib import Path

from gnh.config import ARTIFACT_DIR, GEMMA_27B_IT, LORA_TARGET_MODULES, SFT
from gnh.training.train_dpo import _fold_system


def train_sft(sft_jsonl: Path, output_dir: Path | None = None) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = output_dir or (ARTIFACT_DIR / "sft_diverse_adapter")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_27B_IT.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    with Path(sft_jsonl).open() as f:
        for line in f:
            ex = json.loads(line)
            text = tokenizer.apply_chat_template(
                _fold_system(ex["messages"]), tokenize=False, add_generation_prompt=False
            )
            rows.append({"text": text})
    dataset = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_27B_IT.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    peft_config = LoraConfig(
        r=SFT.lora_rank, lora_alpha=SFT.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
    )
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=SFT.epochs,
        learning_rate=SFT.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=SFT.effective_batch_size,
        logging_steps=10,
        bf16=True,
        save_strategy="epoch",
        dataset_text_field="text",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir
