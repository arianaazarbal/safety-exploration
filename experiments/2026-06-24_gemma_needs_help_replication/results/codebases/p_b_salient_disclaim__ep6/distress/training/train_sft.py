"""SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E Table 9).

2 epochs, lr 1e-4, LoRA rank-64 alpha-128 on all attention+MLP projections,
effective batch size 8, on 650 calm + 500 Dolci-Instruct samples. Implemented
with TRL's ``SFTTrainer`` + PEFT.

The paper trains two variants: 'diverse' (the main calm dataset) and 'teacher'
(generated with the teacher system prompt, Appendix F). ``teacher=True`` selects
the teacher dataset.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from ..config import SFT_CONFIG
from .build_datasets import SFT_PATH
from .lora_utils import adapter_dir, build_lora_config
from ..utils.io import read_jsonl


def train(teacher: bool = False, run_name: str | None = None) -> Path:
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from peft import get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore

    cfg = SFT_CONFIG
    run_name = run_name or ("sft_teacher" if teacher else "sft_diverse")
    out_dir = adapter_dir(run_name)
    data_path = config.DATA_DIR / "sft_teacher.jsonl" if teacher else SFT_PATH

    model_id = config.DPO_TARGET.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = get_peft_model(model, build_lora_config(cfg))

    # TRL's SFTTrainer accepts a "messages" column and applies the chat template.
    dataset = Dataset.from_list([
        {"messages": row["messages"]} for row in read_jsonl(data_path)
    ])

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
