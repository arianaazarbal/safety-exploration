"""SFT finetuning of Gemma-3-27B-it (Section 4, Appendix E/F, Table 9).

LoRA rank-64 / alpha-128 on all attention + MLP projections, 2 epochs, lr 1e-4,
effective batch size 8. Trains on 1,150 samples (650 calm + 500 Dolci-Instruct
mix). The paper finds SFT ineffective ("diverse") or even harmful ("teacher");
this trainer supports both variants via the dataset built by ``build_sft_samples``
(diverse) or calm data generated with the teacher system prompt.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.io import read_jsonl


def _lora_config(cfg_block: dict):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg_block["lora_rank"],
        lora_alpha=cfg_block["lora_alpha"],
        target_modules=cfg_block["target_modules"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def train_sft(
    cfg: dict,
    samples_path: str | Path,
    base_model_id: str | None = None,
    output_dir: str | Path | None = None,
):
    import torch
    from datasets import Dataset
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    sft_cfg = cfg["sft"]
    base_model_id = base_model_id or cfg["models"]["gemma"]["gemma-3-27b-it"]["hf_id"]
    output_dir = Path(output_dir or Path(cfg["run"]["output_dir"]) / "adapters" / f"sft_{sft_cfg['variant']}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model = get_peft_model(model, _lora_config(sft_cfg))

    rows = list(read_jsonl(samples_path))
    # TRL SFTTrainer accepts a "messages" column and applies the chat template.
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    bs = sft_cfg["effective_batch_size"]
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=sft_cfg["epochs"],
        learning_rate=sft_cfg["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=bs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
        max_length=4096,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return str(output_dir)
