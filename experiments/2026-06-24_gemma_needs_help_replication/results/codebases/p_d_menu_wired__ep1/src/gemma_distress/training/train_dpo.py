"""DPO finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64
on all attention + MLP projections, effective batch size 8, DPO beta 0.1.

Uses TRL's DPOTrainer + PEFT LoRA. The function is import-safe: heavy deps
(torch/trl/peft) are imported inside the body so the module can be loaded
without a GPU environment.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Config


def _load_pairs_as_dataset(pairs_path: str | Path):
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    with open(pairs_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    # TRL DPO expects columns: prompt, chosen, rejected. We render the shared
    # context with the tokenizer chat template at train time (below), so here
    # we just carry the raw fields.
    return Dataset.from_list(rows)


def train_dpo(cfg: Config, pairs_path: str | Path, output_dir: str | Path | None = None) -> str:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tcfg = cfg.training
    dcfg = tcfg["dpo"]
    base_model = tcfg["base_model"]
    out_dir = str(output_dir or (cfg.output_dir / "adapters" / "dpo"))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    ds = _load_pairs_as_dataset(pairs_path)

    def _format(row: dict[str, Any]) -> dict[str, str]:
        # Render the shared multi-turn context into a prompt string; chosen and
        # rejected are the final assistant completions.
        prompt = tokenizer.apply_chat_template(
            row["prompt_messages"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}

    ds = ds.map(_format, remove_columns=ds.column_names)

    peft_config = LoraConfig(
        r=tcfg["lora_rank"],
        lora_alpha=dcfg["lora_alpha"],
        target_modules=tcfg["lora_target_modules"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    bs = dcfg["effective_batch_size"]
    args = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=dcfg["epochs"],
        learning_rate=dcfg["learning_rate"],
        beta=dcfg["beta"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=bs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
