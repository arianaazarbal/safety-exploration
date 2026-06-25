"""SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E / Table 9).

650 calm responses mixed with 500 Dolci-Instruct-SFT samples (to mitigate
degeneration), 2 epochs, lr 1e-4, LoRA rank-64 / alpha-128 on all projections.
The paper reports SFT is ineffective (and the 'teacher' variant worsens
distress); we replicate the *setup* so that result can be reproduced.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import (
    DATA_DIR,
    INSTRUCT_MIX_DATASET,
    LORA_TARGET_MODULES,
    RESULTS_DIR,
    SFT as SFT_CFG,
    SFTConfig,
    TRAINABLE_MODEL,
    scaled_n,
)
from ..models import MODELS


def _build_mixed_dataset(calm_path: Path, n_instruct: int):
    """Concatenate calm conversations with Dolci-Instruct-SFT samples."""
    from datasets import Dataset, load_dataset

    calm_rows = [json.loads(l) for l in calm_path.open()] if calm_path.exists() else []

    instruct_rows = []
    try:
        ds = load_dataset(INSTRUCT_MIX_DATASET, split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= n_instruct:
                break
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                instruct_rows.append({"messages": msgs})
    except Exception as e:  # noqa: BLE001
        print(f"[sft] WARNING: could not load {INSTRUCT_MIX_DATASET}: {e}. "
              "Proceeding with calm data only (degeneration mitigation reduced).")

    rows = calm_rows + instruct_rows
    return Dataset.from_list(rows)


def train_sft(
    cfg: SFTConfig = SFT_CFG,
    calm_dataset_path: Path | None = None,
    base_model_key: str = TRAINABLE_MODEL,
    output_dir: Path | None = None,
) -> Path:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    spec = MODELS[base_model_key]
    calm_dataset_path = calm_dataset_path or (DATA_DIR / f"sft_{cfg.dataset}.jsonl")
    output_dir = output_dir or (RESULTS_DIR / "adapters" / f"sft_{cfg.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    peft_config = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
    )

    dataset = _build_mixed_dataset(calm_dataset_path, scaled_n(cfg.n_instruct_mix))

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
        gradient_checkpointing=True,
        report_to=[],
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[sft] saved adapter -> {output_dir}")
    return output_dir
