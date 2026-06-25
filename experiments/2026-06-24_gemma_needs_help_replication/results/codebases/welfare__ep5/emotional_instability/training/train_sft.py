"""LoRA SFT fine-tuning of Gemma-3-27B-it (Section 4.1, Appendix E/Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all attention+MLP projection
layers, effective batch size 8. Trains on calm conversations (+instruct mix).
The paper finds SFT ineffective (and the 'teacher' variant counterproductive);
this trainer reproduces the setup so that failure mode can be measured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .. import config


def _load_rows(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(l) for l in f if l.strip()]


def train_sft(
    sft_path: Path,
    *,
    base_model: str = config.DPO_BASE_MODEL.model_id,
    output_dir: Optional[Path] = None,
    cfg: config.SFTConfig = config.SFT,
    load_in_4bit: bool = True,
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    output_dir = Path(output_dir or (config.CHECKPOINT_DIR / "gemma27b-sft"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    quant = {}
    if load_in_4bit:
        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", **quant
    )

    rows = _load_rows(sft_path)
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    lora = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        task_type="CAUSAL_LM",
    )

    args = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.effective_batch_size,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        # SFTTrainer applies the chat template to the "messages" column and
        # masks the loss to assistant tokens by default for conversational data.
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[sft] saved adapter -> {output_dir}")
    return output_dir
