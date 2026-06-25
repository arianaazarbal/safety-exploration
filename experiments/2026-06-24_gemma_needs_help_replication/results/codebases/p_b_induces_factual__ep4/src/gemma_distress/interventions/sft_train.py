"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4.1).

2 epochs, lr 1e-4, LoRA rank-64 on all layers (config.SFT). The paper finds SFT
ineffective at reducing distress (it serves as the negative control against
which DPO is compared in Figure 5).
"""
from __future__ import annotations

from pathlib import Path

from ..config import SFT, LoraConfigSpec


def _peft_config(spec: LoraConfigSpec):
    from peft import LoraConfig

    kwargs = dict(
        r=spec.r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=list(spec.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if spec.layers_to_transform is not None:
        kwargs["layers_to_transform"] = list(spec.layers_to_transform)
    return LoraConfig(**kwargs)


def train_sft(
    base_model_id: str,
    rows: list[dict],
    output_dir: str | Path,
    *,
    cfg=SFT,
    load_in_4bit: bool = False,
):
    """Train an SFT LoRA adapter. ``rows`` are conversational {"messages": [...]}."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model_kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    dataset = Dataset.from_list(rows)

    args = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_peft_config(cfg.lora),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir
