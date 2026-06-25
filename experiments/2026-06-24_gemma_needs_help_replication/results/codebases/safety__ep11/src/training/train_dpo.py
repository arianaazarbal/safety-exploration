"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4, Appendix E Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all attention+MLP proj
layers, effective batch size 8. Supports the Appendix I layer-ablation study via
``layers_to_tune`` (e.g. range(30, 36) for the "layers 30-35 only" condition).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import config


def _load_pairs(path: Path, tokenizer) -> "list[dict]":
    """Load DPO pairs and render the prompt with the chat template."""
    rows = []
    for line in Path(path).open():
        if not line.strip():
            continue
        ex = json.loads(line)
        prompt = tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": prompt, "chosen": ex["chosen"],
                     "rejected": ex["rejected"]})
    return rows


def _lora_config(layers_to_tune: Optional[Iterable[int]]):
    from peft import LoraConfig

    kwargs = dict(
        r=config.DPO.lora_rank,
        lora_alpha=config.DPO.lora_alpha,
        target_modules=list(config.DPO.target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers_to_tune is not None:
        # Restrict adapters to a subset of decoder layers (Appendix I ablations).
        kwargs["layers_to_transform"] = list(layers_to_tune)
    return LoraConfig(**kwargs)


def train_dpo(
    pairs_path: Path,
    *,
    base_model: str = config.FINETUNE_BASE_MODEL,
    output_dir: Path | None = None,
    layers_to_tune: Optional[Iterable[int]] = None,
) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TrlDPOConfig, DPOTrainer

    output_dir = output_dir or (config.ARTIFACT_DIR / "gemma-dpo")
    repo = config.HF_MODELS[base_model]
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map="auto")

    dataset = Dataset.from_list(_load_pairs(pairs_path, tokenizer))

    args = TrlDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        beta=config.DPO.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.DPO.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(layers_to_tune),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"[dpo] saved adapter -> {output_dir}")
    return output_dir
