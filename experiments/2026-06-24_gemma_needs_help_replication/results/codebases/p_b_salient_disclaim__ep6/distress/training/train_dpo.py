"""DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E Table 9).

1 epoch, lr 5e-5, LoRA rank-64 alpha-64 on all attention+MLP projections, DPO
beta 0.1, effective batch size 8, on the 280 preference pairs. Implemented with
TRL's ``DPOTrainer`` + PEFT.

The ``layer_ablation`` argument reproduces Appendix I by restricting LoRA to a
named layer subset from ``config.LAYER_ABLATION_RANGES``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .. import config
from ..config import DPO_CONFIG
from .build_datasets import DPO_PATH
from .lora_utils import adapter_dir, build_lora_config
from ..utils.io import read_jsonl


def _format_pairs_for_trl(tokenizer):
    """Yield TRL-style {prompt, chosen, rejected} rows. The prompt is rendered
    with the chat template (the history up to the final assistant turn)."""
    rows = []
    for p in read_jsonl(DPO_PATH):
        prompt = tokenizer.apply_chat_template(
            p["prompt_messages"], tokenize=False, add_generation_prompt=True
        )
        rows.append({
            "prompt": prompt,
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        })
    return rows


def train(layer_ablation: str = "all", run_name: str | None = None) -> Path:
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from peft import get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from trl import DPOConfig, DPOTrainer  # type: ignore

    cfg = DPO_CONFIG
    if layer_ablation != "all":
        cfg = replace(cfg, layer_range=config.LAYER_ABLATION_RANGES[layer_ablation])

    run_name = run_name or f"dpo_{layer_ablation}"
    out_dir = adapter_dir(run_name)

    model_id = config.DPO_TARGET.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = get_peft_model(model, build_lora_config(cfg))

    dataset = Dataset.from_list(_format_pairs_for_trl(tokenizer))

    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.effective_batch_size,
        beta=cfg.dpo_beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


def train_all_layer_ablations() -> dict[str, Path]:
    """Appendix I: DPO with LoRA on each layer subset."""
    return {name: train(layer_ablation=name, run_name=f"dpo_{name}")
            for name in config.LAYER_ABLATION_RANGES}
