"""DPO finetuning of Gemma-3-27B-it with LoRA (Section 4, Appendix E Table 9).

Hyperparameters (Table 9):
    dataset size      280 pairs
    epochs            1
    learning rate     5e-5
    LoRA rank         64
    LoRA alpha        64
    effective batch   8
    DPO beta          0.1
    LoRA targets      q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
                      ("all attention and MLP projection layers")

`target_layers` (Appendix I) optionally restricts LoRA to a contiguous layer
range, e.g. (30, 35) -> adapters only on layers 30-34, used by the layer-ablation
study.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config.settings import SETTINGS

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


@dataclass
class DPOHyperParams:
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8     # effective batch size 8
    beta: float = 0.1
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    max_length: int = 4096
    max_prompt_length: int = 3072
    bf16: bool = True
    target_layers: Optional[tuple[int, int]] = None  # (start, end_exclusive); None = all


def _load_dpo_examples(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _layer_restricted_modules(
    base_model_id: str, layer_range: tuple[int, int]
) -> list[str]:
    """Build explicit module names so LoRA is applied only to a layer subset.

    Gemma-3 decoder layers are named `model.layers.{i}.<proj>`. We enumerate the
    requested layers x target projections.
    """
    start, end = layer_range
    mods = []
    for i in range(start, end):
        for proj in LORA_TARGET_MODULES:
            mods.append(f"model.layers.{i}.self_attn.{proj}"
                        if proj in ("q_proj", "k_proj", "v_proj", "o_proj")
                        else f"model.layers.{i}.mlp.{proj}")
    return mods


def train_dpo(
    base_model_id: str,
    dpo_dataset_path: Path,
    output_dir: Path,
    *,
    hp: Optional[DPOHyperParams] = None,
):
    """Run one epoch of LoRA DPO and save the adapter to `output_dir`."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    hp = hp or DPOHyperParams()
    examples = _load_dpo_examples(dpo_dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    def _format(ex):
        # Render the (conversational) prompt to a string with the chat template;
        # chosen/rejected are the assistant completions.
        prompt_text = tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt_text, "chosen": ex["chosen"], "rejected": ex["rejected"]}

    ds = Dataset.from_list([_format(e) for e in examples])

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if hp.bf16 else torch.float32,
        device_map="auto",
    )

    if hp.target_layers is not None:
        target_modules = _layer_restricted_modules(base_model_id, hp.target_layers)
    else:
        target_modules = LORA_TARGET_MODULES

    peft_config = LoraConfig(
        r=hp.lora_rank,
        lora_alpha=hp.lora_alpha,
        lora_dropout=hp.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    config = DPOConfig(
        output_dir=str(output_dir),
        learning_rate=hp.learning_rate,
        num_train_epochs=hp.num_train_epochs,
        per_device_train_batch_size=hp.per_device_train_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        beta=hp.beta,
        max_length=hp.max_length,
        max_prompt_length=hp.max_prompt_length,
        bf16=hp.bf16,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
