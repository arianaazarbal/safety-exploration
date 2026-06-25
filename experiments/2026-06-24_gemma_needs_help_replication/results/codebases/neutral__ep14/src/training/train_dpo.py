"""DPO finetuning of Gemma-3-27B-it with LoRA (Section 4.1, Appendix E).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64,
effective batch size 8, beta 0.1, adapters on all attention + MLP projections.

The ``target_layers`` argument supports the Appendix I layer-ablation study
(restrict LoRA to a contiguous band of decoder layers).
"""

from __future__ import annotations

import json
from pathlib import Path

from config import CHECKPOINTS_DIR, FINETUNE_BASE, HF_TOKEN_ENV, get_env

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _render_prompt(tokenizer, prompt_messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )


def _layer_filter_modules(model, target_layers: tuple[int, int] | None):
    """Return explicit module names restricting LoRA to decoder layers in
    [lo, hi). Used by the Appendix I ablation; None => all layers."""
    if target_layers is None:
        return LORA_TARGET_MODULES
    lo, hi = target_layers
    names = []
    for name, _ in model.named_modules():
        if any(name.endswith(proj) for proj in LORA_TARGET_MODULES):
            # decoder layer index appears as ".layers.<i>."
            parts = name.split(".")
            if "layers" in parts:
                idx = int(parts[parts.index("layers") + 1])
                if lo <= idx < hi:
                    names.append(name)
    return names


def train_dpo(
    dpo_jsonl: Path,
    *,
    output_dir: Path | None = None,
    base_spec=FINETUNE_BASE,
    epochs: int = 1,
    lr: float = 5e-5,
    beta: float = 0.1,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    effective_batch_size: int = 8,
    per_device_batch_size: int = 1,
    target_layers: tuple[int, int] | None = None,
    load_in_4bit: bool = True,
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    output_dir = output_dir or (CHECKPOINTS_DIR / "dpo_gemma27b")
    token = get_env(HF_TOKEN_ENV)
    tokenizer = AutoTokenizer.from_pretrained(base_spec.model_id, token=token)

    rows = [json.loads(l) for l in open(dpo_jsonl) if l.strip()]
    ds = Dataset.from_list(
        [
            {
                "prompt": _render_prompt(tokenizer, r["prompt_messages"]),
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }
            for r in rows
        ]
    )

    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_spec.model_id, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="eager",
        token=token, **quant,
    )

    target_modules = _layer_filter_modules(model, target_layers)
    peft_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=target_modules, task_type="CAUSAL_LM", bias="none",
    )

    grad_accum = max(1, effective_batch_size // per_device_batch_size)
    cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        max_length=4096,
        max_prompt_length=3072,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir
