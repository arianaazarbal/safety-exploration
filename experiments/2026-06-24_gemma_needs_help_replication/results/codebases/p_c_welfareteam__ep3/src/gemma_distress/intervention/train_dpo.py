"""DPO finetuning of Gemma-3-27B-it (paper Section 4.1, Appendix E).

Specified by the paper: 280 preference pairs, 1 epoch, learning rate 5e-5,
LoRA rank-64 adapters on all layers. Appendix E (full training details) and the
DPO beta were not in the provided extraction; we default beta=0.1 (the common
DPO default) and expose it in config -- see DESIGN.md "Filled-in training
hyperparameters".

Supports the internal-vs-external emotion ablation from Section 4.2: pass
``layer_ablation`` as a list of decoder-layer indices to restrict LoRA to a
layer window (e.g. layers 30-35, which the paper finds nearly as effective as
all layers, vs layer 40+ which is not).
"""
from __future__ import annotations

from pathlib import Path


def _lora_config(rank: int, layer_ablation: list[int] | None):
    from peft import LoraConfig

    # Standard Gemma attention + MLP projection modules.
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    kwargs = dict(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    if layer_ablation is not None:
        # restrict adapters to a window of decoder layers (Section 4.2 ablation)
        kwargs["layers_to_transform"] = layer_ablation
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _render_dataset(pairs: list[dict], tokenizer):
    """Render prompts with the chat template; return an HF Dataset for DPOTrainer."""
    from datasets import Dataset

    rows = {"prompt": [], "chosen": [], "rejected": []}
    for p in pairs:
        prompt = tokenizer.apply_chat_template(
            p["prompt"], tokenize=False, add_generation_prompt=True
        )
        rows["prompt"].append(prompt)
        rows["chosen"].append(p["chosen"])
        rows["rejected"].append(p["rejected"])
    return Dataset.from_dict(rows)


def run(
    pairs: list[dict],
    *,
    base_model: str = "google/gemma-3-27b-it",
    output_dir: str | Path = "runs/section4/dpo_model",
    epochs: int = 1,
    learning_rate: float = 5e-5,
    lora_rank: int = 64,
    beta: float = 0.1,
    layer_ablation: list[int] | None = None,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,
    max_length: int = 2048,
    hf_token: str | None = None,
) -> str:
    """Train a DPO LoRA adapter and save it to ``output_dir``. Returns the path."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, token=hf_token, torch_dtype=torch.bfloat16, device_map="auto"
    )

    dataset = _render_dataset(pairs, tokenizer)
    peft_config = _lora_config(lora_rank, layer_ablation)

    cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_length // 2,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return str(output_dir)
