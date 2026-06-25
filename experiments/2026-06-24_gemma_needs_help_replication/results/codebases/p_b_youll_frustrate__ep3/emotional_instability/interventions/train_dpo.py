"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4.1, Table 9).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64
on all attention + MLP projections, effective batch size 8, DPO beta 0.1.

The ``target_layers`` knob supports the Appendix I layer-ablation study: pass a
range like ``range(30, 36)`` to restrict LoRA adapters to layers 30-35 only.

Run with: ``python -m emotional_instability.cli train-dpo --pairs dpo.jsonl``.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence

from .. import config


def _load_pairs(path: str) -> List[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _layer_filtered_modules(
    base_modules: Sequence[str], target_layers: Optional[Sequence[int]]
) -> Sequence[str]:
    """Translate ``target_modules`` + ``target_layers`` into explicit per-layer
    module names so PEFT only adapts the chosen decoder layers (Appendix I).

    When ``target_layers`` is None, return the bare module names (all layers).
    """
    if not target_layers:
        return list(base_modules)
    names: List[str] = []
    for layer in target_layers:
        for mod in base_modules:
            # Gemma 3 decoder layer path in HF.
            names.append(f"model.layers.{layer}.self_attn.{mod}")
            names.append(f"model.layers.{layer}.mlp.{mod}")
    # Deduplicate while keeping only modules that actually exist for that proj.
    valid_attn = {"q_proj", "k_proj", "v_proj", "o_proj"}
    valid_mlp = {"gate_proj", "up_proj", "down_proj"}
    filtered = [
        n
        for n in names
        if (n.endswith(tuple(valid_attn)) and ".self_attn." in n)
        or (n.endswith(tuple(valid_mlp)) and ".mlp." in n)
    ]
    return filtered


def train_dpo(
    pairs_path: str,
    output_dir: str,
    base_model: str = config.GEMMA_INSTRUCT_27B,
    cfg: Optional[config.DPOConfig] = None,
    settings: Optional[config.Settings] = None,
):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    cfg = cfg or config.DPOConfig()
    settings = settings or config.DEFAULT

    rows = _load_pairs(pairs_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # TRL expects prompt/chosen/rejected as strings; render chat-prompt rows.
    def render(example):
        prompt = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "chosen": example["chosen"], "rejected": example["rejected"]}

    dataset = Dataset.from_list(rows).map(render)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(
            _layer_filtered_modules(cfg.target_modules, cfg.target_layers)
        ),
    )

    # Effective batch size 8: choose per-device bs * grad accumulation = 8.
    per_device_bs = 1
    grad_accum = max(1, cfg.effective_batch_size // per_device_bs)

    training_args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
