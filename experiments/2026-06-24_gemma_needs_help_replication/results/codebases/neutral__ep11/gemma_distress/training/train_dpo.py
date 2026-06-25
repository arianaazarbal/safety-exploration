"""LoRA DPO fine-tuning of Gemma-3-27B-it (Section 4.1, Table 9).

Hyper-parameters: 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64 on all
attention + MLP projections, DPO beta 0.1, effective batch size 8.

Optional ``layer_subset`` restricts the LoRA adapters to a contiguous range of
decoder layers, used for the internal-vs-expressed-emotion ablation
(Appendix I): adapters on layers 30-35 are nearly as effective as all layers,
while adapters from layer 40 on are not.
"""

from __future__ import annotations

from pathlib import Path

from ..config import (CHECKPOINTS_DIR, DPO_CFG, LORA_TARGET_MODULES,
                      GEMMA_27B_IT, ModelSpec)


def _layer_filter_modules(model, target_modules, layer_subset):
    """Return explicit module names restricted to `layer_subset` decoder layers."""
    import re
    lo, hi = layer_subset
    names = []
    pat = re.compile(r"\.layers\.(\d+)\.")
    for name, module in model.named_modules():
        if not any(name.endswith(t) for t in target_modules):
            continue
        m = pat.search(name)
        if m and lo <= int(m.group(1)) < hi:
            names.append(name)
    return names


def train_dpo(
    dataset_path: Path,
    base_spec: ModelSpec = GEMMA_27B_IT,
    output_name: str = "dpo",
    layer_subset: tuple[int, int] | None = None,
    cfg=DPO_CFG,
) -> Path:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    out_dir = CHECKPOINTS_DIR / f"{output_name}_{base_spec.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_spec.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_spec.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    if layer_subset is None:
        target_modules = LORA_TARGET_MODULES
    else:
        target_modules = _layer_filter_modules(model, LORA_TARGET_MODULES, layer_subset)

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    # effective batch size 8 via grad accumulation
    per_device_bs = 1
    grad_accum = max(1, cfg.effective_batch_size // per_device_bs)

    args = TRLDPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        max_prompt_length=3072,
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
