"""LoRA DPO finetuning of Gemma-3-27B-it (paper Section 4.1, Appendix E).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64
on all attention+MLP projections, effective batch size 8, DPO beta 0.1.

Heavy ML imports are deferred to call-time so the rest of the package imports
without torch/trl installed.
"""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..utils import read_jsonl


def _to_trl_dpo_rows(pairs_path: Path, tokenizer) -> list[dict]:
    """Convert our preference pairs into TRL's {prompt, chosen, rejected} format,
    rendering the chat prompt with the model's template."""
    rows = []
    for p in read_jsonl(pairs_path):
        prompt_text = tokenizer.apply_chat_template(
            p["prompt"], tokenize=False, add_generation_prompt=True)
        rows.append({
            "prompt": prompt_text,
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        })
    return rows


def train_dpo(*, base_model: str = "google/gemma-3-27b-it",
              pairs_path: Path | None = None,
              output_dir: Path | None = None,
              layers_to_transform=None,
              cfg=config.DPO):
    """Run LoRA DPO. Returns the adapter output directory.

    `layers_to_transform` restricts LoRA to a subset of layers (Appendix I layer
    ablations); None == all layers.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    pairs_path = pairs_path or (config.DATA_DIR / "dpo_pairs.jsonl")
    output_dir = output_dir or (config.CHECKPOINT_DIR / "gemma-3-27b-dpo")
    output_dir = Path(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")

    rows = _to_trl_dpo_rows(pairs_path, tokenizer)
    dataset = Dataset.from_list(rows)

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        layers_to_transform=(list(layers_to_transform)
                             if layers_to_transform is not None else None),
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
        bias="none",
    )

    # Effective batch size 8: choose per-device bs * grad-accum == 8.
    per_device_bs = 1
    grad_accum = max(1, cfg.effective_batch_size // per_device_bs)

    dpo_args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=2048,
        max_prompt_length=1024,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
