"""DPO finetuning of Gemma-3-27B-it with LoRA (Section 4.1, Appendix E).

Hyperparameters (Table 9):
  dataset 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64, beta 0.1,
  effective batch size 8, adapters on all attention + MLP projections.

Uses trl.DPOTrainer + peft.LoraConfig. The 27B model is large; pass
``load_in_4bit=True`` (QLoRA) on a single GPU. The ``target_layers`` argument
supports the Appendix I ablation (restricting adapters to a layer range, e.g.
30--35).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import RESULTS_DIR

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _layer_filtered_targets(model, target_layers: tuple[int, int] | None):
    """Return explicit module names restricted to a decoder layer range.

    ``target_layers=(30, 35)`` -> adapters only on layers [30, 35). Used for the
    Appendix I "which layers matter" ablation. Returns None to adapt all layers.
    """
    if target_layers is None:
        return LORA_TARGET_MODULES
    lo, hi = target_layers
    names = []
    for name, _ in model.named_modules():
        if any(name.endswith(t) for t in LORA_TARGET_MODULES):
            # decoder layers are named '...layers.<i>....'
            parts = name.split(".")
            for p_i, p in enumerate(parts):
                if p == "layers" and p_i + 1 < len(parts):
                    try:
                        layer = int(parts[p_i + 1])
                    except ValueError:
                        break
                    if lo <= layer < hi:
                        names.append(name)
                    break
    return names


def train_dpo(
    base_model: str = "google/gemma-3-27b-it",
    dpo_pairs_path: str | Path = None,
    *,
    output_dir: str | Path = None,
    epochs: int = 1,
    learning_rate: float = 5e-5,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    beta: float = 0.1,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,
    load_in_4bit: bool = True,
    target_layers: tuple[int, int] | None = None,
    hf_token: str | None = None,
    max_length: int = 4096,
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from ..config import DATA_DIR, API

    dpo_pairs_path = Path(dpo_pairs_path or (DATA_DIR / "dpo_pairs.jsonl"))
    output_dir = Path(output_dir or (RESULTS_DIR / "checkpoints" / "dpo"))
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_token = hf_token or API.hf_token

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)

    quant = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant, token=hf_token)

    targets = _layer_filtered_targets(model, target_layers)
    peft_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=targets)

    dataset = _to_hf_dpo_dataset(dpo_pairs_path, tokenizer)

    cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_length // 2,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=cfg, train_dataset=dataset,
        processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def _to_hf_dpo_dataset(path: Path, tokenizer):
    from datasets import Dataset

    prompts, chosen, rejected = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_text = tokenizer.apply_chat_template(
                row["prompt_messages"], tokenize=False,
                add_generation_prompt=True)
            prompts.append(prompt_text)
            chosen.append(row["chosen"])
            rejected.append(row["rejected"])
    return Dataset.from_dict(
        {"prompt": prompts, "chosen": chosen, "rejected": rejected})
