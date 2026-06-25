"""LoRA DPO fine-tuning of Gemma-3-27B-it (Section 4.1, Appendix E/Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all attention+MLP
projection layers, effective batch size 8. Optionally restrict adapters to a
subset of layers (Appendix I layer-ablation: e.g. layers 30-35 only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from .. import config


def _load_pairs(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(l) for l in f if l.strip()]


def _render_prompt(tokenizer, prompt_messages) -> str:
    # Fold system into first user turn (Gemma idiom), then apply chat template
    # with a trailing generation prompt so chosen/rejected are pure completions.
    msgs = list(prompt_messages)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _layer_filter(target_layers: Optional[Sequence[int]]):
    """Return a predicate selecting LoRA target module names by layer index.

    PEFT's ``layers_to_transform`` handles this directly; we expose it via the
    LoraConfig below. This helper is kept for documentation/validation.
    """
    return None if target_layers is None else list(target_layers)


def train_dpo(
    pairs_path: Path,
    *,
    base_model: str = config.DPO_BASE_MODEL.model_id,
    output_dir: Optional[Path] = None,
    target_layers: Optional[Sequence[int]] = None,
    cfg: config.DPOConfig = config.DPO,
    load_in_4bit: bool = True,
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    output_dir = Path(output_dir or (config.CHECKPOINT_DIR / "gemma27b-dpo"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", **quant
    )

    pairs = _load_pairs(pairs_path)
    ds = Dataset.from_list([
        {
            "prompt": _render_prompt(tokenizer, p["prompt"]),
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ])

    lora = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        layers_to_transform=_layer_filter(target_layers),
        task_type="CAUSAL_LM",
    )

    # Effective batch size 8 via per-device batch x grad accumulation.
    args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.effective_batch_size,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[dpo] saved adapter -> {output_dir}")
    return output_dir
