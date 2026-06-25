"""DPO LoRA finetuning of Gemma-3-27B-it (Section 4.1, Appendix E Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank/alpha 64, effective batch size 8, on the
280 preference pairs. The ``layers_to_transform`` field on the LoRA config
supports the Appendix-I layer ablations (e.g. layers 30–35 only).
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from .build_dpo_dataset import DPO_PATH


def _format_prompt(tokenizer, messages: list[dict]) -> str:
    """Render the chat context into the model's prompt string for DPO.

    TRL's DPOTrainer expects ``prompt``/``chosen``/``rejected`` text fields; we
    pre-render the prompt with the chat template (add_generation_prompt=True) so
    chosen/rejected are plain assistant completions.
    """
    folded = _fold_system(messages)
    return tokenizer.apply_chat_template(
        folded, tokenize=False, add_generation_prompt=True)


def _fold_system(messages):
    # Gemma has no system role; this mirrors hf_backend folding.
    if messages and messages[0]["role"] == "system":
        sys = messages[0]["content"]
        rest = messages[1:]
        if rest and rest[0]["role"] == "user":
            rest = [{"role": "user",
                     "content": f"{sys}\n\n{rest[0]['content']}"}] + rest[1:]
        return rest
    return messages


def train_dpo(output_dir: str | None = None,
              cfg: config.DPOTrainConfig | None = None,
              base_model: str = "gemma-3-27b-it") -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    cfg = cfg or config.DPOTrainConfig()
    output_dir = output_dir or str(config.CHECKPOINT_DIR / "dpo-gemma-27b")
    model_id = config.MODEL_REGISTRY[base_model].model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=config.HF_TOKEN or None)
    rows = [json.loads(l) for l in DPO_PATH.read_text().splitlines() if l]
    ds = Dataset.from_list([{
        "prompt": _format_prompt(tokenizer, r["prompt"]),
        "chosen": r["chosen"],
        "rejected": r["rejected"],
    } for r in rows])

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        token=config.HF_TOKEN or None)

    peft_config = LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        layers_to_transform=(list(cfg.lora.layers_to_transform)
                             if cfg.lora.layers_to_transform else None),
        task_type="CAUSAL_LM",
    )

    grad_accum = max(1, cfg.effective_batch_size // 1)
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model, args=dpo_config, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[dpo] adapter saved to {output_dir}")
    return Path(output_dir)
