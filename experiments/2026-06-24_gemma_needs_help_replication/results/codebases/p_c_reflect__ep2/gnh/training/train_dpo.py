"""DPO finetuning of Gemma-3-27B-it with LoRA (Table 9, Appendix E).

1 epoch, lr 5e-5, LoRA rank 64 / alpha 64 on all attention+MLP projections,
effective batch size 8, beta 0.1. Supports the Appendix-I layer-subset ablation
via ``lora_layers=(start, end)``.

Uses TRL's DPOTrainer. The prompt is rendered with Gemma's chat template; chosen
and rejected are the two candidate final assistant turns.
"""

from __future__ import annotations

import json
from pathlib import Path

from gnh.config import (
    ARTIFACT_DIR,
    DPO,
    GEMMA_27B_IT,
    LORA_TARGET_MODULES,
)


def _layer_filter_modules(start: int, end: int) -> list[str]:
    """Return module-name patterns restricting LoRA to layers [start, end)."""

    mods = []
    for layer in range(start, end):
        for proj in LORA_TARGET_MODULES:
            mods.append(f"model.layers.{layer}.*{proj}")
    return mods


def train_dpo(
    dpo_jsonl: Path,
    output_dir: Path | None = None,
    *,
    lora_layers: tuple[int, int] | None = None,
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    output_dir = output_dir or (ARTIFACT_DIR / "dpo_adapter")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_27B_IT.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Render the chat-formatted prompt; chosen/rejected are bare assistant text.
    rows = []
    with Path(dpo_jsonl).open() as f:
        for line in f:
            p = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                _fold_system(p["prompt"]), tokenize=False, add_generation_prompt=True
            )
            rows.append({"prompt": prompt, "chosen": p["chosen"], "rejected": p["rejected"]})
    dataset = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_27B_IT.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    target_modules = (
        _layer_filter_modules(*lora_layers) if lora_layers else LORA_TARGET_MODULES
    )
    peft_config = LoraConfig(
        r=DPO.lora_rank, lora_alpha=DPO.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
    )

    args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=DPO.epochs,
        learning_rate=DPO.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=DPO.effective_batch_size,
        beta=DPO.beta,
        logging_steps=10,
        bf16=True,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=dataset,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir


def _fold_system(messages: list[dict]) -> list[dict]:
    """Gemma has no system role; fold any system message into the first user turn."""

    if messages and messages[0]["role"] == "system":
        sys, rest = messages[0]["content"], messages[1:]
        if rest and rest[0]["role"] == "user":
            rest = [{"role": "user", "content": f"{sys}\n\n{rest[0]['content']}"}] + rest[1:]
            return rest
        return [{"role": "user", "content": sys}] + rest
    return messages
