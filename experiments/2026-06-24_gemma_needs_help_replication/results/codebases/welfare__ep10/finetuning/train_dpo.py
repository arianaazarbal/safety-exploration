"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4, Table 9).

Hyperparameters (Appendix E): 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64 on all
attention + MLP projection layers, effective batch size 8, DPO beta 0.1, on 280
preference pairs.

Each pair shares a conversation history (`prompt`) and differs only in the final
assistant turn (`chosen` calm vs `rejected` frustrated). We render the shared
history with Gemma's chat template so the model trains on the same format it sees
at eval time.

The optional `layer_range` (Appendix I) restricts LoRA to a contiguous band of
decoder layers — used to show that intervening only on central layers (30-35) is
nearly as effective as all layers, evidence the intervention touches internal
(not just final-layer) emotion representations.
"""

from __future__ import annotations

import json
from pathlib import Path

import config
from config import DPOConfig


def _load_pairs(path: Path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _build_lora_config(cfg: DPOConfig, model):
    """Build a LoRA config, optionally restricted to a decoder-layer band."""
    from peft import LoraConfig

    target_modules = list(cfg.target_modules)
    layers_to_transform = None
    if cfg.layer_range is not None:
        start, end = cfg.layer_range
        layers_to_transform = list(range(start, end))
    return LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        task_type="CAUSAL_LM",
        bias="none",
    )


def _render_prompt(tokenizer, history: list[dict]) -> str:
    """Render the shared history with a generation prompt so chosen/rejected
    continue from the same point."""
    return tokenizer.apply_chat_template(
        history, add_generation_prompt=True, tokenize=False)


def train(
    pairs_path: Path | None = None,
    *,
    cfg: DPOConfig | None = None,
    base_model: str = config.FINETUNE_BASE_MODEL,
    output_dir: Path | None = None,
    load_in_4bit: bool = True,
):
    """Run LoRA DPO and save the adapter to `output_dir`."""
    import torch
    from datasets import Dataset
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    cfg = cfg or DPOConfig()
    pairs_path = pairs_path or (config.FINETUNE_DIR / "dpo_pairs.jsonl")
    output_dir = Path(output_dir or (config.ADAPTER_DIR / "dpo"))
    model_id = config.MODELS[base_model].model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model = get_peft_model(model, _build_lora_config(cfg, model))

    pairs = _load_pairs(pairs_path)
    rows = []
    for p in pairs:
        prompt_text = _render_prompt(tokenizer, p["prompt"])
        rows.append({
            "prompt": prompt_text,
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        })
    dataset = Dataset.from_list(rows)

    # effective_batch_size = per_device_batch * grad_accum (assume 1 device here)
    per_device = 1
    grad_accum = max(1, cfg.effective_batch_size // per_device)

    training_args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[dpo] saved adapter to {output_dir}")
    return output_dir


if __name__ == "__main__":
    train()
