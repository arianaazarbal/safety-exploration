"""LoRA DPO of Gemma-3-27B-it on 280 calm/frustrated preference pairs
(Section 4.1 / Table 9).

Hyperparameters (Table 9): 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64, effective
batch size 8, DPO beta 0.1. The `layers` argument supports the Appendix I
layer-ablation experiment (e.g. layers=[30,31,32,33,34] to adapt layers 30-35).

The dataset's `prompt` field is a list of chat messages; we render it with the
Gemma chat template into the prompt string DPOTrainer expects.
"""

from __future__ import annotations

from pathlib import Path

import config

from .lora import make_lora_config


def _render_prompts(dataset, tokenizer):
    def _map(row):
        row["prompt"] = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True,
        )
        return row
    return dataset.map(_map)


def train_dpo(
    dataset_path: Path,
    output_dir: Path,
    *,
    base_model: str = "google/gemma-3-27b-it",
    epochs: int = 1,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    layers: list[int] | None = None,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,            # effective batch size 8
    max_length: int = 4096,
    max_prompt_length: int = 3072,
    load_in_4bit: bool = True,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", **quant,
    )

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = _render_prompts(dataset, tokenizer)

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        gradient_checkpointing=True,
        report_to=[],
    )
    # With a PEFT model, DPOTrainer builds the reference model implicitly by
    # disabling the adapter, so no separate ref model is needed.
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=make_lora_config(rank=lora_rank, alpha=lora_alpha, layers=layers),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DATASETS_DIR / "dpo_dataset.jsonl"))
    ap.add_argument("--output", default=str(config.CHECKPOINTS_DIR / "dpo"))
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="restrict LoRA to these layer indices (App I ablation)")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()
    train_dpo(Path(args.dataset), Path(args.output),
              layers=args.layers, load_in_4bit=not args.no_4bit)
