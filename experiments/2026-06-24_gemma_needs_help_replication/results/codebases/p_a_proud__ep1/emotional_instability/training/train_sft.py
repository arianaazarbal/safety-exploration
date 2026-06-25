"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4.1 / Table 9).

Trains for 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all projection layers,
effective batch size 8. The paper finds SFT ineffective (and the 'teacher' variant
counterproductive); we implement it for the comparison in Figure 5.
"""

from __future__ import annotations

from pathlib import Path

from ..config import CHECKPOINTS_DIR, GENERATION, SFT, ensure_dirs, get_model
from .lora import build_lora_config
from .registry import register_adapter


def train_sft(
    *,
    data_path: Path,
    output_key: str = "gemma-3-27b-it-sft-diverse",
    base_model_key: str = SFT.target_model,
    per_device_batch_size: int = 1,
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = True,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    ensure_dirs()
    base = get_model(base_model_key)
    out_dir = CHECKPOINTS_DIR / output_key

    tokenizer = AutoTokenizer.from_pretrained(base.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(base.model_id, **model_kwargs)

    dataset = load_dataset("json", data_files=str(data_path), split="train")

    grad_accum = max(1, SFT.effective_batch_size // per_device_batch_size)
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=SFT.epochs,
        learning_rate=SFT.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=gradient_checkpointing,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        max_length=4096,
        report_to=[],
        seed=GENERATION.seed,
        # dataset is already conversational ({"messages": [...]}); TRL applies the
        # chat template and masks the prompt tokens.
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(SFT.lora_rank, SFT.lora_alpha),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    register_adapter(output_key, base_model_key, out_dir, display_name=output_key)
    print(f"[sft] saved adapter -> {out_dir}; registered as {output_key!r}")
    return out_dir


def _main() -> None:
    import argparse

    from ..config import TRAINING_DIR

    ap = argparse.ArgumentParser(description="LoRA SFT finetuning of Gemma-3-27B-it")
    ap.add_argument("--calm-source", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()
    data_path = TRAINING_DIR / f"sft_{args.calm_source}.jsonl"
    train_sft(data_path=data_path, output_key=f"gemma-3-27b-it-sft-{args.calm_source}",
              load_in_4bit=args.load_in_4bit)


if __name__ == "__main__":
    _main()
