"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4.1 / Table 9).

Trains for 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all projection
layers, effective batch size 8. Saves the adapter and registers it so the eval
can score it by name.
"""

from __future__ import annotations

from pathlib import Path

from ..config import CHECKPOINTS_DIR, DPO, GENERATION, ensure_dirs, get_model
from .build_dpo import DPO_DATA_PATH
from .lora import build_lora_config
from .registry import register_adapter


def train_dpo(
    *,
    data_path: Path = DPO_DATA_PATH,
    output_key: str = "gemma-3-27b-it-dpo",
    base_model_key: str = DPO.target_model,
    layer_range: tuple[int, int] | None = None,
    per_device_batch_size: int = 1,
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = True,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

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

    grad_accum = max(1, DPO.effective_batch_size // per_device_batch_size)
    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=DPO.epochs,
        learning_rate=DPO.learning_rate,
        beta=DPO.beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=gradient_checkpointing,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        max_length=4096,
        max_prompt_length=3072,
        report_to=[],
        seed=GENERATION.seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(DPO.lora_rank, DPO.lora_alpha, layer_range),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    register_adapter(output_key, base_model_key, out_dir,
                     display_name="DPO Gemma (ours)")
    print(f"[dpo] saved adapter -> {out_dir}; registered as {output_key!r}")
    return out_dir


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="LoRA DPO finetuning of Gemma-3-27B-it")
    ap.add_argument("--output-key", default="gemma-3-27b-it-dpo")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    args = ap.parse_args()
    train_dpo(output_key=args.output_key, load_in_4bit=args.load_in_4bit,
              per_device_batch_size=args.per_device_batch_size)


if __name__ == "__main__":
    _main()
