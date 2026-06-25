"""Section 4 / Appendix E: DPO finetuning of Gemma-3-27B-it with LoRA.

280 preference pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64 on all attention
+ MLP projections, beta 0.1 (Table 9).

Supports the Appendix I layer-subset ablation via ``layers_to_transform`` (e.g.
``[30, 31, 32, 33, 34]`` for "layers 30-35 only").

Gemma-3-27B is large; pass ``--qlora`` to train in 4-bit on a single 48GB+ GPU.
See DESIGN.md for hardware notes.
"""
from __future__ import annotations

from pathlib import Path

from . import config


def train_dpo(dataset_path: Path | None = None,
              output_dir: Path | None = None,
              qlora: bool = False,
              layers_to_transform: list[int] | None = None) -> Path:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    dataset_path = dataset_path or (config.DATA_DIR / "dpo_pairs.jsonl")
    output_dir = output_dir or (config.CHECKPOINT_DIR / "dpo")
    model_id = config.get_model("gemma-3-27b-it").model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "eager"}
    if qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    peft_config = LoraConfig(
        r=config.DPO.lora_rank,
        lora_alpha=config.DPO.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.DPO.target_modules),
        layers_to_transform=layers_to_transform,   # None => all layers
    )

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # Effective batch size 8 via grad accumulation (per-device batch 1 for 27B).
    per_device = 1
    grad_accum = max(1, config.DPO.effective_batch_size // per_device)

    dpo_config = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        beta=config.DPO.beta,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        max_prompt_length=3072,
        seed=config.SEED,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[dpo] saved adapter to {output_dir}")
    return output_dir
