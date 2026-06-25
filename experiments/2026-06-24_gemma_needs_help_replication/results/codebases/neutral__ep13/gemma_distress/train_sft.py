"""Section 4 / Appendix E: SFT finetuning of Gemma-3-27B-it with LoRA.

650 calm responses mixed with 500 standard instruct samples (Dolci-Instruct-SFT)
to mitigate degeneration; 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 (Table 9).

The paper finds SFT ineffective (the 'teacher' variant even increases distress);
this script reproduces both the 'diverse' and 'teacher' datasets so that failure
mode can be replicated.
"""
from __future__ import annotations

from pathlib import Path

from . import config


def _load_instruct_mix(n: int) -> list[dict]:
    """Sample standard instruct conversations from Dolci-Instruct-SFT."""
    try:
        from datasets import load_dataset
        ds = load_dataset(config.SFT.instruct_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # pragma: no cover - dataset dependent
        print(f"[sft] could not load {config.SFT.instruct_dataset} ({exc}); "
              f"proceeding without instruct mix.")
        return []


def train_sft(variant: str = "diverse", qlora: bool = False,
              output_dir: Path | None = None) -> Path:
    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    output_dir = output_dir or (config.CHECKPOINT_DIR / f"sft_{variant}")
    calm_path = config.DATA_DIR / f"sft_calm_{variant}.jsonl"
    model_id = config.get_model("gemma-3-27b-it").model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "eager"}
    if qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    calm = load_dataset("json", data_files=str(calm_path), split="train")
    mix = _load_instruct_mix(config.scaled(config.SFT.n_instruct_mix))
    records = [{"messages": r["messages"]} for r in calm] + mix
    dataset = Dataset.from_list(records).shuffle(seed=config.SEED)

    peft_config = LoraConfig(
        r=config.SFT.lora_rank,
        lora_alpha=config.SFT.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.SFT.target_modules),
    )

    per_device = 1
    grad_accum = max(1, config.SFT.effective_batch_size // per_device)

    sft_config = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        seed=config.SEED,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[sft:{variant}] saved adapter to {output_dir}")
    return output_dir
