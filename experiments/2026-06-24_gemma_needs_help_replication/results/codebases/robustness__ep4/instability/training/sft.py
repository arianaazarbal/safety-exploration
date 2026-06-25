"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4.1 / Appendix E).

Hyperparameters from the paper: 2 epochs, learning rate 1e-4, LoRA rank-64
adapters on all layers. SFT is included as the *ineffective* baseline the paper
reports (it fails to reduce — and in one variant slightly increases — distress).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SFTConfig:
    base_model: str = "google/gemma-3-27b-it"
    dataset_path: str = "outputs/data/sft.jsonl"
    output_dir: str = "outputs/models/gemma-sft"
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 4096
    bf16: bool = True
    load_in_4bit: bool = False
    seed: int = 0


# LoRA on "all layers": every linear projection in attention + MLP.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def train_sft(cfg: SFTConfig):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
                        device_map="auto")
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)

    peft_config = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM", bias="none",
    )

    dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")

    args = TRLSFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_length=cfg.max_seq_length,
        bf16=cfg.bf16,
        logging_steps=10,
        save_strategy="epoch",
        seed=cfg.seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset,
        peft_config=peft_config, processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[train_sft] adapter saved -> {cfg.output_dir}")
    return cfg.output_dir
