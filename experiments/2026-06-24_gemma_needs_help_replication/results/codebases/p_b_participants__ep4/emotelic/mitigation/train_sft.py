"""SFT of Gemma-3-27B-it on calm data (Section 4.1, Appendix E).

Hyperparameters (Table 9): 1,150 samples, 2 epochs, lr 1e-4, LoRA rank 64 /
alpha 128, effective batch size 8. The paper finds SFT ineffective (and the
'teacher' variant counter-productive); this script reproduces the procedure so
that failure can be replicated.
"""
from __future__ import annotations

from emotelic.mitigation.lora import lora_config
from emotelic.utils.logging import get_logger

log = get_logger("train_sft")


def train_sft(
    sft_jsonl: str,
    *,
    base_model: str = "google/gemma-3-27b-it",
    output_dir: str = "artifacts/sft/gemma-3-27b-sft-diverse",
    epochs: int = 2,
    learning_rate: float = 1e-4,
    rank: int = 64,
    alpha: int = 128,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,           # effective batch size 8
    max_seq_len: int = 4096,
    load_in_4bit: bool = False,
    layers: list[int] | None = None,
) -> str:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(base_model)
    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", **quant,
    )

    ds = load_dataset("json", data_files=sft_jsonl, split="train")

    def to_text(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}

    ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "messages"])

    cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        peft_config=lora_config(rank=rank, alpha=alpha, layers=layers),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(output_dir)
    log.info("Saved SFT adapter -> %s", output_dir)
    return output_dir
