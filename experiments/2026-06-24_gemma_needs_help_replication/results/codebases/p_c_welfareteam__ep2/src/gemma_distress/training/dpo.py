"""LoRA DPO of Gemma-3-27B-it on 280 preference pairs (Section 4, Table 9).

Hyperparameters (Table 9): 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64
on all attention + MLP projections, effective batch size 8. This is the
paper's headline mitigation: it drops the average high-frustration rate from
35% to 0.3%.

``cfg.lora.layers_to_transform`` enables the Appendix I layer ablations (e.g.
restricting adapters to layers 30-35).
"""

from __future__ import annotations

from pathlib import Path

from gemma_distress.config import DPOConfig
from gemma_distress.training.sft import _lora_config


def _format_prompt(tokenizer, prompt_messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )


def train_dpo(
    model_id: str,
    pairs: list[dict],
    cfg: DPOConfig,
    per_device_batch_size: int = 1,
) -> Path:
    """Run LoRA DPO and return the adapter output directory.

    ``pairs`` items have keys ``prompt`` (chat message list), ``chosen`` and
    ``rejected`` (assistant completion strings); we render the prompt with the
    Gemma chat template so the preference is over the assistant turn only.
    """
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    rows = [
        {
            "prompt": _format_prompt(tokenizer, p["prompt"]),
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ]
    dataset = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16")

    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)
    args = TRLDPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=cfg.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=cfg.seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg.lora),
    )
    trainer.train()
    out = Path(cfg.output_dir) / "final"
    trainer.save_model(str(out))
    return out
