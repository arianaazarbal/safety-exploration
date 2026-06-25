"""DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

1 epoch, lr 5e-5, beta 0.1, LoRA rank-64/alpha-64 on all attention + MLP
projections, effective batch size 8. Trains on 280 preference pairs.
"""

from __future__ import annotations

from ..config import DPOConfig
from .lora import build_peft_config


def train_dpo(
    pairs: list[dict],
    cfg: DPOConfig,
    *,
    output_dir: str,
    seed: int = 0,
):
    """Run DPO. ``pairs`` are ``{"prompt": [messages], "chosen", "rejected"}``.

    Returns the path to the saved LoRA adapter (``output_dir``).
    """
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Render the chat ``prompt`` messages into a string the trainer can use.
    def _format(row: dict) -> dict:
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt_text, "chosen": row["chosen"], "rejected": row["rejected"]}

    dataset = Dataset.from_list([_format(p) for p in pairs])

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    grad_accum = max(1, cfg.effective_batch_size // cfg.per_device_batch_size)
    args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        seed=seed,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_peft_config(cfg.lora),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
