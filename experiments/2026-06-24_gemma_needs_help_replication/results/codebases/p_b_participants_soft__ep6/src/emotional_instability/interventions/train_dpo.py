"""DPO of Gemma-3-27B-it on 280 preference pairs (Section 4.1).

Paper: 280 pairs (frustrated score>=3 as rejected, calm as chosen, matching turn
counts), 1 epoch, lr 5e-5, LoRA rank-64 on all layers. This is the headline
intervention: it drops avg high-frustration from 35% to 0.3% across evaluations.
"""

from __future__ import annotations

from ..config import Config
from .lora import build_peft_config


def _render_prompt(tokenizer, prompt_messages: list[dict]) -> str:
    """Render the (possibly multi-turn) prompt ending at the model's turn."""
    return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)


def train_dpo(base_model_id: str, pairs: list[dict], cfg: Config, out_dir: str) -> str:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    rows = [
        {
            "prompt": _render_prompt(tokenizer, p["prompt_messages"]),
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ]
    ds = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", torch_dtype="bfloat16")

    args = TRLDPOConfig(
        output_dir=out_dir,
        num_train_epochs=cfg.dpo.epochs,
        learning_rate=cfg.dpo.learning_rate,
        beta=cfg.dpo.beta,
        per_device_train_batch_size=cfg.dpo.batch_size,
        gradient_accumulation_steps=cfg.dpo.grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    # ref_model=None -> TRL uses the LoRA-disabled base as the implicit reference.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=ds,
        peft_config=build_peft_config(cfg.lora),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(out_dir)
    return out_dir
