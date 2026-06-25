"""DPO finetuning of Gemma-3-27B-it (Section 4, Appendix E, Table 9).

LoRA rank-64 / alpha-64 adapters on all attention + MLP projections, 1 epoch,
lr 5e-5, beta 0.1, effective batch size 8. Trains on 280 (prompt, chosen,
rejected) pairs where chosen = calm response and rejected = frustrated response.

The ``layers`` config supports the Appendix I ablation: pass an explicit list of
layer indices (e.g. [30,31,32,33,34]) to restrict adapters to a central band and
test whether a shallow final-layer intervention suffices (it doesn't).
"""
from __future__ import annotations

from pathlib import Path

from ..utils.io import read_jsonl


def _lora_config(cfg_block: dict):
    from peft import LoraConfig

    layers = cfg_block.get("layers", "all")
    kwargs = dict(
        r=cfg_block["lora_rank"],
        lora_alpha=cfg_block["lora_alpha"],
        target_modules=cfg_block["target_modules"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if isinstance(layers, list):  # Appendix I: restrict to a layer subset
        kwargs["layers_to_transform"] = layers
    return LoraConfig(**kwargs)


def _format_prompt(tokenizer, prompt_messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)


def train_dpo(
    cfg: dict,
    pairs_path: str | Path,
    base_model_id: str | None = None,
    output_dir: str | Path | None = None,
):
    import torch
    from datasets import Dataset
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dpo_cfg = cfg["dpo"]
    base_model_id = base_model_id or cfg["models"]["gemma"]["gemma-3-27b-it"]["hf_id"]
    output_dir = Path(output_dir or Path(cfg["run"]["output_dir"]) / "adapters" / "dpo")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model = get_peft_model(model, _lora_config(dpo_cfg))

    rows = list(read_jsonl(pairs_path))
    ds = Dataset.from_list([{
        "prompt": _format_prompt(tokenizer, r["prompt"]),
        "chosen": r["chosen"],
        "rejected": r["rejected"],
    } for r in rows])

    bs = dpo_cfg["effective_batch_size"]
    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=dpo_cfg["epochs"],
        learning_rate=dpo_cfg["learning_rate"],
        beta=dpo_cfg["beta"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=bs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return str(output_dir)
