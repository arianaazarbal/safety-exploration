"""DPO / SFT LoRA training of Gemma-3-27B-it (Section 4.1, Appendix E / Table 9).

Thin wrappers around TRL's ``DPOTrainer`` and ``SFTTrainer`` with PEFT LoRA. The
effective batch size of 8 is realised as ``per_device_batch_size * grad_accum``;
defaults assume a single device with grad accumulation, overridable for multi-GPU.

Only the LoRA adapter is saved. Nothing here has been executed; it is provided for
review and for running on the lab's training hardware.
"""

from __future__ import annotations

from pathlib import Path

from ..models.registry import model_info
from .lora import build_lora_config


def _batch_split(effective_batch_size: int, per_device: int) -> int:
    grad_accum = max(1, effective_batch_size // per_device)
    return grad_accum


def train_dpo(training_cfg: dict, dpo_pairs: list[dict], out_dir: str | Path,
              per_device_batch_size: int = 1) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    base = model_info(training_cfg["base_model"])["hf_id"]
    dcfg = training_cfg["dpo"]
    out_dir = Path(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)

    # TRL expects prompt as a string; render the shared neutral context with the
    # chat template and a generation prompt so chosen/rejected are completions.
    def _render(example: dict) -> dict:
        prompt = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "chosen": example["chosen"],
                "rejected": example["rejected"]}

    dataset = Dataset.from_list([_render(p) for p in dpo_pairs])

    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=dcfg["epochs"],
        learning_rate=dcfg["learning_rate"],
        beta=dcfg["beta"],
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=_batch_split(dcfg["effective_batch_size"],
                                                 per_device_batch_size),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=training_cfg.get("seed", 0),
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=dataset,
        processing_class=tokenizer, peft_config=build_lora_config(training_cfg, "dpo"),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir


def train_sft(training_cfg: dict, sft_examples: list[dict], out_dir: str | Path,
              per_device_batch_size: int = 1) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base = model_info(training_cfg["base_model"])["hf_id"]
    scfg = training_cfg["sft"]
    out_dir = Path(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)

    # Filter out any placeholder examples (e.g. unavailable instruct mix).
    examples = [e for e in sft_examples if e.get("messages")]
    dataset = Dataset.from_list([{"messages": e["messages"]} for e in examples])

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=scfg["epochs"],
        learning_rate=scfg["learning_rate"],
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=_batch_split(scfg["effective_batch_size"],
                                                 per_device_batch_size),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=training_cfg.get("seed", 0),
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset,
        processing_class=tokenizer, peft_config=build_lora_config(training_cfg, "sft"),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir
