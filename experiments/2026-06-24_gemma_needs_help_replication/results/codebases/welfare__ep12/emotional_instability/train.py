"""Section 4.1 / Appendix E -- LoRA DPO and SFT finetuning of Gemma-3-27B-it.

Uses TRL's DPOTrainer / SFTTrainer with PEFT LoRA adapters. Hyperparameters
follow Table 9. The DPO layer-restriction option (Appendix I ablation) is
supported via ``DPOConfig.lora_layers``.

Heavy imports (torch, trl, peft, datasets) are deferred to call time.
"""

from __future__ import annotations

import json

from . import config


def _read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _lora_config(rank: int, alpha: int, target_modules: list[str],
                 layers: list[int] | None):
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers is not None:
        # Restrict adapters to specific transformer layers (Appendix I ablation).
        kwargs["layers_to_transform"] = list(layers)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------

def train_dpo(base_model: str = config.GEMMA_27B_IT,
              pairs_path: str = "data/dpo_pairs.jsonl",
              output_dir: str = "checkpoints/dpo-gemma-27b",
              cfg=config.DPO, per_device_batch_size: int = 1):
    """Run one epoch of LoRA DPO on the preference pairs.

    Builds the TRL-format dataset: each row needs ``prompt`` (rendered via the
    chat template up to the assistant turn), ``chosen``, ``rejected``.
    """
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    pairs = _read_jsonl(pairs_path)

    def render_prompt(prompt_messages):
        return tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)

    rows = [{
        "prompt": render_prompt(p["prompt_messages"]),
        "chosen": p["chosen"],
        "rejected": p["rejected"],
    } for p in pairs]
    dataset = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")

    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)
    training_args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg.lora_rank, cfg.lora_alpha,
                                 cfg.target_modules, cfg.lora_layers),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------

def train_sft(base_model: str = config.GEMMA_27B_IT,
              calm_path: str = "data/sft_calm.jsonl",
              output_dir: str = "checkpoints/sft-gemma-27b",
              cfg=config.SFT, per_device_batch_size: int = 1,
              system_prompt: str | None = None):
    """Run two epochs of LoRA SFT on calm data mixed with Dolci instruct data.

    ``system_prompt`` lets you reproduce the 'teacher' variant (Appendix F) by
    passing ``prompts.TEACHER_SYSTEM_PROMPT`` when generating the calm data; the
    SFT data itself is plain chat conversations.
    """
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    from .data import load_instruct_mix

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    calm = _read_jsonl(calm_path)
    instruct = load_instruct_mix(cfg.n_instruct_mix, dataset_name=cfg.instruct_dataset)
    if not instruct:
        print("WARNING: Dolci instruct-mix unavailable; training on calm data only. "
              "Degeneration mitigation (Section 4.1) will be weaker.")

    rows = [{"messages": r["messages"]} for r in calm[: cfg.n_calm]] + \
           [{"messages": r["messages"]} for r in instruct]
    dataset = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")

    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)
    training_args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg.lora_rank, cfg.lora_alpha,
                                 cfg.target_modules, None),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
