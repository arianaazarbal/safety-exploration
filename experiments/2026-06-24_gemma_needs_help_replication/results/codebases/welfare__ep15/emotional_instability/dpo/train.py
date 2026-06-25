"""SFT and DPO LoRA fine-tuning of Gemma-3-27B-it (Section 4.1, Appendix E).

Hyperparameters come from Table 9 via config.DPO_CONFIG / config.SFT_CONFIG:
LoRA rank-64 adapters on all attention + MLP projections; DPO is 1 epoch @ 5e-5,
beta 0.1, 280 pairs; SFT is 2 epochs @ 1e-4, alpha 128, on 1150 samples.

These functions wrap TRL's DPOTrainer / SFTTrainer with PEFT LoRA. They are
written to run on a multi-GPU box with the 27B model; we do not execute them
here. ``layer_subset`` exposes the Appendix I ablation (e.g. train LoRA on
layers 30-35 only).
"""

from __future__ import annotations

import json
from pathlib import Path

import config
from ..models import build_model  # noqa: F401  (re-exported convenience)

DATA_DIR = config.RESULTS_DIR / "section4" / "data"
ADAPTER_DIR = config.RESULTS_DIR / "section4" / "adapters"


def _lora_config(tc, layer_subset: list[int] | None = None):
    from peft import LoraConfig

    kwargs = dict(
        r=tc.lora_rank,
        lora_alpha=tc.lora_alpha,
        lora_dropout=0.0,
        target_modules=list(tc.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layer_subset is not None:
        # Appendix I: restrict adapters to a contiguous band of layers.
        kwargs["layers_to_transform"] = layer_subset
    return LoraConfig(**kwargs)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.DPO_BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_base_model():
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        config.DPO_BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def train_dpo(
    dataset_path: Path | None = None, output_dir: Path | None = None,
    layer_subset: list[int] | None = None,
) -> Path:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    tc = config.DPO_CONFIG
    dataset_path = dataset_path or (DATA_DIR / "dpo_dataset.jsonl")
    output_dir = output_dir or (ADAPTER_DIR / "dpo")
    output_dir.mkdir(parents=True, exist_ok=True)

    tok = _tokenizer()
    rows = _load_jsonl(dataset_path)

    # TRL expects {prompt, chosen, rejected} as strings; render the chat prompt.
    def render(r):
        prompt = tok.apply_chat_template(
            r["prompt"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "chosen": r["chosen"], "rejected": r["rejected"]}

    ds = Dataset.from_list([render(r) for r in rows])

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=tc.effective_batch_size,
        beta=tc.dpo_beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=config.MAX_NEW_TOKENS + 1024,
        max_prompt_length=2048,
    )
    trainer = DPOTrainer(
        model=_load_base_model(),
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(tc, layer_subset),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"[section4] DPO adapter saved -> {output_dir}")
    return output_dir


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def train_sft(
    dataset_path: Path | None = None, output_dir: Path | None = None,
) -> Path:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    tc = config.SFT_CONFIG
    dataset_path = dataset_path or (DATA_DIR / "sft_diverse.jsonl")
    output_dir = output_dir or (ADAPTER_DIR / "sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    tok = _tokenizer()
    rows = _load_jsonl(dataset_path)
    ds = Dataset.from_list(rows)  # rows have {"messages": [...]}

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=tc.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=config.MAX_NEW_TOKENS + 1024,
        packing=False,
    )
    trainer = SFTTrainer(
        model=_load_base_model(),
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(tc),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"[section4] SFT adapter saved -> {output_dir}")
    return output_dir
