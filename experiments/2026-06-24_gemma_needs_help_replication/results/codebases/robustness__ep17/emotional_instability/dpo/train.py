"""LoRA SFT and DPO training of Gemma-3-27B-it (paper Section 4.1 / Appendix E).

Hyperparameters follow Table 9:

* **DPO** — 280 pairs, 1 epoch, lr 5e-5, LoRA r=64 / alpha=64, beta=0.1,
  effective batch size 8.
* **SFT** — 650 calm + 500 instruct samples, 2 epochs, lr 1e-4, LoRA r=64 /
  alpha=128, effective batch size 8.

LoRA targets all attention + MLP projections. The internal-vs-expressed ablation
(Section 4.2) is supported via ``config.DPO.lora_layers_subset`` — e.g. restrict
adapters to layers 30-35 (nearly as effective) or >=40 (ineffective).

Adapters are written under ``artifacts/`` and can be loaded for evaluation by
passing ``adapter_path`` to ``eval.evaluate_model``.
"""

from __future__ import annotations

from pathlib import Path

import config
from emotional_instability.models.base import Message
from emotional_instability.utils import log, read_jsonl


# --------------------------------------------------------------------------- #
# LoRA config (optionally restricted to a layer range for the ablation)
# --------------------------------------------------------------------------- #
def _lora_config(rank: int, alpha: int):
    from peft import LoraConfig

    target_modules = config.LORA_TARGET_MODULES
    layers_to_transform = None
    subset = config.DPO.lora_layers_subset
    if subset is not None:
        lo, hi = subset
        layers_to_transform = list(range(lo, hi + 1))
        log.info("Restricting LoRA to layers %d-%d (internal-emotion ablation)", lo, hi)
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
    )


def _load_base():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = config.MODELS[config.INTERVENTION_BASE_MODEL].model_id
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tok


def _render_prompt(tok, messages: list[Message]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def train_dpo(
    pairs_path: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    from datasets import Dataset
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    pairs_path = pairs_path or (config.ARTIFACTS_DIR / "dpo_pairs.jsonl")
    out_dir = out_dir or (config.ARTIFACTS_DIR / "gemma-3-27b-it-dpo")
    rows = read_jsonl(pairs_path)
    if not rows:
        raise RuntimeError(f"No DPO pairs at {pairs_path}; run generate_data first.")

    model, tok = _load_base()
    ds = Dataset.from_list([
        {
            "prompt": _render_prompt(tok, r["prompt"]),
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }
        for r in rows
    ])

    args = TRLDPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.DPO.effective_batch_size,
        beta=config.DPO.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(config.DPO.lora_rank, config.DPO.lora_alpha),
    )
    log.info("Starting DPO: %d pairs, %d epoch(s), lr=%g, beta=%g",
             len(rows), config.DPO.epochs, config.DPO.learning_rate, config.DPO.beta)
    trainer.train()
    trainer.save_model(str(out_dir))
    log.info("Saved DPO adapter -> %s", out_dir)
    return out_dir


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def _load_instruct_mix(n: int, tok) -> list[dict]:
    """Load instruct samples to mix in (Dolci-Instruct-SFT) to avoid degeneration."""
    try:
        from datasets import load_dataset

        ds = load_dataset(config.INSTRUCT_MIX_DATASET, split=f"train[:{n}]")
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            # Expect alternating user/assistant; take first user + first assistant.
            user = next((m["content"] for m in msgs if m["role"] == "user"), None)
            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if user and asst:
                out.append({"prompt": [{"role": "user", "content": user}], "completion": asst})
        return out[:n]
    except Exception as e:  # noqa: BLE001
        log.warning("Could not load %s (%s); SFT will use calm data only.",
                    config.INSTRUCT_MIX_DATASET, e)
        return []


def train_sft(
    sft_path: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    from datasets import Dataset
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    sft_path = sft_path or (config.ARTIFACTS_DIR / "sft_dataset.jsonl")
    out_dir = out_dir or (config.ARTIFACTS_DIR / "gemma-3-27b-it-sft")
    calm = read_jsonl(sft_path)
    if not calm:
        raise RuntimeError(f"No SFT data at {sft_path}; run generate_data first.")

    model, tok = _load_base()
    mix = _load_instruct_mix(config.SFT.instruct_mix_samples, tok)
    combined = calm[: config.SFT.calm_samples] + mix

    def _to_text(r: dict) -> dict:
        prompt = _render_prompt(tok, r["prompt"])
        return {"text": prompt + r["completion"] + (tok.eos_token or "")}

    ds = Dataset.from_list([_to_text(r) for r in combined])

    args = TRLSFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.SFT.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(config.SFT.lora_rank, config.SFT.lora_alpha),
    )
    log.info("Starting SFT: %d samples (%d calm + %d instruct), %d epochs, lr=%g",
             len(combined), len(calm[: config.SFT.calm_samples]), len(mix),
             config.SFT.epochs, config.SFT.learning_rate)
    trainer.train()
    trainer.save_model(str(out_dir))
    log.info("Saved SFT adapter -> %s", out_dir)
    return out_dir
