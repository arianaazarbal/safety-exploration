"""LoRA SFT and DPO training of Gemma-3-27B-it (Section 4.1 / Appendix E).

Hyperparameters (Table 9):
  DPO: 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64, beta 0.1, eff bs 8
  SFT: 1150 samples, 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, eff bs 8
LoRA on all attention + MLP projections (q,k,v,o,gate,up,down).

Uses trl's SFTTrainer / DPOTrainer with peft LoraConfig. The resulting adapter
directory is passed to HFLocalClient(adapter_path=...) for re-evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]


def _lora_config(rank, alpha, target_modules):
    from peft import LoraConfig

    return LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules,
    )


def _load_jsonl(path):
    with Path(path).open() as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _batching(effective_batch_size, per_device_bs=1):
    """Resolve grad-accumulation to hit the target effective batch size."""
    grad_accum = max(1, effective_batch_size // per_device_bs)
    return per_device_bs, grad_accum


def train_sft(base_model_id, sft_jsonl, output_dir, cfg, per_device_bs=1):
    """LoRA SFT. `cfg` is config['finetune']['sft']."""
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(base_model_id)
    rows = _load_jsonl(sft_jsonl)
    # Render chat-format conversations into text via the chat template.
    texts = [tok.apply_chat_template(r["messages"], tokenize=False)
             for r in rows]
    ds = Dataset.from_dict({"text": texts})

    per_dev, grad_accum = _batching(cfg["effective_batch_size"], per_device_bs)
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=per_dev,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
    )
    trainer = SFTTrainer(
        model=base_model_id,
        args=args,
        train_dataset=ds,
        peft_config=_lora_config(cfg["lora_rank"], cfg["lora_alpha"],
                                 DEFAULT_TARGET_MODULES),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir


def train_dpo(base_model_id, dpo_jsonl, output_dir, cfg, per_device_bs=1):
    """LoRA DPO. `cfg` is config['finetune']['dpo']."""
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(base_model_id)
    rows = _load_jsonl(dpo_jsonl)

    # trl DPO expects prompt/chosen/rejected. Render the prompt messages with
    # the chat template (add_generation_prompt=True so chosen/rejected continue
    # the assistant turn).
    prompts_text, chosen, rejected = [], [], []
    for r in rows:
        p = tok.apply_chat_template(r["prompt"], tokenize=False,
                                    add_generation_prompt=True)
        prompts_text.append(p)
        chosen.append(r["chosen"])
        rejected.append(r["rejected"])
    ds = Dataset.from_dict({"prompt": prompts_text, "chosen": chosen,
                            "rejected": rejected})

    per_dev, grad_accum = _batching(cfg["effective_batch_size"], per_device_bs)
    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=per_dev,
        gradient_accumulation_steps=grad_accum,
        beta=cfg["beta"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        max_prompt_length=3072,
    )
    trainer = DPOTrainer(
        model=base_model_id,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(cfg["lora_rank"], cfg["lora_alpha"],
                                 DEFAULT_TARGET_MODULES),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
