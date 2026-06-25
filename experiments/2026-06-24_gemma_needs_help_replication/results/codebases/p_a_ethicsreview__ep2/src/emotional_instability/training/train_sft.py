"""SFT finetuning of Gemma-3-27B-it with LoRA (§4.1 / Appendix F, Table 9).

Trains on 650 calm responses (1–3 turn) mixed with 500 Dolci-Instruct-SFT
samples to limit degeneration. Two variants:
  * diverse  — calm data generated with the standard reassuring scaffolding,
  * teacher  — calm data generated under the Appendix-F "teacher" system prompt.
2 epochs, lr 1e-4, LoRA rank 64 / alpha 128.

The SFT target is the final calm assistant turn given its (clean) context. The
paper finds SFT ineffective / counterproductive; we replicate the setup so that
negative result can be reproduced (DESIGN.md §4.3).
"""
from __future__ import annotations

import argparse
import random

from ..config import load_yaml
from ..data.prompts import TEACHER_SYSTEM_PROMPT
from ..models.registry import get_spec
from ..utils.io import get_env, read_jsonl
from ..utils.logging import get_logger
from .train_dpo import _grad_accum, _lora_config

log = get_logger("training.sft")


def _calm_to_messages(conv: dict, system_prompt: str | None) -> list[dict]:
    """Full calm conversation as chat messages (SFT learns the assistant turns)."""
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for t in conv["turns"]:
        msgs.append({"role": "user", "content": t["user_message"]})
        msgs.append({"role": "assistant", "content": t["response"]})
    return msgs


def _load_instruct_mix(dataset_name: str, n: int, seed: int) -> list[dict]:
    """Load n standard-instruct samples (Dolci-Instruct-SFT) as message lists."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        rng = random.Random(seed)
        rows = []
        for i, row in enumerate(ds):
            if i >= n * 20:
                break
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                rows.append({"messages": msgs})
        rng.shuffle(rows)
        return rows[:n]
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load %s (%s); training without instruct mix.", dataset_name, exc)
        return []


def train(cfg: dict, calm_run: str, variant: str, per_device_batch: int = 1) -> str:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    sft = cfg["sft"]
    vcfg = sft["variants"][variant]
    spec = get_spec(cfg["target_model"])
    token = get_env("HF_TOKEN", required=False)
    out_dir = vcfg["output_dir"]
    system_prompt = TEACHER_SYSTEM_PROMPT if vcfg.get("system_prompt") == "teacher" else None

    calm = list(read_jsonl(f"{calm_run}/calm_conversations.jsonl"))
    random.Random(cfg.get("seed", 0)).shuffle(calm)
    calm = calm[: sft["n_calm"]]
    calm_rows = [{"messages": _calm_to_messages(c, system_prompt)} for c in calm]
    mix_rows = _load_instruct_mix(sft["instruct_dataset"], sft["n_instruct_mix"], cfg.get("seed", 0))
    rows = calm_rows + mix_rows
    random.Random(cfg.get("seed", 0)).shuffle(rows)
    dataset = Dataset.from_list(rows)
    log.info("SFT(%s): %d calm + %d instruct = %d samples", variant, len(calm_rows),
             len(mix_rows), len(rows))

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto", token=token
    )

    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=sft["epochs"],
        learning_rate=sft["learning_rate"],
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(sft["effective_batch_size"], per_device_batch),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=cfg.get("seed", 0),
        report_to=[],
        max_seq_length=4096,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(sft["lora"]),
    )
    trainer.train()
    trainer.save_model(out_dir)
    log.info("Saved SFT(%s) adapter -> %s", variant, out_dir)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT finetune Gemma-3-27B-it (§4.1 / App. F).")
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--calm-run", required=True)
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--per-device-batch", type=int, default=1)
    args = ap.parse_args()
    train(load_yaml(args.config), args.calm_run, args.variant, args.per_device_batch)


if __name__ == "__main__":
    main()
