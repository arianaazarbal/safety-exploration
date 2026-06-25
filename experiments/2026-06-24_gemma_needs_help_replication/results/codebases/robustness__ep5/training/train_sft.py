"""SFT training of Gemma-3-27B-it (Section 4 / Appendix E, Table 9).

2 epochs, lr 1e-4, effective batch size 8, LoRA rank-64 / alpha-128. Trains on
650 calm responses + 500 Dolci-Instruct-SFT samples (built by build_dataset.py).
The paper finds SFT is *ineffective* at reducing frustration (Section 4.2,
Appendix F); this script exists so that negative result can be reproduced, and
supports the 'teacher' system-prompt variant via --teacher.
"""
from __future__ import annotations

# --- PATH SHIM: ensure repo root is importable when run as `python training/x.py`
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

from emotional_instability import config_bridge as cfg
from emotional_instability import prompts


def load_sft(path: Path, teacher: bool) -> "Dataset":
    from datasets import Dataset

    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    out = []
    for r in rows:
        msgs = [dict(m) for m in r["messages"]]
        if teacher and r.get("source") == "calm":
            msgs = [{"role": "system", "content": prompts.TEACHER_SYSTEM_PROMPT}] + msgs
        out.append({"messages": msgs})
    return Dataset.from_list(out)


def train(data_path: Path, output_dir: Path, teacher: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    model_id = cfg.INTERVENTION_BASE_MODEL.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")

    dataset = load_sft(data_path, teacher=teacher)

    peft_config = LoraConfig(
        r=cfg.SFT.lora.r, lora_alpha=cfg.SFT.lora.alpha,
        lora_dropout=cfg.SFT.lora.dropout,
        target_modules=list(cfg.SFT.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    args = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.SFT.epochs,
        learning_rate=cfg.SFT.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.SFT.effective_batch_size,
        max_length=4096,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        seed=cfg.SEED,
        report_to="none",
        packing=False,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"saved SFT adapter -> {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=cfg.DATA_DIR / "sft_data.jsonl")
    ap.add_argument("--out", type=Path, default=cfg.ADAPTER_DIR / "sft")
    ap.add_argument("--teacher", action="store_true",
                    help="use the 'teacher' system prompt variant (Appendix F)")
    args = ap.parse_args()
    out = args.out if not args.teacher else args.out.with_name("sft_teacher")
    train(args.data, out, teacher=args.teacher)
