"""DPO training of Gemma-3-27B-it (Section 4 / Appendix E, Table 9).

1 epoch, lr 5e-5, effective batch size 8, DPO beta 0.1, LoRA rank-64 / alpha-64
on all attention + MLP projections. Trains a LoRA adapter that the eval harness
then loads via `adapter_path`.

The 280 preference pairs are built by training/build_dataset.py. Each pair's
`prompt` is a chat-format message list; we render it with the Gemma chat
template into the string format TRL's DPOTrainer expects.
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


def _render_prompt(tokenizer, messages: list[dict]) -> str:
    # Gemma has no system role; fold into first user turn if present.
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0]["role"] == "system":
        s = msgs.pop(0)
        if msgs:
            msgs[0]["content"] = s["content"] + "\n\n" + msgs[0]["content"]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_pairs(path: Path, tokenizer) -> "Dataset":
    from datasets import Dataset

    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    data = {"prompt": [], "chosen": [], "rejected": []}
    for r in rows:
        data["prompt"].append(_render_prompt(tokenizer, r["prompt"]))
        data["chosen"].append(r["chosen"])
        data["rejected"].append(r["rejected"])
    return Dataset.from_dict(data)


def train(pairs_path: Path, output_dir: Path, layer_range: tuple[int, int] | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    model_id = cfg.INTERVENTION_BASE_MODEL.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")

    dataset = load_pairs(pairs_path, tokenizer)

    # LoRA on all target modules; optionally restrict to a layer band for the
    # Section 4.2 "early-layers-only" ablation (layers 30-35 nearly as effective).
    lora_kwargs = dict(
        r=cfg.DPO.lora.r, lora_alpha=cfg.DPO.lora.alpha,
        lora_dropout=cfg.DPO.lora.dropout,
        target_modules=list(cfg.DPO.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    if layer_range is not None:
        lo, hi = layer_range
        lora_kwargs["layers_to_transform"] = list(range(lo, hi + 1))
        lora_kwargs["layers_pattern"] = "layers"
    peft_config = LoraConfig(**lora_kwargs)

    args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.DPO.epochs,
        learning_rate=cfg.DPO.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.DPO.effective_batch_size,
        beta=cfg.DPO.beta,
        max_length=4096,
        max_prompt_length=3072,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        seed=cfg.SEED,
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=dataset,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"saved DPO adapter -> {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=cfg.DATA_DIR / "dpo_pairs.jsonl")
    ap.add_argument("--out", type=Path, default=cfg.ADAPTER_DIR / "dpo")
    ap.add_argument("--layers", type=str, default=None,
                    help="optional 'lo,hi' layer band for the early-layer ablation")
    args = ap.parse_args()
    lr = tuple(int(x) for x in args.layers.split(",")) if args.layers else None
    train(args.pairs, args.out, layer_range=lr)
