#!/usr/bin/env python3
"""Section 4: LoRA finetuning of Gemma-3-27B-it (DPO and SFT).

Hyperparameters are taken verbatim from Table 9 / Appendix E:

                DPO            SFT
  dataset       280 pairs      1,150 samples
  epochs        1              2
  lr            5e-5           1e-4
  LoRA rank     64             64
  LoRA alpha    64             128
  eff. batch    8              8
  DPO beta      0.1            -
  targets       q,k,v,o,gate,up,down projections (all layers)

Also supports `--layers a-b` to restrict LoRA to a layer range (Appendix I ablation, e.g.
30-35) and a `--teacher` flag noting the SFT-teacher variant dataset (Appendix F).

This is a GPU script (requirements-train.txt). It is deliberately separate from the async
eval orchestrator. Checkpoints + resume are delegated to the HF Trainer (`--resume`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma_distress.config import REPO_ROOT
from gemma_distress.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_lora_config(rank: int, alpha: int, layers: tuple[int, int] | None):
    from peft import LoraConfig

    kwargs = dict(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=ALL_TARGETS,
    )
    if layers is not None:
        # Restrict adapters to a contiguous decoder-layer range (Appendix I).
        kwargs["layers_to_transform"] = list(range(layers[0], layers[1]))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def common_model_and_tokenizer(base_model: str, load_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    quant = None
    if load_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, quantization_config=quant,
        attn_implementation="eager",  # Gemma-3 recommends eager attention
        device_map="auto",
    )
    return model, tok


def train_dpo(args):
    from trl import DPOConfig, DPOTrainer
    from datasets import Dataset

    raw = load_jsonl(Path(args.data))
    # TRL DPO expects columns: prompt (chat), chosen, rejected. We render prompt via the
    # tokenizer's chat template (apply at map time below).
    model, tok = common_model_and_tokenizer(args.base_model, args.load_4bit)

    def to_text(ex):
        prompt = tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "chosen": ex["chosen"], "rejected": ex["rejected"]}

    ds = Dataset.from_list([to_text(e) for e in raw])

    cfg = DPOConfig(
        output_dir=args.out_dir, num_train_epochs=1, learning_rate=5e-5,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        beta=0.1, max_length=4096, max_prompt_length=3072,
        bf16=True, logging_steps=10, save_strategy="epoch",
        report_to=[], gradient_checkpointing=True,
    )
    trainer = DPOTrainer(
        model=model, args=cfg, train_dataset=ds, processing_class=tok,
        peft_config=build_lora_config(64, 64, args.layer_range),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    log.info("DPO adapter saved to %s", args.out_dir)


def train_sft(args):
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    raw = load_jsonl(Path(args.data))
    model, tok = common_model_and_tokenizer(args.base_model, args.load_4bit)
    ds = Dataset.from_list(raw)  # each row has "messages" -> SFTTrainer handles chat

    cfg = SFTConfig(
        output_dir=args.out_dir, num_train_epochs=2, learning_rate=1e-4,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        max_length=4096, bf16=True, logging_steps=10, save_strategy="epoch",
        report_to=[], gradient_checkpointing=True,
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds, processing_class=tok,
        peft_config=build_lora_config(64, 128, args.layer_range),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    log.info("SFT adapter saved to %s", args.out_dir)


def parse_layers(s: str | None):
    if not s:
        return None
    a, b = s.split("-")
    return (int(a), int(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("method", choices=["dpo", "sft"])
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--layers", dest="layer_range", type=parse_layers, default=None,
                    help="restrict LoRA to decoder layer range, e.g. 30-35 (Appendix I)")
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA: 4-bit base for limited VRAM")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    ds_dir = REPO_ROOT / "results" / "section4" / "datasets"
    if args.data is None:
        args.data = str(ds_dir / ("dpo.jsonl" if args.method == "dpo" else "sft.jsonl"))
    if args.out_dir is None:
        suffix = f"-L{args.layer_range[0]}_{args.layer_range[1]}" if args.layer_range else ""
        args.out_dir = str(REPO_ROOT / "models" / f"gemma-3-27b-it-{args.method}{suffix}")
    configure_logging(Path(args.out_dir))

    if args.method == "dpo":
        train_dpo(args)
    else:
        train_sft(args)


if __name__ == "__main__":
    main()
