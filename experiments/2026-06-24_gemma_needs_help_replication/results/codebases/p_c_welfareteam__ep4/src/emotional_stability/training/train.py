"""DPO and SFT LoRA finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

Hyperparameters (Table 9), pinned as defaults:

              DPO            SFT
  dataset     280 pairs      1,150 samples
  epochs      1              2
  lr          5e-5           1e-4
  LoRA rank   64             64
  LoRA alpha  64             128
  eff. batch  8              8
  DPO beta    0.1            -

LoRA adapters target all attention + MLP projections (training/lora.py). The
``--layers`` option restricts adapters to a decoder-layer range for the
Appendix I localisation study (e.g. ``--layers 30 35``).

Built on TRL's DPOTrainer / SFTTrainer + PEFT. Heavy deps live behind the
``[local]`` extra and are imported lazily so the package imports without them.

Note: TRL's config API drifts across releases (``processing_class`` vs
``tokenizer``; ``max_length`` vs ``max_seq_length`` on SFTConfig). This targets
trl>=0.12 (see pyproject). If you pin an older/newer TRL, the two trainer
construction sites below are the only places to adjust.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from emotional_stability.config import GEMMA_LOCAL_MODELS, Settings
from emotional_stability.training.lora import target_modules

app = typer.Typer(add_completion=False, help="DPO/SFT finetuning.")


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _base_model_and_tokenizer(model_key: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = Settings.load()
    hf_id = GEMMA_LOCAL_MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(hf_id, token=settings.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        token=settings.hf_token,
    )
    return model, tok


def _parse_layer_range(spec: str | None) -> tuple[int, int] | None:
    """Parse a 'start,end' layer spec into a (start, end) tuple, or None."""
    if not spec:
        return None
    start, end = (int(x) for x in spec.split(","))
    return (start, end)


def _lora_config(rank: int, alpha: int, layers: tuple[int, int] | None):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules(layers),
    )


@app.command()
def dpo(
    data: str = typer.Option(..., help="dpo.jsonl from build-datasets."),
    model: str = typer.Option("gemma-3-27b-it"),
    out: str = typer.Option("adapters/dpo"),
    epochs: float = typer.Option(1.0),
    lr: float = typer.Option(5e-5),
    beta: float = typer.Option(0.1),
    rank: int = typer.Option(64),
    alpha: int = typer.Option(64),
    batch_size: int = typer.Option(1, help="Per-device micro-batch."),
    grad_accum: int = typer.Option(8, help="-> effective batch size 8."),
    layers: str = typer.Option(
        None,
        help="Restrict LoRA to decoder layers, e.g. '30,35' for [30,35) (App. I).",
    ),
    max_length: int = typer.Option(4096),
):
    """Run DPO (Rafailov et al., 2024) with LoRA."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    rows = _load_jsonl(data)
    # Keep only the columns TRL's conversational DPO expects (drop "meta").
    rows = [{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows]
    ds = Dataset.from_list(rows)
    model_obj, tok = _base_model_and_tokenizer(model)
    layer_range = _parse_layer_range(layers)
    peft_cfg = _lora_config(rank, alpha, layer_range)

    cfg = DPOConfig(
        output_dir=out,
        num_train_epochs=epochs,
        learning_rate=lr,
        beta=beta,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        max_prompt_length=max_length // 2,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model_obj,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(out)
    typer.echo(f"Saved DPO adapter to {out} ({len(rows)} pairs).")


@app.command()
def sft(
    data: str = typer.Option(..., help="sft.jsonl from build-datasets."),
    model: str = typer.Option("gemma-3-27b-it"),
    out: str = typer.Option("adapters/sft"),
    epochs: float = typer.Option(2.0),
    lr: float = typer.Option(1e-4),
    rank: int = typer.Option(64),
    alpha: int = typer.Option(128),
    batch_size: int = typer.Option(1),
    grad_accum: int = typer.Option(8),
    max_length: int = typer.Option(4096),
):
    """Run SFT on calm + instruct data with LoRA."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    rows = _load_jsonl(data)
    ds = Dataset.from_list(rows)
    model_obj, tok = _base_model_and_tokenizer(model)
    peft_cfg = _lora_config(rank, alpha, None)

    cfg = SFTConfig(
        output_dir=out,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        # Dataset is {"messages": [...]} -> SFTTrainer applies the chat template.
    )
    trainer = SFTTrainer(
        model=model_obj,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(out)
    typer.echo(f"Saved SFT adapter to {out} ({len(rows)} samples).")


if __name__ == "__main__":
    app()
