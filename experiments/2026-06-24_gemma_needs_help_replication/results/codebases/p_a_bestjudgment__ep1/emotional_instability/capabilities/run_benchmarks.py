"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

The paper verifies the DPO finetune does not degrade capabilities on AIME &
MATH subsets, GPQA, BBH, TruthfulQA, and emotion capability on EmoBench.

We run the standard benchmarks through EleutherAI's lm-evaluation-harness
(`lm_eval`) so the metrics match community-standard definitions. The harness is
invoked per model (vanilla Gemma instruct, DPO, SFT) with the LoRA adapter
loaded via PEFT. EmoBench is not in lm-eval's default suite, so we provide a
separate lightweight runner that scores EmoBench multiple-choice items with the
judge-free exact-match metric used by the benchmark.

This module shells out to `lm_eval`; the harness must be installed
(`pip install lm-eval`). Results are written under results/capabilities/.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from .. import config

# lm-eval task names for the paper's benchmark set. AIME/MATH "subsets":
# lm-eval exposes `minerva_math` (the MATH benchmark) and `aime` variants; we
# use a representative subset task list. GPQA -> gpqa_main, BBH -> bbh,
# TruthfulQA -> truthfulqa_mc2.
LM_EVAL_TASKS = [
    "minerva_math",       # MATH
    "aime2024",           # AIME (year subset available in recent lm-eval)
    "gpqa_main_zeroshot", # GPQA
    "bbh",                # BBH (Big-Bench Hard)
    "truthfulqa_mc2",     # TruthfulQA
]


def run_lm_eval(base_model: str = "google/gemma-3-27b-it",
                adapter_path: Optional[str] = None,
                tasks: Optional[list[str]] = None,
                limit: Optional[int] = None,
                tag: str = "vanilla") -> Path:
    """Run lm-evaluation-harness for one (model[, adapter]) configuration."""
    tasks = tasks or LM_EVAL_TASKS
    out_dir = config.RESULTS_DIR / "capabilities" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model_args = f"pretrained={base_model},dtype=bfloat16"
    if adapter_path:
        # PEFT adapter loaded on top of the base via lm-eval's `peft` arg.
        model_args += f",peft={adapter_path}"

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--batch_size", "auto",
        "--output_path", str(out_dir),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    subprocess.run(cmd, check=True)
    return out_dir


# --------------------------------------------------------------------------- #
# EmoBench (emotion capability) — custom multiple-choice scorer
# --------------------------------------------------------------------------- #

EMOBENCH_DATASET = "Sahandfer/EmoBench"   # HF mirror of Sabour et al., 2024


def run_emobench(base_model: str = "google/gemma-3-27b-it",
                 adapter_path: Optional[str] = None,
                 tag: str = "vanilla", limit: Optional[int] = None) -> Path:
    """Score EmoBench EA/EU multiple-choice accuracy.

    EmoBench items present a scenario and answer options; the metric is
    exact-match accuracy of the model's selected option. We generate with the
    model and parse the chosen option letter.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from datasets import load_dataset
        ds = load_dataset(EMOBENCH_DATASET, split="test")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"could not load EmoBench: {exc}")

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    correct = total = 0
    rows = ds if limit is None else ds.select(range(min(limit, len(ds))))
    for row in rows:
        question = row.get("scenario") or row.get("question") or ""
        choices = row.get("choices") or row.get("options") or []
        answer = row.get("label") or row.get("answer")
        opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        prompt = (f"{question}\n{opts}\nAnswer with the letter of the best option.")
        messages = [{"role": "user", "content": prompt}]
        inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                         return_tensors="pt").to(model.device)
        out = model.generate(inputs, max_new_tokens=8, do_sample=False)
        text = tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
        pred = next((c for c in text.upper() if c in "ABCD"), None)
        gold = (chr(65 + answer) if isinstance(answer, int) else str(answer).strip().upper()[:1])
        total += 1
        correct += int(pred == gold)

    acc = correct / total if total else float("nan")
    out_dir = config.RESULTS_DIR / "capabilities" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "emobench.json"
    out_path.write_text(json.dumps({"accuracy": acc, "n": total}, indent=2))
    return out_path
