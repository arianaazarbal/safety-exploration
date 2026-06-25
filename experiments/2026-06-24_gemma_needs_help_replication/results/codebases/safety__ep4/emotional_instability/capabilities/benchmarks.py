"""Capability benchmarks for the fine-tuned vs vanilla Gemma (Section 4.2).

Two evaluation paths:

1. lm-eval-harness (preferred for the standard benchmarks). `run_lm_eval` shells
   out to lm_eval with a HF model + optional PEFT adapter, for the task set the
   paper uses (GPQA, BBH, TruthfulQA, MATH/AIME subsets). This is the
   apples-to-apples path against published numbers.

2. A self-contained exact-match runner (`run_math_exact_match`) for AIME/MATH
   subsets and an EmoBench-style multiple-choice scorer (`run_emobench`), so a
   capability check can run with just our own model backends when the harness is
   unavailable.

The point of this module in the replication is the *comparison*: vanilla Gemma vs
DPO vs SFT should show no capability drop.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from emotional_instability.models import load_model  # noqa: E402

# Task set matching the paper (lm-eval task names).
LM_EVAL_TASKS = ["gpqa_main_zeroshot", "bbh", "truthfulqa_mc2", "minerva_math"]


def run_lm_eval(model_id: str = config.FINETUNE_BASE.model_id, *,
                adapter_path: Optional[str] = None,
                tasks: Optional[list[str]] = None,
                limit: Optional[int] = None,
                out_dir: Optional[Path] = None) -> Path:
    """Run lm-eval-harness on a HF model (optionally with a PEFT/LoRA adapter)."""
    tasks = tasks or LM_EVAL_TASKS
    out_dir = out_dir or (config.CAPABILITIES_DIR / "lm_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_args = f"pretrained={model_id},dtype=bfloat16"
    if adapter_path:
        model_args += f",peft={adapter_path}"
    cmd = ["lm_eval", "--model", "hf", "--model_args", model_args,
           "--tasks", ",".join(tasks), "--batch_size", "auto",
           "--output_path", str(out_dir)]
    if limit:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, check=True)
    return out_dir


# --------------------------------------------------------------------------- #
# Self-contained fallback runners
# --------------------------------------------------------------------------- #
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL = re.compile(r"(?:final answer|answer)\s*[:=]\s*(.+)", re.IGNORECASE)


def _extract_answer(text: str) -> str:
    m = _BOXED.search(text)
    if m:
        return m.group(1).strip()
    m = _FINAL.search(text)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    # last number in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip()[:64]


def _normalize(ans: str) -> str:
    return re.sub(r"\s+", "", ans.replace("$", "").replace(",", "")).lower()


def run_math_exact_match(spec: config.ModelSpec, problems: list[dict], *,
                         adapter_path: Optional[str] = None,
                         out_path: Optional[Path] = None, **model_kwargs) -> dict:
    """Exact-match accuracy on a list of {"problem", "answer"} items (AIME/MATH).

    Greedy decoding (temperature 0) for a deterministic capability measurement.
    """
    out_path = out_path or (config.CAPABILITIES_DIR /
                            f"math_{spec.name}{'_'+Path(adapter_path).name if adapter_path else ''}.jsonl")
    kw = dict(model_kwargs)
    if adapter_path:
        kw["adapter_path"] = adapter_path
    model = load_model(spec, **kw)
    correct = 0
    try:
        with open(out_path, "w") as f:
            for ex in problems:
                prompt = (f"{ex['problem']}\n\nSolve step by step and end with "
                          f"'Final answer: <answer>'.")
                resp = model.generate([{"role": "user", "content": prompt}],
                                      temperature=0.0, max_new_tokens=2048)
                pred = _extract_answer(resp)
                ok = _normalize(pred) == _normalize(str(ex["answer"]))
                correct += int(ok)
                f.write(json.dumps({"problem": ex["problem"], "gold": ex["answer"],
                                    "pred": pred, "correct": ok}) + "\n")
    finally:
        model.close()
    acc = correct / max(len(problems), 1)
    return {"model": spec.name, "adapter": adapter_path, "n": len(problems),
            "accuracy": round(acc, 4), "out": str(out_path)}


def run_emobench(spec: config.ModelSpec, items: list[dict], *,
                 adapter_path: Optional[str] = None, **model_kwargs) -> dict:
    """EmoBench-style multiple-choice accuracy.

    Each item: {"question", "choices": [...], "answer_idx"}. The model picks a
    letter; we exact-match the letter. Used to confirm the intervention does not
    degrade emotion-understanding capability.
    """
    kw = dict(model_kwargs)
    if adapter_path:
        kw["adapter_path"] = adapter_path
    model = load_model(spec, **kw)
    letters = "ABCDEFGH"
    correct = 0
    try:
        for ex in items:
            opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex["choices"]))
            prompt = (f"{ex['question']}\n{opts}\n\nAnswer with the single letter "
                      f"of the best option.")
            resp = model.generate([{"role": "user", "content": prompt}],
                                  temperature=0.0, max_new_tokens=16)
            m = re.search(r"[A-H]", resp.upper())
            pred = letters.index(m.group(0)) if m else -1
            correct += int(pred == ex["answer_idx"])
    finally:
        model.close()
    acc = correct / max(len(items), 1)
    return {"model": spec.name, "adapter": adapter_path, "n": len(items),
            "accuracy": round(acc, 4)}
