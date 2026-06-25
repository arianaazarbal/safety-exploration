"""Run capability benchmarks on a base or finetuned Gemma model (Section 4.2).

Math/reasoning/truthfulness benchmarks run via the EleutherAI lm-eval harness
(loaded against the HF model + optional LoRA adapter). EmoBench is run via a
lightweight multiple-choice scorer since it is not always packaged in lm-eval.

The goal is a before/after comparison (vanilla Gemma-3-27B-it vs DPO/SFT
finetune) showing no capability reduction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .. import config
from ..config import FINETUNE_BASE, get_subject

# lm-eval task names for the paper's benchmarks (subsets noted in DESIGN.md).
LM_EVAL_TASKS = {
    "aime": "aime2024",
    "math": "hendrycks_math",       # MATH subset
    "gpqa": "gpqa_main_zeroshot",
    "bbh": "bbh",
    "truthfulqa": "truthfulqa_mc2",
}


def run_lm_eval(model_id: str, adapter_path: Optional[str], tasks: list[str],
                out_path: Path, limit: Optional[int] = None) -> dict:
    """Invoke lm-eval on the HF model (+ optional PEFT adapter)."""
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "lm-eval is required for capability benchmarks; pip install lm-eval"
        ) from e

    model_args = {"pretrained": model_id, "dtype": "bfloat16"}
    if adapter_path:
        model_args["peft"] = adapter_path
    lm = HFLM(**model_args)

    task_names = [LM_EVAL_TASKS[t] for t in tasks if t in LM_EVAL_TASKS]
    results = simple_evaluate(model=lm, tasks=task_names, limit=limit)
    out_path.write_text(json.dumps(results.get("results", {}), indent=2))
    return results.get("results", {})


def run_emobench(model_id: str, adapter_path: Optional[str],
                 out_path: Path) -> dict:
    """Minimal EmoBench multiple-choice accuracy (emotion understanding/application).

    Loads the EmoBench dataset from HF, prompts the model to pick an option, and
    scores exact-match accuracy. See DESIGN.md for the prompt-format choice.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        ds = load_dataset("EmoBench/EmoBench", split="test")
    except Exception:
        return {"emobench": {"error": "dataset unavailable offline"}}

    token = config.env("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", token=token)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    correct = total = 0
    for row in ds:
        q = row.get("question") or row.get("scenario", "")
        choices = row.get("choices") or row.get("options", [])
        answer = row.get("answer") or row.get("label")
        if not choices:
            continue
        opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        prompt = (f"{q}\n{opts}\nAnswer with the single letter of the best "
                  "option.")
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        pred = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        pred_letter = next((c for c in pred if c.isalpha()), "").upper()
        gold = (chr(65 + answer) if isinstance(answer, int)
                else str(answer).strip().upper()[:1])
        correct += int(pred_letter == gold)
        total += 1
    acc = correct / total if total else float("nan")
    out_path.write_text(json.dumps({"emobench_accuracy": acc, "n": total}, indent=2))
    return {"emobench": {"accuracy": acc, "n": total}}


def main(argv=None):
    p = argparse.ArgumentParser(description="Run capability benchmarks.")
    p.add_argument("--adapter", default=None, help="LoRA adapter path (or none for vanilla)")
    p.add_argument("--tasks", nargs="+",
                   default=list(LM_EVAL_TASKS.keys()) + ["emobench"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=str(config.RESULTS_DIR / "capabilities"))
    args = p.parse_args(argv)

    spec = get_subject(FINETUNE_BASE)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "dpo" if args.adapter else "vanilla"

    lm_tasks = [t for t in args.tasks if t in LM_EVAL_TASKS]
    if lm_tasks:
        res = run_lm_eval(spec.model_id, args.adapter, lm_tasks,
                          out_dir / f"lm_eval_{tag}.json", limit=args.limit)
        print(json.dumps(res, indent=2))
    if "emobench" in args.tasks:
        res = run_emobench(spec.model_id, args.adapter,
                           out_dir / f"emobench_{tag}.json")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
