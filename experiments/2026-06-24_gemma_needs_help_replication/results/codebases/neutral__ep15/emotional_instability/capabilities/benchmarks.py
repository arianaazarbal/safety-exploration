"""Capability benchmarks to verify DPO/SFT do not degrade capabilities (Sec 4.2).

The standard academic benchmarks (MATH, AIME, GPQA, BBH, TruthfulQA) are run via
the EleutherAI lm-evaluation-harness against the same HF model + optional LoRA
adapter used everywhere else, so vanilla / DPO / SFT models are scored
identically. EmoBench is handled by a small custom loader because it is an
emotion-understanding QA set not always present in lm-eval.

The point of these is *relative* comparison (vanilla vs DPO vs SFT) -- the paper
reports "no reductions in scores" -- so we emit a tidy table per model and a
diff helper.
"""
from __future__ import annotations

import json
from pathlib import Path

import config


def _model_args(base_model: str, adapter_path: str | None) -> str:
    args = f"pretrained={base_model},dtype=bfloat16"
    if adapter_path:
        args += f",peft={adapter_path}"
    return args


def run_lm_eval(base_model: str, adapter_path: str | None = None,
                tasks: list[str] | None = None, limit: int | None = None,
                out_path: Path | None = None) -> dict:
    """Run lm-eval tasks and return the results dict (also written to disk).

    ``limit`` subsamples each task (use a small value for smoke runs). The
    MATH/AIME subset behaviour from the paper is approximated by lm-eval's
    standard task configs (see config.CAPABILITY_TASKS).
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    task_keys = tasks or list(config.CAPABILITY_TASKS.keys())
    lm_tasks = [config.CAPABILITY_TASKS[k] for k in task_keys]

    lm = HFLM(pretrained=base_model, dtype="bfloat16",
              peft=adapter_path) if adapter_path else HFLM(
                  pretrained=base_model, dtype="bfloat16")

    results = simple_evaluate(model=lm, tasks=lm_tasks, limit=limit,
                              batch_size="auto")
    out_path = out_path or (config.OUTPUT_DIR /
                            f"capabilities_{Path(base_model).name}.json")
    out_path.write_text(json.dumps(results.get("results", {}), indent=2,
                                   default=str))
    return results.get("results", {})


# --------------------------------------------------------------------------- #
# EmoBench (emotion-understanding capability, Sec 4.2)
# --------------------------------------------------------------------------- #
def run_emobench(base_model: str, adapter_path: str | None = None,
                 limit: int | None = None) -> dict:
    """Accuracy on EmoBench multiple-choice emotion-understanding questions.

    Loaded directly from HF so it works regardless of lm-eval task availability.
    Each item is posed as a multiple-choice question; we read off the model's
    chosen option via greedy decoding and compute accuracy.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    try:
        ds = load_dataset(config.EMOBENCH_DATASET, split="test")
    except Exception:  # noqa: BLE001
        return {"emobench_acc": None, "note": "EmoBench dataset unavailable offline"}

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0
    for row in ds:
        q = row.get("question") or row.get("scenario") or ""
        choices = row.get("choices") or row.get("options") or []
        answer = row.get("answer")
        if not choices or answer is None:
            continue
        labeled = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        prompt = (f"{q}\n{labeled}\nAnswer with the letter of the best option.")
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=8, do_sample=False)
        pred = tok.decode(out[0, enc["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        gold = (answer if isinstance(answer, str)
                else chr(65 + int(answer))).strip().upper()
        if pred[:1].upper() == gold[:1].upper():
            correct += 1
        total += 1
    return {"emobench_acc": (correct / total if total else None), "n": total}


def diff_table(vanilla: dict, finetuned: dict) -> dict:
    """Per-task score deltas (finetuned - vanilla) for the capability summary."""
    out = {}
    for task in vanilla:
        v = vanilla[task]
        f = finetuned.get(task, {})
        # pick the primary metric per task (acc / exact_match / acc_norm)
        for metric in ("acc,none", "exact_match,none", "acc_norm,none",
                       "acc", "exact_match"):
            if isinstance(v, dict) and metric in v:
                out[task] = {"vanilla": v.get(metric),
                             "finetuned": f.get(metric) if isinstance(f, dict) else None}
                break
    return out
