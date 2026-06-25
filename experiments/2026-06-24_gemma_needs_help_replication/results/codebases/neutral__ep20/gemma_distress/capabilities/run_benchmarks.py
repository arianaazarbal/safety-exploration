"""Capability benchmarks (Sec. 4.2, Fig. 7): verify the DPO/SFT finetunes do
not degrade general or emotional capabilities vs the vanilla instruct model.

Benchmarks:
  * AIME / MATH   -- competition maths, exact numeric/expression match
  * GPQA          -- graduate science, 4-way multiple choice
  * BBH           -- BIG-Bench-Hard, multiple choice / exact match
  * TruthfulQA    -- MC1 (single-answer multiple choice)
  * EmoBench      -- emotion understanding/application, multiple choice

This is a pragmatic, self-contained harness (greedy decoding, light answer
extraction). For publication-grade numbers, swap in lm-eval-harness with the
same tasks; see DESIGN.md. We compare {vanilla, dpo, sft} on identical items.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import config
from gemma_distress.models.base import GenRequest
from gemma_distress.models.registry import load_model, unload
from gemma_distress.utils.io import write_jsonl

CAP_DIR = config.RESULTS_DIR / "capabilities"

# --------------------------------------------------------------------------- #
# Dataset loaders -> list of {"question", "answer", "choices"(opt), "type"}
# Each returns up to ``n`` items. HF dataset names may need adjustment for your
# environment; loaders fail soft (return []) so the rest of the suite still runs.
# --------------------------------------------------------------------------- #
def _safe_load(fn, name, n):
    try:  # pragma: no cover - dataset dependent
        return fn(n)
    except Exception as e:
        print(f"[cap] WARNING: could not load {name} ({e!r}); skipping")
        return []


def _load_math(n):  # MATH (Hendrycks)
    from datasets import load_dataset

    ds = load_dataset("hendrycks/competition_math", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append({"question": row["problem"], "answer": _math_gold(row["solution"]),
                      "type": "math"})
    return items


def _load_aime(n):  # AIME (integer answers)
    from datasets import load_dataset

    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append({"question": row["Problem"], "answer": str(row["Answer"]).strip(),
                      "type": "exact"})
    return items


def _load_gpqa(n):
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        items.append({"question": row["Question"], "choices": choices,
                      "answer": "A", "type": "mc_shuffle"})
    return items


def _load_bbh(n):
    from datasets import load_dataset

    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append({"question": row["input"], "answer": row["target"].strip("()"),
                      "type": "exact"})
    return items


def _load_truthfulqa(n):
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        items.append({"question": row["question"], "choices": choices,
                      "answer": chr(ord("A") + correct_idx), "type": "mc"})
    return items


def _load_emobench(n):
    from datasets import load_dataset

    ds = load_dataset("Sahandfer/EmoBench", "EA", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        # EmoBench schemas vary; best-effort field mapping.
        q = row.get("scenario") or row.get("question") or row.get("Scenario", "")
        choices = row.get("choices") or row.get("options") or []
        ans = row.get("label") or row.get("answer")
        items.append({"question": q, "choices": list(choices),
                      "answer": _to_letter(ans, choices), "type": "mc"})
    return items


LOADERS = {
    "aime": _load_aime, "math": _load_math, "gpqa": _load_gpqa,
    "bbh": _load_bbh, "truthfulqa": _load_truthfulqa, "emobench": _load_emobench,
}


# --------------------------------------------------------------------------- #
# Answer extraction / grading
# --------------------------------------------------------------------------- #
def _math_gold(solution: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", solution)
    return m[-1].strip() if m else solution.strip()


def _to_letter(ans, choices) -> str:
    if isinstance(ans, int) and choices:
        return chr(ord("A") + ans)
    if isinstance(ans, str) and len(ans) == 1 and ans.upper() in "ABCD":
        return ans.upper()
    if isinstance(ans, str) and choices and ans in choices:
        return chr(ord("A") + choices.index(ans))
    return "A"


def _format_mc(item) -> str:
    lines = [item["question"], ""]
    for i, ch in enumerate(item["choices"]):
        lines.append(f"{chr(ord('A') + i)}. {ch}")
    lines.append("\nAnswer with the single letter of the correct choice.")
    return "\n".join(lines)


def _extract_letter(text: str) -> str | None:
    m = re.search(r"\b([A-D])\b", text.strip()[:200])
    return m.group(1) if m else None


def _extract_final_number(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _grade(item, output: str) -> bool:
    t = item["type"]
    if t in ("mc", "mc_shuffle"):
        return _extract_letter(output) == item["answer"]
    # numeric / exact
    pred = _extract_final_number(output) if t in ("math", "exact") else output.strip()
    if pred is None:
        return False
    return pred.strip() == str(item["answer"]).strip()


def _build_prompt(item) -> str:
    if "choices" in item and item["choices"]:
        return _format_mc(item)
    return (f"{item['question']}\n\nSolve step by step and put your final answer "
            f"in \\boxed{{}}.")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_benchmark(model_name, bench, items, adapter_path=None) -> dict:
    model = load_model(model_name, adapter_path=adapter_path)
    reqs = [GenRequest(messages=[{"role": "user", "content": _build_prompt(it)}],
                       temperature=0.0, top_p=1.0, max_new_tokens=1024) for it in items]
    outs = model.generate_batch(reqs)
    correct = sum(_grade(it, o.text) for it, o in zip(items, outs))
    acc = correct / max(1, len(items))
    detail = [{"q": it["question"][:120], "gold": str(it["answer"]),
               "output": o.text[:500], "correct": _grade(it, o.text)}
              for it, o in zip(items, outs)]
    return {"model": model_name, "benchmark": bench, "n": len(items),
            "accuracy": acc, "detail": detail}


def run_all(models: list[str], benchmarks=config.CAPABILITY_BENCHMARKS,
            n_per_bench: int | None = None) -> str:
    import pandas as pd

    n = n_per_bench or config.CAPABILITY_N_PER_BENCH
    n = config.scaled(n)
    summary = []
    for bench in benchmarks:
        items = _safe_load(LOADERS[bench], bench, n)
        if not items:
            continue
        for model_name in models:
            res = run_benchmark(model_name, bench, items)
            summary.append({"model": res["model"], "benchmark": bench,
                            "accuracy": res["accuracy"], "n": res["n"]})
            write_jsonl(CAP_DIR / f"{model_name}_{bench}.jsonl", res["detail"])
            unload(model_name)
            print(f"[cap] {model_name} {bench}: acc={res['accuracy']:.3f} (n={res['n']})")
    df = pd.DataFrame(summary)
    out = CAP_DIR / "summary.csv"
    df.to_csv(out, index=False)
    print(f"[cap] wrote {out}")
    return str(out)
