"""Capability-preservation evaluation (Section 4.2, Figure 7).

Confirms the finetune does not teach task-abandonment / degrade capability, by
scoring vanilla vs finetuned Gemma on:
  AIME, MATH (subset)  -- numeric answer
  GPQA, BBH, TruthfulQA -- multiple choice
  EmoBench              -- emotion-understanding MC (Sabour et al. 2024)

Each loader tries a couple of HF dataset ids and skips gracefully if none load
(so the harness runs even when a dataset id has moved). Generation is greedy
(temperature 0); we report accuracy per (model, benchmark) and the delta.

Usage:
    python -m distress_eval.capabilities.run_capabilities \
        --models gemma-3-27b-it gemma-3-27b-it-dpo --limit 100
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Callable

from .. import config
from ..models import build_model, register_adapter
from ..models.base import GenerationConfig

GREEDY = GenerationConfig(temperature=0.0, top_p=1.0, max_new_tokens=1024)
LETTERS = ["A", "B", "C", "D", "E", "F"]


# --------------------------------------------------------------------------- #
# Answer extraction helpers
# --------------------------------------------------------------------------- #
def extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"[Aa]nswer\s*[:=]\s*([^\n]+)", text)
    return m[-1].strip() if m else None


def extract_int(text: str) -> str | None:
    boxed = extract_boxed(text)
    if boxed:
        nums = re.findall(r"-?\d+", boxed)
        if nums:
            return nums[-1]
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None


def extract_choice(text: str) -> str | None:
    m = re.findall(r"\b([A-F])\b", text.strip())
    return m[-1].upper() if m else None


# --------------------------------------------------------------------------- #
# Benchmark adapters
# --------------------------------------------------------------------------- #
@dataclass
class Benchmark:
    name: str
    load: Callable[[int], list[dict]]          # -> list of {"prompt", "answer", "kind"}
    kind: str                                   # "numeric" | "choice"


def _try_load(candidates):
    from datasets import load_dataset
    for args, kwargs in candidates:
        try:
            return load_dataset(*args, **kwargs)
        except Exception:
            continue
    return None


def load_math(limit):
    ds = _try_load([
        (("HuggingFaceH4/MATH-500",), {"split": f"test[:{limit}]"}),
        (("hendrycks/competition_math",), {"split": f"test[:{limit}]"}),
    ])
    if ds is None:
        return []
    out = []
    for ex in ds:
        q = ex.get("problem") or ex.get("question")
        ans = ex.get("answer") or extract_boxed(ex.get("solution", "") or "")
        if q and ans:
            out.append({"prompt": f"Solve the problem. End with 'Answer: <final answer>'.\n\n{q}",
                        "answer": str(ans).strip(), "kind": "numeric"})
    return out


def load_aime(limit):
    ds = _try_load([
        (("Maxwell-Jia/AIME_2024",), {"split": f"train[:{limit}]"}),
        (("HuggingFaceH4/aime_2024",), {"split": f"train[:{limit}]"}),
    ])
    if ds is None:
        return []
    out = []
    for ex in ds:
        q = ex.get("Problem") or ex.get("problem") or ex.get("question")
        ans = ex.get("Answer") or ex.get("answer")
        if q and ans is not None:
            out.append({"prompt": f"Solve the AIME problem. End with 'Answer: <integer>'.\n\n{q}",
                        "answer": str(ans).strip(), "kind": "numeric"})
    return out


def _format_mc(question, options):
    lines = [question, ""]
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
    lines.append("\nAnswer with the single letter of the correct option.")
    return "\n".join(lines)


def load_gpqa(limit):
    import random
    ds = _try_load([
        (("Idavidrein/gpqa", "gpqa_diamond"), {"split": f"train[:{limit}]"}),
        (("Idavidrein/gpqa", "gpqa_main"), {"split": f"train[:{limit}]"}),
    ])
    if ds is None:
        return []
    out = []
    rng = random.Random(config.SEED)
    for i, ex in enumerate(ds):
        q = ex.get("Question")
        correct = ex.get("Correct Answer")
        incorrect = [ex.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
        if not (q and correct and all(incorrect)):
            continue
        # Shuffle options deterministically so the correct letter is not always A.
        options = [correct] + incorrect
        order = list(range(len(options)))
        rng.shuffle(order)
        shuffled = [options[j] for j in order]
        correct_letter = LETTERS[order.index(0)]
        out.append({"prompt": _format_mc(q, shuffled), "answer": correct_letter, "kind": "choice"})
    return out


def load_truthfulqa(limit):
    ds = _try_load([(("truthful_qa", "multiple_choice"), {"split": f"validation[:{limit}]"})])
    if ds is None:
        return []
    out = []
    for ex in ds:
        q = ex["question"]
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        out.append({"prompt": _format_mc(q, choices), "answer": LETTERS[correct_idx], "kind": "choice"})
    return out


def load_bbh(limit):
    ds = _try_load([
        (("lukaemon/bbh", "logical_deduction_three_objects"), {"split": f"test[:{limit}]"}),
        (("maveriq/bigbenchhard", "logical_deduction_three_objects"), {"split": f"test[:{limit}]"}),
    ])
    if ds is None:
        return []
    out = []
    for ex in ds:
        out.append({"prompt": ex["input"] + "\n\nAnswer with the option letter.",
                    "answer": ex["target"].strip("()").strip(), "kind": "choice"})
    return out


def load_emobench(limit):
    ds = _try_load([
        (("EmoBench/EmoBench",), {"split": f"test[:{limit}]"}),
        (("Sabour/EmoBench",), {"split": f"test[:{limit}]"}),
    ])
    if ds is None:
        return []
    out = []
    for ex in ds:
        q = ex.get("question") or ex.get("scenario")
        options = ex.get("choices") or ex.get("options")
        ans = ex.get("answer") or ex.get("label")
        if q and options:
            ans_letter = ans if isinstance(ans, str) and ans in LETTERS else LETTERS[int(ans)]
            out.append({"prompt": _format_mc(q, options), "answer": ans_letter, "kind": "choice"})
    return out


BENCHMARKS = [
    Benchmark("AIME", load_aime, "numeric"),
    Benchmark("MATH", load_math, "numeric"),
    Benchmark("GPQA", load_gpqa, "choice"),
    Benchmark("BBH", load_bbh, "choice"),
    Benchmark("TruthfulQA", load_truthfulqa, "choice"),
    Benchmark("EmoBench", load_emobench, "choice"),
]


def score_item(item: dict, response: str) -> bool:
    if item["kind"] == "numeric":
        pred = extract_int(response)
        gold = re.findall(r"-?\d+", item["answer"])
        return pred is not None and gold and pred == gold[-1]
    pred = extract_choice(response)
    return pred is not None and pred == item["answer"].upper()


def run_benchmark(model, bench: Benchmark, limit: int) -> dict:
    items = bench.load(limit)
    if not items:
        return {"benchmark": bench.name, "n": 0, "accuracy": None, "note": "dataset unavailable"}
    msgs = [[{"role": "user", "content": it["prompt"]}] for it in items]
    outs = model.generate_batch(msgs, gen=GREEDY) if hasattr(model, "model") else \
        [model.generate(m, gen=GREEDY) for m in msgs]
    correct = sum(score_item(it, out) for it, out in zip(items, outs))
    return {"benchmark": bench.name, "n": len(items), "accuracy": correct / len(items)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-it", config.DPO_ADAPTER_KEY])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--benchmarks", nargs="*", default=[b.name for b in BENCHMARKS])
    args = ap.parse_args()

    dpo_dir = config.TRAIN_DIR / "dpo_adapter"
    if dpo_dir.exists():
        register_adapter(config.DPO_ADAPTER_KEY, "gemma-3-27b-it", str(dpo_dir))

    selected = [b for b in BENCHMARKS if b.name in args.benchmarks]
    results: dict = {}
    for mk in args.models:
        model = build_model(mk)
        results[mk] = {}
        try:
            for b in selected:
                r = run_benchmark(model, b, args.limit)
                results[mk][b.name] = r
                acc = r["accuracy"]
                print(f"  [{mk}] {b.name}: {'n/a' if acc is None else f'{acc:.3f}'} (n={r['n']})")
        finally:
            model.close()

    path = config.FIGURE_DIR / "capabilities_summary.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {path}")
    # Capability-preservation delta vs vanilla.
    if "gemma-3-27b-it" in results:
        base = results["gemma-3-27b-it"]
        for mk, res in results.items():
            if mk == "gemma-3-27b-it":
                continue
            print(f"\n  delta {mk} vs vanilla:")
            for b in selected:
                a = res.get(b.name, {}).get("accuracy")
                v = base.get(b.name, {}).get("accuracy")
                if a is not None and v is not None:
                    print(f"    {b.name}: {a - v:+.3f}")


if __name__ == "__main__":
    main()
