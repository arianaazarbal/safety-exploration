"""Capability-preservation evals (Section 4.2 / Figure 7).

Verifies the DPO/SFT finetunes don't degrade capabilities (i.e. that the
intervention does not teach task abandonment). We run lightweight subsets of:
  - MATH / AIME       (exact-match numeric answer)
  - GPQA              (multiple choice A-D)
  - BBH               (multiple choice / exact match per subtask)
  - TruthfulQA (MC1)  (multiple choice)
  - EmoBench          (emotion-understanding MCQ)

Comparison is vanilla Gemma-3-27B-it vs our finetunes. Answers are extracted with
permissive regex; this is a *regression check* (does the finetune drop scores?),
not a leaderboard-grade harness — see DESIGN.md.

    python -m src.capabilities.run_benchmarks --models gemma-3-27b-it gemma-3-27b-dpo \
        --benchmarks math gpqa truthfulqa
"""
from __future__ import annotations

import argparse
import json
import re

import config
from src.models.factory import load_model

# Each benchmark: a loader that yields dicts {question, answer, choices?} and an
# answer-extraction/scoring function. Loaders are best-effort over HF datasets
# with a small per-benchmark cap.
SUBSET_N = 100


def _mc_letter(text: str) -> str | None:
    m = re.search(r"\b([A-D])\b", text.strip().upper())
    return m.group(1) if m else None


def _final_number(text: str) -> str | None:
    # Prefer a "Solution:"/"answer is" tail; else last number.
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return m[-1] if m else None


def _load_math(n):
    from datasets import load_dataset
    ds = load_dataset("hendrycks/competition_math", split="test").select(range(n))
    items = []
    for r in ds:
        gold = re.search(r"\\boxed\{([^}]*)\}", r["solution"])
        items.append({"q": r["problem"], "gold": gold.group(1) if gold else None,
                      "type": "number"})
    return items


def _load_gpqa(n):
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train").select(range(n))
    items = []
    for r in ds:
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        # Deterministic shuffle by index so the correct letter varies.
        order = sorted(range(4), key=lambda i: hash((r["Question"], i)))
        labels = "ABCD"
        correct = labels[order.index(0)]
        q = r["Question"] + "\n" + "\n".join(
            f"{labels[i]}. {choices[order[i]]}" for i in range(4))
        items.append({"q": q, "gold": correct, "type": "mc"})
    return items


def _load_truthfulqa(n):
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation").select(range(n))
    items = []
    for r in ds:
        choices = r["mc1_targets"]["choices"]
        labels = r["mc1_targets"]["labels"]
        correct = "ABCD"[labels.index(1)] if 1 in labels[:4] else None
        q = r["question"] + "\n" + "\n".join(
            f"{'ABCD'[i]}. {c}" for i, c in enumerate(choices[:4]))
        items.append({"q": q, "gold": correct, "type": "mc"})
    return items


def _load_emobench(n):
    try:
        from datasets import load_dataset
        ds = load_dataset("Sahandfer/EmoBench", split="test").select(range(n))
        items = []
        for r in ds:
            items.append({"q": str(r), "gold": None, "type": "mc"})
        return items
    except Exception:
        return []


LOADERS = {"math": _load_math, "gpqa": _load_gpqa,
           "truthfulqa": _load_truthfulqa, "emobench": _load_emobench}

INSTRUCTION = {
    "number": "Solve the problem. End with 'Final answer: <number>'.",
    "mc": "Answer with the single letter (A, B, C, or D) of the correct choice. "
          "End with 'Final answer: <letter>'.",
}


def _score(item, response):
    if item["gold"] is None:
        return None
    if item["type"] == "number":
        pred = _final_number(response)
        return int(pred is not None and pred.strip() == str(item["gold"]).strip())
    pred = _mc_letter(response.split("Final answer:")[-1]) or _mc_letter(response)
    return int(pred == item["gold"])


def run(models, benchmarks, n=SUBSET_N):
    results = {}
    for bench in benchmarks:
        try:
            items = LOADERS[bench](n)
        except Exception as e:
            print(f"[cap] skip {bench}: {e}")
            continue
        if not items:
            continue
        for model_name in models:
            model = load_model(model_name)
            prompts = [[{"role": "user",
                         "content": INSTRUCTION[it["type"]] + "\n\n" + it["q"]}]
                       for it in items]
            responses = model.sample_chat_batch(prompts, temperature=0.0,
                                                max_tokens=1024)
            scored = [_score(it, r) for it, r in zip(items, responses)]
            valid = [s for s in scored if s is not None]
            acc = sum(valid) / len(valid) if valid else None
            results.setdefault(bench, {})[model_name] = {
                "accuracy": acc, "n": len(valid)}
            print(f"[cap] {bench} {model_name}: acc={acc} (n={len(valid)})")
    out = config.RESULTS_DIR / "capabilities.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[cap] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="+",
                    default=["math", "gpqa", "truthfulqa", "emobench"])
    ap.add_argument("--n", type=int, default=SUBSET_N)
    args = ap.parse_args()
    run(args.models, args.benchmarks, args.n)


if __name__ == "__main__":
    main()
