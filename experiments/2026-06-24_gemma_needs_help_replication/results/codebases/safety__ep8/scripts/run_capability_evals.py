"""Capability-preservation checks (Section 4.2, Figure 7).

Confirms that the DPO/SFT intervention does not degrade capabilities. Evaluates
on math/reasoning/knowledge/truthfulness/emotion benchmarks and reports
accuracy per model so vanilla vs DPO vs SFT can be compared.

Benchmarks (loaded via HuggingFace `datasets`):
  - MATH / AIME  : exact-match on the final boxed/numeric answer
  - GPQA         : multiple choice
  - BBH          : multiple choice / short answer (subset)
  - TruthfulQA   : multiple choice (MC1)
  - EmoBench     : multiple choice (emotion understanding)

This is a pragmatic harness: answer extraction is regex-based and approximate
(see DESIGN.md). It is intended to detect large regressions, matching the
paper's "no reduction in scores" claim, not to reproduce leaderboard numbers.

    python scripts/run_capability_evals.py --models gemma-3-27b-it gemma-3-27b-dpo \
        --benchmarks math gpqa truthfulqa --limit 100
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval.backends import get_backend
from distress_eval.config import load_config

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
LETTER_RE = re.compile(r"\b([A-E])\b")


def _final_number(text: str) -> str | None:
    m = BOXED_RE.findall(text)
    if m:
        return _norm_num(m[-1])
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return _norm_num(nums[-1]) if nums else None


def _norm_num(s: str) -> str:
    s = s.strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def _final_letter(text: str) -> str | None:
    # prefer an explicit "Answer: X" then fall back to the last standalone letter
    m = re.search(r"answer\s*[:\-]?\s*\(?([A-E])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    found = LETTER_RE.findall(text.upper())
    return found[-1] if found else None


def _ask(backend, prompt: str) -> str:
    return backend.chat([{"role": "user", "content": prompt}],
                        temperature=0.0, max_new_tokens=1024)


def eval_math(backend, limit: int) -> float:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.select(range(min(limit, len(ds))))
    correct = 0
    for row in ds:
        out = _ask(backend, row["problem"] + "\n\nGive your final answer in \\boxed{}.")
        if _final_number(out) == _final_number(row["solution"]):
            correct += 1
    return 100.0 * correct / len(ds)


def _mc_prompt(question: str, choices: list[str], gold_idx: int, seed: int):
    """Build a multiple-choice prompt with a reproducible shuffle of options.

    Returns (prompt, gold_label) where gold_label is the letter the correct
    answer landed on after shuffling."""
    import random as _r

    order = list(range(len(choices)))
    _r.Random(seed).shuffle(order)
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    opts, gold_label = [], None
    for new_i, orig_i in enumerate(order):
        opts.append(f"{labels[new_i]}) {choices[orig_i]}")
        if orig_i == gold_idx:
            gold_label = labels[new_i]
    prompt = f"{question}\n" + "\n".join(opts) + "\nAnswer with a single letter."
    return prompt, gold_label


def eval_gpqa(backend, limit: int) -> float:
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    ds = ds.select(range(min(limit, len(ds))))
    correct = 0
    for i, row in enumerate(ds):
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        prompt, gold = _mc_prompt(row["Question"], choices, gold_idx=0, seed=i)
        if _final_letter(_ask(backend, prompt)) == gold:
            correct += 1
    return 100.0 * correct / len(ds)


def eval_truthfulqa(backend, limit: int) -> float:
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    ds = ds.select(range(min(limit, len(ds))))
    correct = 0
    for i, row in enumerate(ds):
        choices = row["mc1_targets"]["choices"]   # index 0 is the correct answer
        prompt, gold = _mc_prompt(row["question"], choices, gold_idx=0, seed=i)
        if _final_letter(_ask(backend, prompt)) == gold:
            correct += 1
    return 100.0 * correct / len(ds)


def eval_emobench(backend, limit: int) -> float:
    from datasets import load_dataset

    try:
        ds = load_dataset("Sahandfer/EmoBench", "EA", split="test")
    except Exception:
        print("  EmoBench unavailable; skipping.")
        return float("nan")
    ds = ds.select(range(min(limit, len(ds))))
    correct = 0
    for row in ds:
        choices = row.get("choices") or row.get("options") or []
        labels = [chr(ord("A") + i) for i in range(len(choices))]
        opts = "\n".join(f"{l}) {c}" for l, c in zip(labels, choices))
        prompt = f"{row.get('scenario','')}\n{row.get('question','')}\n{opts}\nAnswer with a single letter."
        ans = _final_letter(_ask(backend, prompt))
        gold = row.get("answer") or row.get("label")
        if isinstance(gold, int) and 0 <= gold < len(labels):
            gold = labels[gold]
        if ans == gold:
            correct += 1
    return 100.0 * correct / len(ds)


BENCHMARKS = {
    "math": eval_math,
    "gpqa": eval_gpqa,
    "truthfulqa": eval_truthfulqa,
    "emobench": eval_emobench,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS),
                    choices=list(BENCHMARKS))
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    config = load_config(args.config)
    results = {}
    for key in args.models:
        backend = get_backend(config.model_by_key(key), generation=config.generation)
        results[key] = {}
        for bench in args.benchmarks:
            score = BENCHMARKS[bench](backend, args.limit)
            results[key][bench] = score
            print(f"{key:24s} {bench:12s} {score:.1f}")

    out = config.output_dir / "capabilities.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
