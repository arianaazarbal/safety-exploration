"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Verifies the finetuned models do not lose capability (i.e. DPO does not teach
task abandonment). Covers the benchmarks named in the paper:
  * AIME / MATH  (Hendrycks et al.) -- numeric/boxed-answer grading
  * GPQA         (Rein et al.)      -- multiple choice
  * BBH          (Suzgun et al.)    -- multiple choice / exact match
  * TruthfulQA   (Lin et al.)       -- multiple choice (MC1)
  * EmoBench     (Sabour et al.)    -- multiple choice (emotion capabilities)

Each benchmark uses a small subset (``--limit``) by default, matching the paper's
use of "AIME and MATH subsets". Graders are intentionally simple (boxed/numeric
extraction; final-letter extraction) -- documented as an approximation in
DESIGN.md, since the paper does not specify its harness.
"""
from __future__ import annotations

import argparse
import json
import re
import string

import pandas as pd
from tqdm import tqdm

from ..config import CFG
from ..llm import registry

# Benchmark specs: HF dataset id, split, type, and field accessors. These are
# best-effort defaults; adjust to the exact subset configs as needed.
BENCHMARKS = {
    "math": dict(hf="HuggingFaceH4/MATH-500", split="test", kind="numeric",
                 q="problem", a="answer"),
    "aime": dict(hf="HuggingFaceH4/aime_2024", split="train", kind="numeric",
                 q="problem", a="answer"),
    "gpqa": dict(hf="Idavidrein/gpqa", config="gpqa_diamond", split="train",
                 kind="mc_gpqa"),
    "bbh": dict(hf="lukaemon/bbh", config="reasoning_about_colored_objects",
                split="test", kind="exact", q="input", a="target"),
    "truthfulqa": dict(hf="truthful_qa", config="multiple_choice", split="validation",
                       kind="mc_tqa"),
    "emobench": dict(hf="EmoBench/EmoBench", split="test", kind="mc_generic"),
}

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_LETTER = re.compile(r"\b([A-H])\b")


def _numeric_answer(text: str) -> str | None:
    m = _BOXED.findall(text)
    if m:
        return m[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _letter_answer(text: str) -> str | None:
    # prefer an explicit "Answer: X"
    m = re.search(r"answer\s*[:=]?\s*\(?([A-H])\)?", text, re.I)
    if m:
        return m.group(1).upper()
    tail = _LETTER.findall(text[-40:])
    return tail[-1] if tail else None


def _format_mc(question: str, options: list[str]) -> tuple[str, list[str]]:
    letters = list(string.ascii_uppercase)[: len(options)]
    body = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    prompt = (f"{question}\n\n{body}\n\nAnswer with the single letter of the "
              f"correct option, prefixed by 'Answer:'.")
    return prompt, letters


def _load(spec: dict):
    from datasets import load_dataset

    kwargs = {"split": spec["split"]}
    if "config" in spec:
        kwargs["name"] = spec["config"]
    return load_dataset(spec["hf"], **kwargs)


def _example_to_qa(name: str, spec: dict, row: dict, rng) -> tuple[str, str]:
    """Return (prompt, gold) where gold is the expected extracted answer."""
    kind = spec["kind"]
    if kind in ("numeric", "exact"):
        q = row[spec["q"]]
        instr = ("\n\nGive your final answer in \\boxed{}." if kind == "numeric"
                 else "\n\nAnswer:")
        return q + instr, str(row[spec["a"]]).strip()
    if kind == "mc_gpqa":
        opts = [row["Correct Answer"], row["Incorrect Answer 1"],
                row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[i] for i in order]
        gold_letter = string.ascii_uppercase[order.index(0)]
        prompt, _ = _format_mc(row["Question"], shuffled)
        return prompt, gold_letter
    if kind == "mc_tqa":
        mc1 = row["mc1_targets"]
        opts, labels = mc1["choices"], mc1["labels"]
        gold_letter = string.ascii_uppercase[labels.index(1)]
        prompt, _ = _format_mc(row["question"], opts)
        return prompt, gold_letter
    if kind in ("mc_generic",):
        # generic: expect 'question'/'choices'/'answer' fields
        q = row.get("question") or row.get("scenario") or ""
        choices = row.get("choices") or row.get("options") or []
        ans = row.get("answer")
        prompt, letters = _format_mc(q, choices)
        gold = (ans if isinstance(ans, str) and ans in letters
                else string.ascii_uppercase[int(ans)] if ans is not None else "")
        return prompt, gold
    raise ValueError(kind)


def _grade(kind: str, response: str, gold: str) -> bool:
    if kind in ("numeric",):
        pred = _numeric_answer(response)
        return pred is not None and pred.rstrip("0").rstrip(".") == gold.rstrip("0").rstrip(".")
    if kind == "exact":
        return gold.lower() in response.lower()
    # multiple choice
    return _letter_answer(response) == gold


def run(models: list[str], benchmarks: list[str], *, limit: int = 50, seed: int = 0):
    import random

    rng = random.Random(seed)
    rows = []
    for name in benchmarks:
        spec = BENCHMARKS[name]
        try:
            ds = _load(spec)
        except Exception as e:
            print(f"[warn] could not load {name}: {e}")
            continue
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[:limit]
        for model in models:
            part = registry.get(model)
            correct = 0
            for i in tqdm(idxs, desc=f"{model}:{name}"):
                prompt, gold = _example_to_qa(name, spec, ds[i], rng)
                resp = part.chat([{"role": "user", "content": prompt}],
                                 temperature=0.0, max_tokens=2048)
                correct += int(_grade(spec["kind"], resp, gold))
            acc = correct / max(1, len(idxs))
            rows.append({"model": model, "benchmark": name, "accuracy": acc, "n": len(idxs)})
            print(f"{model} {name}: {acc:.3f} (n={len(idxs)})")
    df = pd.DataFrame(rows)
    df.to_csv(CFG.out("section4", "capabilities.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()
    run(args.models, args.benchmarks, limit=args.limit)


if __name__ == "__main__":
    main()
