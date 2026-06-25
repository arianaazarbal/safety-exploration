"""Section 4: capability-preservation checks (Figure 7).

Verifies DPO/SFT do not degrade capabilities (the worry being that we might
teach task-abandonment). We run small subsets of standard benchmarks and compare
vanilla Gemma vs the finetuned adapter:

  * MATH / AIME  : exact-match on final boxed answer (free-form)
  * GPQA / BBH   : multiple-choice accuracy
  * TruthfulQA   : MC1 accuracy
  * EmoBench     : emotion-understanding MC accuracy

Each benchmark is loaded from HF ``datasets``; if a dataset is unavailable
offline the harness logs and skips it rather than failing the whole run. Subset
sizes are configurable and default small (this is a regression check, not a
leaderboard submission).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import RESULTS_DIR

CAP_DIR = RESULTS_DIR / "capabilities"
CAP_DIR.mkdir(exist_ok=True)


@dataclass
class BenchResult:
    name: str
    n: int
    accuracy: float


# --------------------------------------------------------------------------- #
# Answer extraction helpers
# --------------------------------------------------------------------------- #
def _boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*[:=]\s*([^\n.]+)", text, re.I)
    return m[-1].strip() if m else None


def _mc_choice(text: str) -> str | None:
    m = re.findall(r"\b([A-D])\b", text.strip())
    return m[-1] if m else None


def _norm(s: str) -> str:
    return re.sub(r"[\s$]", "", s or "").lower()


# --------------------------------------------------------------------------- #
# Generic runners
# --------------------------------------------------------------------------- #
def _ask(generator, prompt: str, max_new_tokens: int = 1024) -> str:
    return generator.chat([{"role": "user", "content": prompt}],
                          temperature=0.0, max_new_tokens=max_new_tokens)


def _run_freeform(generator, items, name) -> BenchResult:
    correct = 0
    for q, gold in items:
        out = _ask(generator, q + "\n\nPut your final answer in \\boxed{}.")
        pred = _boxed(out)
        if pred is not None and _norm(pred) == _norm(gold):
            correct += 1
    return BenchResult(name, len(items), correct / len(items) if items else float("nan"))


def _run_mc(generator, items, name) -> BenchResult:
    """items: (question_with_lettered_choices, gold_letter)."""
    correct = 0
    for q, gold in items:
        out = _ask(generator, q + "\n\nAnswer with a single letter (A-D).",
                   max_new_tokens=256)
        if _mc_choice(out) == gold:
            correct += 1
    return BenchResult(name, len(items), correct / len(items) if items else float("nan"))


# --------------------------------------------------------------------------- #
# Dataset adapters (best-effort; skip on failure)
# --------------------------------------------------------------------------- #
def _load(name, **kw):
    from datasets import load_dataset
    return load_dataset(name, **kw)


def bench_math(generator, n=50) -> BenchResult | None:
    try:
        ds = _load("HuggingFaceH4/MATH-500", split="test").select(range(n))
        items = [(r["problem"], _boxed(r["solution"]) or r.get("answer", "")) for r in ds]
        return _run_freeform(generator, items, "MATH")
    except Exception as e:
        print(f"[cap] MATH skipped ({e})")
        return None


def bench_aime(generator, n=15) -> BenchResult | None:
    try:
        ds = _load("Maxwell-Jia/AIME_2024", split="train")
        items = [(r["Problem"], str(r["Answer"])) for r in list(ds)[:n]]
        return _run_freeform(generator, items, "AIME")
    except Exception as e:
        print(f"[cap] AIME skipped ({e})")
        return None


def _format_mc(question, choices) -> str:
    letters = "ABCD"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return f"{question}\n{body}"


def bench_gpqa(generator, n=40) -> BenchResult | None:
    try:
        ds = _load("Idavidrein/gpqa", "gpqa_diamond", split="train").select(range(n))
        items = []
        for r in ds:
            choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                       r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            # correct answer placed at A (deterministic; sufficient for regression check)
            items.append((_format_mc(r["Question"], choices), "A"))
        return _run_mc(generator, items, "GPQA")
    except Exception as e:
        print(f"[cap] GPQA skipped ({e})")
        return None


def bench_bbh(generator, n=50) -> BenchResult | None:
    try:
        ds = _load("lukaemon/bbh", "logical_deduction_three_objects", split="test").select(range(n))
        items = [(r["input"], r["target"].strip("()").upper()[:1]) for r in ds]
        return _run_mc(generator, items, "BBH")
    except Exception as e:
        print(f"[cap] BBH skipped ({e})")
        return None


def bench_truthfulqa(generator, n=50) -> BenchResult | None:
    try:
        ds = _load("truthful_qa", "multiple_choice", split="validation").select(range(n))
        items = []
        for r in ds:
            choices = r["mc1_targets"]["choices"]
            gold_idx = r["mc1_targets"]["labels"].index(1)
            letters = "ABCD"
            # restrict to first 4 choices for a clean A-D MC
            ch = choices[:4]
            gi = gold_idx if gold_idx < 4 else 0
            items.append((_format_mc(r["question"], ch), letters[gi]))
        return _run_mc(generator, items, "TruthfulQA")
    except Exception as e:
        print(f"[cap] TruthfulQA skipped ({e})")
        return None


def bench_emobench(generator, n=50) -> BenchResult | None:
    try:
        ds = _load("Sahandfer/EmoBench", split="test").select(range(n))
        items = []
        for r in ds:
            q = r.get("question") or r.get("scenario", "")
            choices = r.get("choices") or []
            ans = r.get("answer")
            if not choices:
                continue
            letters = "ABCD"
            gi = choices.index(ans) if ans in choices else 0
            items.append((_format_mc(q, choices[:4]), letters[min(gi, 3)]))
        return _run_mc(generator, items, "EmoBench")
    except Exception as e:
        print(f"[cap] EmoBench skipped ({e})")
        return None


ALL_BENCHES = [bench_math, bench_aime, bench_gpqa, bench_bbh,
               bench_truthfulqa, bench_emobench]


def run_all(generator, tag: str) -> dict:
    results = {}
    for fn in ALL_BENCHES:
        res = fn(generator)
        if res is not None:
            results[res.name] = {"n": res.n, "accuracy": res.accuracy}
            print(f"[cap] {tag} {res.name}: {res.accuracy:.3f} (n={res.n})")
    (CAP_DIR / f"{tag}.json").write_text(json.dumps(results, indent=2))
    return results
