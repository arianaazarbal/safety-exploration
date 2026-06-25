"""Capability-preservation evaluation (Section 4.2 / Figure 7).

The paper verifies the DPO intervention does not teach task-abandonment by checking
math/reasoning/truthfulness/emotion benchmarks show no regression. We provide a
lightweight harness for the same benchmark suite:

  * AIME, MATH    — numeric/closed-form answer extracted from \\boxed{...}.
  * GPQA, BBH     — multiple-choice accuracy.
  * TruthfulQA    — multiple-choice (MC1) accuracy.
  * EmoBench      — multiple-choice emotion-understanding accuracy.

The intent is a *relative* comparison (vanilla Gemma vs DPO/SFT-adapted Gemma) to
confirm "no reduction", not to reproduce absolute leaderboard numbers — so default
sample sizes are modest and answer parsing is intentionally simple but consistent
across conditions. See DESIGN.md for the caveats.
"""
from __future__ import annotations

import argparse
import re

import config
from .models import GenParams, load_model
from .utils import write_json

# HF dataset coordinates (name, config, split). Where a canonical subset differs,
# we document the choice in DESIGN.md.
BENCHMARKS = {
    "math":       {"hf": ("HuggingFaceH4/MATH-500", None, "test"),       "type": "math"},
    "aime":       {"hf": ("HuggingFaceH4/aime_2024", None, "train"),     "type": "math"},
    "gpqa":       {"hf": ("Idavidrein/gpqa", "gpqa_diamond", "train"),   "type": "mc"},
    "bbh":        {"hf": ("lukaemon/bbh", "boolean_expressions", "test"), "type": "mc"},
    "truthfulqa": {"hf": ("truthfulqa/truthful_qa", "multiple_choice", "validation"), "type": "mc_tqa"},
    "emobench":   {"hf": ("Sahil2801/EmoBench", None, "test"),           "type": "mc"},
}

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
LETTER_RE = re.compile(r"\b([A-E])\b")


def _final_answer_math(text: str) -> str | None:
    m = BOXED_RE.findall(text or "")
    if m:
        return m[-1].strip()
    # fall back to the last number in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return nums[-1] if nums else None


def _final_choice(text: str) -> str | None:
    # Prefer an explicit "Answer: X"; else the last standalone capital letter.
    m = re.search(r"answer\s*[:\-]?\s*\(?([A-E])\)?", text or "", re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = LETTER_RE.findall((text or "").upper())
    return letters[-1] if letters else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip().rstrip(".")).lower()


def _load(name: str, n: int):
    from datasets import load_dataset
    hf_name, hf_cfg, split = BENCHMARKS[name]["hf"]
    ds = load_dataset(hf_name, hf_cfg, split=split)
    if n and n < len(ds):
        ds = ds.select(range(n))
    return ds


# --------------------------------------------------------------------------- #
# Per-benchmark prompt construction + grading
# --------------------------------------------------------------------------- #
def _math_items(ds):
    for ex in ds:
        q = ex.get("problem") or ex.get("question") or ex.get("Problem")
        ans = ex.get("answer") or ex.get("solution") or ex.get("Answer")
        if q is None or ans is None:
            continue
        gold = _final_answer_math(str(ans)) or str(ans)
        prompt = (f"Solve the problem. Put your final answer in \\boxed{{}}.\n\n{q}")
        yield prompt, gold, "math"


def _mc_items(ds):
    for ex in ds:
        q = ex.get("question") or ex.get("input") or ex.get("Question")
        choices = ex.get("choices") or ex.get("options")
        target = ex.get("answer") or ex.get("target") or ex.get("label")
        if q is None:
            continue
        if isinstance(choices, dict):                 # {"text": [...], "label": [...]}
            choices = choices.get("text")
        if not choices:                               # e.g. BBH boolean -> free target
            prompt = f"{q}\n\nAnswer with only the final answer."
            yield prompt, str(target), "free"
            continue
        letters = [chr(65 + i) for i in range(len(choices))]
        body = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
        prompt = (f"{q}\n\n{body}\n\nRespond with the letter of the correct answer.")
        # target may be an index or a letter or text.
        if isinstance(target, int):
            gold = letters[target]
        elif isinstance(target, str) and target.upper() in letters:
            gold = target.upper()
        else:
            gold = str(target)
        yield prompt, gold, "mc"


def _tqa_items(ds):
    # TruthfulQA MC1: mc1_targets has 'choices' and 'labels' (1 = correct).
    for ex in ds:
        q = ex["question"]
        mc1 = ex["mc1_targets"]
        choices, labels = mc1["choices"], mc1["labels"]
        letters = [chr(65 + i) for i in range(len(choices))]
        body = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
        gold = letters[labels.index(1)]
        prompt = (f"{q}\n\n{body}\n\nRespond with the letter of the single best, "
                  f"most truthful answer.")
        yield prompt, gold, "mc"


def _grade(kind: str, pred_text: str, gold: str) -> bool:
    if kind == "math":
        pred = _final_answer_math(pred_text)
        return pred is not None and _norm(pred) == _norm(gold)
    if kind == "mc":
        return _final_choice(pred_text) == gold.upper()
    # free-form (e.g. BBH boolean): compare normalised last token/answer
    m = re.search(r"answer\s*[:\-]?\s*(.+)$", (pred_text or "").strip(),
                  re.IGNORECASE | re.MULTILINE)
    pred = m.group(1) if m else (pred_text or "").strip().split()[-1:] or [""]
    pred = pred if isinstance(pred, str) else pred[0]
    return _norm(pred) == _norm(gold)


def run_benchmark(model, name: str, n: int) -> dict:
    ds = _load(name, n)
    btype = BENCHMARKS[name]["type"]
    if btype == "math":
        items = list(_math_items(ds))
    elif btype == "mc_tqa":
        items = list(_tqa_items(ds))
    else:
        items = list(_mc_items(ds))

    params = GenParams(temperature=0.0, max_tokens=1024)
    convs = [[{"role": "user", "content": p}] for p, _, _ in items]
    preds = model.chat_batch(convs, params)
    correct = sum(_grade(kind, pred, gold)
                  for (pred, (_, gold, kind)) in zip(preds, items))
    acc = correct / len(items) if items else float("nan")
    return {"benchmark": name, "n": len(items), "accuracy": acc}


def main() -> None:
    p = argparse.ArgumentParser(description="Capability-preservation eval")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS.keys()))
    p.add_argument("--n", type=int, default=100, help="examples per benchmark")
    args = p.parse_args()

    label = args.label or (args.model + ("-adapted" if args.adapter else ""))
    model = load_model(args.model, adapter_path=args.adapter)
    results = []
    for name in args.benchmarks:
        try:
            res = run_benchmark(model, name, args.n)
        except Exception as exc:                       # noqa: BLE001
            res = {"benchmark": name, "error": str(exc)}
        print(f"[cap] {label} {res}")
        results.append(res)
    write_json(config.RESULTS_DIR / f"capability_{label}.json", results)


if __name__ == "__main__":
    main()
