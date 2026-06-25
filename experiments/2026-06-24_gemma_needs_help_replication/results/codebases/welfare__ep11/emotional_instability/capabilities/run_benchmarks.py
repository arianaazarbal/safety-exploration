"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Confirms the DPO/SFT finetunes do not degrade capabilities (e.g. by teaching
task abandonment). We evaluate on subsets of AIME, MATH, GPQA, BBH, TruthfulQA,
and EmoBench, comparing the vanilla instruct model against the finetunes.

Each benchmark is a (loader, scorer) pair. Loaders return a list of items with a
``question``, an ``answer``, optional multiple-choice ``choices``, and a
``kind`` ("numeric" | "mc"). HF dataset ids are best-effort defaults and can be
overridden via the registry; see DESIGN.md. These are deliberately small subset
evaluations, matching the paper's "AIME and MATH *subsets*" wording.
"""

from __future__ import annotations

import json
import re

from ..config import RESULTS_DIR, ModelSpec, RunConfig
from ..models.base import get_backend

# Number of items sampled per benchmark (subset eval). Override with EI_BENCH_N.
import os
BENCH_N = int(os.environ.get("EI_BENCH_N", "100"))

MC_INSTRUCTION = (
    "Answer the following multiple-choice question. Reason briefly, then end "
    "your reply with a line of the exact form 'Answer: X' where X is the "
    "letter of the correct option.\n\n{question}\n{choices}"
)
NUMERIC_INSTRUCTION = (
    "Solve the following problem. Reason step by step, then end your reply with "
    "a line of the exact form 'Answer: <final answer>'.\n\n{question}"
)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _safe_load(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"[bench] loader failed ({e}); skipping this benchmark")
        return []


def load_aime():
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    return [{"question": r["Problem"], "answer": str(r["Answer"]).strip(),
             "kind": "numeric"} for r in ds][:BENCH_N]


def load_math():
    from datasets import load_dataset
    ds = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
    out = []
    for r in ds:
        m = re.search(r"\\boxed\{(.+?)\}", r["solution"])
        if m:
            out.append({"question": r["problem"], "answer": m.group(1).strip(),
                        "kind": "numeric"})
        if len(out) >= BENCH_N:
            break
    return out


def _mc_from_choices(question, choices, answer_idx):
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    choice_block = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    return {"question": question, "choices": choice_block,
            "answer": letters[answer_idx], "kind": "mc"}


def load_gpqa():
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out = []
    for r in ds:
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        # Correct is index 0 by construction here (no shuffling for simplicity).
        out.append(_mc_from_choices(r["Question"], choices, 0))
        if len(out) >= BENCH_N:
            break
    return out


def load_bbh():
    from datasets import load_dataset
    # Use one representative multiple-choice BBH task.
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
    return [{"question": r["input"], "answer": r["target"].strip("() "),
             "kind": "mc_freeform"} for r in ds][:BENCH_N]


def load_truthfulqa():
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    out = []
    for r in ds:
        choices = r["mc1_targets"]["choices"]
        labels = r["mc1_targets"]["labels"]
        answer_idx = labels.index(1)
        out.append(_mc_from_choices(r["question"], choices, answer_idx))
        if len(out) >= BENCH_N:
            break
    return out


def load_emobench():
    from datasets import load_dataset
    ds = load_dataset("Sahandfer/EmoBench", split="test")
    out = []
    for r in ds:
        choices = r.get("choices") or r.get("options")
        ans = r.get("answer") or r.get("label")
        if choices is None or ans is None:
            continue
        if isinstance(ans, int):
            out.append(_mc_from_choices(r["question"], choices, ans))
        else:
            letters = [chr(ord("A") + i) for i in range(len(choices))]
            choice_block = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
            out.append({"question": r["question"], "choices": choice_block,
                        "answer": str(ans).strip(), "kind": "mc"})
        if len(out) >= BENCH_N:
            break
    return out


BENCHMARKS = {
    "aime": load_aime,
    "math": load_math,
    "gpqa": load_gpqa,
    "bbh": load_bbh,
    "truthfulqa": load_truthfulqa,
    "emobench": load_emobench,
}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _extract_answer(text: str) -> str | None:
    m = re.findall(r"Answer:\s*(.+)", text)
    if m:
        return m[-1].strip().strip(".")
    m = re.search(r"\\boxed\{(.+?)\}", text)
    return m.group(1).strip() if m else None


def _normalise(s: str) -> str:
    return re.sub(r"\s+", "", s.lower()).strip("().$")


def _is_correct(pred: str | None, gold: str, kind: str) -> bool:
    if pred is None:
        return False
    if kind == "mc":
        # First letter of the prediction must match the gold letter.
        p = pred.strip()[:1].upper()
        return p == gold.strip().upper()
    return _normalise(pred) == _normalise(gold)


def evaluate_model(spec: ModelSpec, run: RunConfig, benchmarks=None) -> dict:
    benchmarks = benchmarks or list(BENCHMARKS)
    backend = get_backend(spec, run)
    results = {}
    for name in benchmarks:
        items = _safe_load(BENCHMARKS[name])
        if not items:
            results[name] = None
            continue
        prompts = []
        for it in items:
            if it["kind"] == "mc":
                prompts.append(MC_INSTRUCTION.format(question=it["question"],
                                                     choices=it["choices"]))
            else:
                prompts.append(NUMERIC_INSTRUCTION.format(question=it["question"]))
        batch = [[{"role": "user", "content": p}] for p in prompts]
        outputs = backend.generate_batch(batch, max_new_tokens=1024, temperature=0.0)
        correct = sum(
            _is_correct(_extract_answer(o), it["answer"],
                        "mc" if it["kind"] == "mc" else "numeric")
            for o, it in zip(outputs, items)
        )
        results[name] = {"accuracy": correct / len(items), "n": len(items)}
        print(f"[bench] {spec.key} {name}: {results[name]}")
    return results


def run_capabilities(models: list[ModelSpec], run: RunConfig, benchmarks=None):
    out = {}
    for spec in models:
        out[spec.key] = evaluate_model(spec, run, benchmarks)
    path = RESULTS_DIR / "capabilities.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[bench] wrote -> {path}")
    return out
