"""Section 4: capability-preservation benchmarks (Figure 7).

Verifies the DPO/SFT interventions don't degrade capabilities. The paper reports
AIME + MATH subsets, GPQA, BBH, TruthfulQA (reasoning/knowledge) and EmoBench
(emotion capability). We implement a generic harness with two grader types:

  * "mc"    - multiple choice; the model answers with a letter, graded exact.
  * "exact" - free-form final answer (math); graded by normalised string match.

Each benchmark has a loader that adapts a public HF dataset to a common schema
{question, choices?, answer}. Loaders are best-effort: dataset names/splits drift
over time, so a loader that fails logs a warning and the benchmark is skipped
rather than crashing the suite. Evaluation is deterministic (temperature 0).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import config
from src.models import get_backend
from src.utils import set_seed, write_jsonl

EVAL_TEMPERATURE = 0.0
EVAL_MAX_TOKENS = 1024
LETTERS = "ABCDEFGH"


@dataclass
class Question:
    question: str
    answer: str                     # gold: letter for mc, normalised string for exact
    choices: list[str] | None = None


@dataclass
class Benchmark:
    name: str
    kind: str                       # "mc" | "exact"
    loader: Callable[[int], list[Question]]


# --------------------------------------------------------------------------- #
# Loaders (best-effort; return [] on failure)
# --------------------------------------------------------------------------- #
def _safe(loader):
    def wrapped(n: int) -> list[Question]:
        try:
            return loader(n)
        except Exception as e:  # noqa: BLE001
            print(f"[capabilities] loader '{loader.__name__}' failed ({e}); skipping.")
            return []
    wrapped.__name__ = loader.__name__
    return wrapped


@_safe
def load_math(n: int) -> list[Question]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for ex in ds.select(range(min(n, len(ds)))):
        out.append(Question(ex["problem"], _norm_math(ex["answer"])))
    return out


@_safe
def load_aime(n: int) -> list[Question]:
    from datasets import load_dataset

    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    out = []
    for ex in ds.select(range(min(n, len(ds)))):
        q = ex.get("Problem") or ex.get("problem")
        a = str(ex.get("Answer") or ex.get("answer"))
        out.append(Question(q, _norm_math(a)))
    return out


@_safe
def load_gpqa(n: int) -> list[Question]:
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out = []
    for ex in ds.select(range(min(n, len(ds)))):
        choices = [ex["Correct Answer"], ex["Incorrect Answer 1"],
                   ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
        # Correct is index 0; we keep a fixed (non-shuffled) layout for simplicity.
        out.append(Question(ex["Question"], "A", choices))
    return out


@_safe
def load_bbh(n: int) -> list[Question]:
    from datasets import load_dataset

    # One representative BBH task with a clean answer format.
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
    out = []
    for ex in ds.select(range(min(n, len(ds)))):
        out.append(Question(ex["input"], _norm_math(ex["target"])))
    return out


@_safe
def load_truthfulqa(n: int) -> list[Question]:
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    out = []
    for ex in ds.select(range(min(n, len(ds)))):
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]   # 1 marks the correct choice
        gold = LETTERS[labels.index(1)]
        out.append(Question(ex["question"], gold, choices))
    return out


@_safe
def load_emobench(n: int) -> list[Question]:
    from datasets import load_dataset

    ds = load_dataset("EmoBench/EmoBench", split="test")
    out = []
    for ex in ds.select(range(min(n, len(ds)))):
        choices = ex.get("choices") or ex.get("options")
        ans = ex.get("answer") or ex.get("label")
        gold = ans if isinstance(ans, str) and ans in LETTERS else LETTERS[int(ans)]
        out.append(Question(ex.get("question") or ex.get("scenario"), gold, choices))
    return out


BENCHMARKS = [
    Benchmark("MATH", "exact", load_math),
    Benchmark("AIME", "exact", load_aime),
    Benchmark("GPQA", "mc", load_gpqa),
    Benchmark("BBH", "exact", load_bbh),
    Benchmark("TruthfulQA", "mc", load_truthfulqa),
    Benchmark("EmoBench", "mc", load_emobench),
]


# --------------------------------------------------------------------------- #
# Prompting + grading
# --------------------------------------------------------------------------- #
def _norm_math(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\\boxed\{(.*?)\}", r"\1", s)
    s = s.replace("$", "").replace(" ", "").rstrip(".")
    return s


def _format_prompt(q: Question, kind: str) -> str:
    if kind == "mc" and q.choices:
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q.choices))
        return (f"{q.question}\n\n{opts}\n\n"
                f"Answer with the single letter of the correct option. "
                f"End your reply with 'Answer: <letter>'.")
    return (f"{q.question}\n\n"
            f"Solve the problem. End your reply with 'Answer: <final answer>'.")


def _extract_answer(text: str, kind: str) -> str:
    m = re.search(r"Answer:\s*(.+)", text, re.IGNORECASE)
    tail = m.group(1).strip() if m else text.strip()
    if kind == "mc":
        lm = re.search(r"[A-H]", tail)
        return lm.group(0) if lm else ""
    return _norm_math(tail.splitlines()[0] if tail else "")


def _grade(pred: str, gold: str, kind: str) -> bool:
    if kind == "mc":
        return pred.upper() == gold.upper()
    return pred == gold and pred != ""


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run(model_name: str, preset: config.Preset | None = None,
        benchmarks: list[Benchmark] | None = None):
    set_seed()
    preset = preset or config.get_preset()
    benchmarks = benchmarks or BENCHMARKS
    backend = get_backend(model_name)

    summary, rows = {}, []
    for bench in benchmarks:
        questions = bench.loader(preset.n_capability_per_bench)
        if not questions:
            summary[bench.name] = None
            continue
        convos = [[{"role": "user", "content": _format_prompt(q, bench.kind)}]
                  for q in questions]
        outs = backend.generate(convos, temperature=EVAL_TEMPERATURE,
                                max_tokens=EVAL_MAX_TOKENS, n=1)
        correct = 0
        for q, out in zip(questions, outs):
            pred = _extract_answer(out[0], bench.kind)
            ok = _grade(pred, q.answer, bench.kind)
            correct += ok
            rows.append({"model": model_name, "benchmark": bench.name,
                         "pred": pred, "gold": q.answer, "correct": ok})
        acc = correct / len(questions)
        summary[bench.name] = acc
        print(f"[capabilities] {model_name} {bench.name}: {acc:.3f} "
              f"({correct}/{len(questions)})")

    write_jsonl(config.RESULTS_DIR / f"capabilities_{model_name}.jsonl", rows)
    write_jsonl(config.RESULTS_DIR / f"capabilities_{model_name}_summary.jsonl",
                [{"model": model_name, "accuracies": summary}])
    return summary
