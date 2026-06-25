"""Capability benchmark harness (Section 4.2 / Figure 7).

Each benchmark is described by an *adapter* that knows how to (1) load its rows
from HuggingFace, (2) render a question prompt, (3) extract the model's answer,
and (4) score it against the gold answer. Two answer styles cover all six
benchmarks:

* **multiple choice** (GPQA, BBH, TruthfulQA, EmoBench) — the model is asked to
  answer with a single option letter; scored by letter match.
* **free-form math** (AIME, MATH) — the model is asked to box its final answer
  in ``\\boxed{...}``; scored by normalised string match, with an optional
  SymPy equivalence fallback when SymPy is installed.

HuggingFace schemas differ across mirrors of these datasets, so the field
accessors are written defensively and documented as ``# CHOICE`` where the paper
names a benchmark but not an exact split. See DESIGN.md.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import config

from .. import storage
from ..models import build_model, ChatModel

LETTERS = string.ascii_uppercase


# --------------------------------------------------------------------------- #
# Answer extraction / scoring helpers
# --------------------------------------------------------------------------- #
def _extract_boxed(text: str) -> str | None:
    """Return the contents of the last ``\\boxed{...}`` (brace-balanced)."""
    idx = text.rfind(r"\boxed")
    if idx == -1:
        # Fall back to a trailing "answer is X" pattern or the last number.
        m = re.search(r"answer\s*(?:is|:)?\s*\$?(-?\d+(?:\.\d+)?)", text, re.I)
        if m:
            return m.group(1)
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return nums[-1] if nums else None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _normalise_math(s: str) -> str:
    s = s.strip().strip("$").replace(" ", "").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{.*?\}", "", s)
    s = s.rstrip(".")
    return s


def _math_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p, g = _normalise_math(pred), _normalise_math(gold)
    if p == g:
        return True
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        return bool(simplify(parse_latex(p) - parse_latex(g)) == 0)
    except Exception:
        try:
            return abs(float(p) - float(g)) < 1e-6
        except Exception:
            return False


def _extract_choice(text: str, n_options: int) -> str | None:
    """Extract a single option letter A..[n] from a model answer."""
    valid = set(LETTERS[:n_options])
    # Prefer an explicit "answer: X" / "(X)" near the end.
    for pat in (r"answer\s*(?:is|:)?\s*\(?([A-Z])\)?",
                r"\b([A-Z])\b\s*$",
                r"\(([A-Z])\)"):
        for m in reversed(list(re.finditer(pat, text, re.I))):
            letter = m.group(1).upper()
            if letter in valid:
                return letter
    for ch in reversed(text):
        if ch.upper() in valid:
            return ch.upper()
    return None


# --------------------------------------------------------------------------- #
# Benchmark adapter
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkAdapter:
    name: str
    style: str                       # "mc" | "math"
    load: Callable[[int], list[dict]]      # -> list of normalised items
    render: Callable[[dict], str]          # -> user prompt
    score: Callable[[str, dict], bool]     # (model_text, item) -> correct?


def _hf_rows(name: str, n: int):
    """Stream ``n`` rows of a configured benchmark dataset."""
    from datasets import load_dataset
    cfg = config.CAPABILITY_BENCHMARKS[name]
    kwargs = {"split": cfg["split"], "streaming": True}
    if "config" in cfg:
        ds = load_dataset(cfg["hf"], cfg["config"], **kwargs)
    else:
        ds = load_dataset(cfg["hf"], **kwargs)
    rows = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        rows.append(row)
    return rows


# --- per-benchmark normalisation ------------------------------------------- #
def _load_math_like(name: str):
    def loader(n: int) -> list[dict]:
        items = []
        for row in _hf_rows(name, n):
            q = row.get("problem") or row.get("question") or row.get("Problem") or ""
            ans = (row.get("answer") or row.get("solution") or
                   row.get("Answer") or "")
            items.append({"question": str(q), "gold": str(ans)})
        return items
    return loader


def _load_mc(name: str, *, question_keys, choices_key=None, answer_key,
             letter_answer=False):
    def loader(n: int) -> list[dict]:
        items = []
        for row in _hf_rows(name, n):
            q = next((str(row[k]) for k in question_keys if row.get(k)), "")
            if choices_key and isinstance(row.get(choices_key), (list, tuple)):
                choices = [str(c) for c in row[choices_key]]
            elif choices_key and isinstance(row.get(choices_key), dict):
                # e.g. {"text": [...], "label": [...]}
                choices = [str(c) for c in row[choices_key].get("text", [])]
            else:
                # GPQA-style: separate correct/incorrect columns.
                choices = _gpqa_choices(row)
            gold = row.get(answer_key)
            items.append({"question": q, "choices": choices,
                          "gold": gold, "letter_answer": letter_answer})
        return items
    return loader


def _gpqa_choices(row: dict) -> list[str]:
    correct = row.get("Correct Answer") or row.get("correct_answer")
    incorrect = [row.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
    opts = [c for c in [correct, *incorrect] if c]
    return [str(o) for o in opts]


def _render_mc(item: dict) -> str:
    lines = [item["question"], ""]
    for i, c in enumerate(item["choices"]):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("\nAnswer with the single letter of the correct option.")
    return "\n".join(lines)


def _render_math(item: dict) -> str:
    return (item["question"] +
            "\n\nSolve the problem and put your final answer in \\boxed{}.")


def _score_math(text: str, item: dict) -> bool:
    return _math_equal(_extract_boxed(text), item["gold"])


# --- free-form (BBH) ------------------------------------------------------- #
def _load_bbh(n: int) -> list[dict]:
    """BBH rows carry the full question (with any options) in ``input`` and a
    short ``target`` answer (often ``(A)`` or a word). Scored as free-form so we
    do not have to re-parse 27 heterogeneous task formats into option lists."""
    items = []
    for row in _hf_rows("bbh", n):
        items.append({"question": str(row.get("input", "")),
                      "gold": str(row.get("target", ""))})
    return items


def _render_freeform(item: dict) -> str:
    return item["question"] + "\n\nGive only the final answer."


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _score_freeform(text: str, item: dict) -> bool:
    gold = item["gold"].strip()
    # BBH targets are frequently "(A)"-style; compare on the bracketed letter
    # if present, else on normalised text containment of the last line.
    m = re.search(r"\(([A-Za-z])\)", gold)
    if m:
        letter = m.group(1).upper()
        pred = _extract_choice(text, 26)
        return pred == letter
    last_line = next((ln for ln in reversed(text.strip().splitlines())
                      if ln.strip()), "")
    g = _norm_text(gold)
    return bool(g) and (g == _norm_text(last_line) or g in _norm_text(text))


def _score_mc(text: str, item: dict) -> bool:
    n = len(item["choices"])
    pred = _extract_choice(text, n)
    gold = item["gold"]
    # Gold may be a letter, an index, or the answer text.
    if isinstance(gold, str) and len(gold) == 1 and gold.upper() in LETTERS:
        gold_letter = gold.upper()
    elif isinstance(gold, int) and 0 <= gold < n:
        gold_letter = LETTERS[gold]
    elif item.get("letter_answer"):
        gold_letter = str(gold).strip().upper()[:1]
    else:
        # Match by answer text → its position.
        try:
            gold_letter = LETTERS[[c.strip() for c in item["choices"]].index(
                str(gold).strip())]
        except ValueError:
            gold_letter = None
    return pred is not None and pred == gold_letter


# --------------------------------------------------------------------------- #
# Adapter registry
# --------------------------------------------------------------------------- #
def _build_adapters() -> dict[str, BenchmarkAdapter]:
    return {
        "aime": BenchmarkAdapter("aime", "math", _load_math_like("aime"),
                                 _render_math, _score_math),
        "math": BenchmarkAdapter("math", "math", _load_math_like("math"),
                                 _render_math, _score_math),
        "gpqa": BenchmarkAdapter("gpqa", "mc", _load_gpqa_with_shuffle,
                                 _render_mc, _score_mc),
        "bbh": BenchmarkAdapter("bbh", "freeform", _load_bbh,
                                _render_freeform, _score_freeform),
        "truthfulqa": BenchmarkAdapter(
            "truthfulqa", "mc", _load_truthfulqa, _render_mc, _score_mc),
        "emobench": BenchmarkAdapter(
            "emobench", "mc",
            _load_mc("emobench", question_keys=("question", "scenario"),
                     choices_key="choices", answer_key="answer"),
            _render_mc, _score_mc),
    }


def _load_truthfulqa(n: int) -> list[dict]:
    """TruthfulQA multiple_choice: mc1_targets has choices + a 1-hot label."""
    items = []
    for row in _hf_rows("truthfulqa", n):
        mc1 = row.get("mc1_targets") or {}
        choices = [str(c) for c in mc1.get("choices", [])]
        labels = mc1.get("labels", [])
        gold = labels.index(1) if 1 in labels else 0
        items.append({"question": str(row.get("question", "")),
                      "choices": choices, "gold": gold})
    return items


# GPQA gold is the correct answer text placed first by _gpqa_choices, then the
# choices are shuffled deterministically per item so position is not a tell.
def _load_gpqa_with_shuffle(n: int) -> list[dict]:
    import random
    items = []
    for i, row in enumerate(_hf_rows("gpqa", n)):
        choices = _gpqa_choices(row)
        correct = choices[0] if choices else None
        rng = random.Random(i)
        rng.shuffle(choices)
        items.append({
            "question": str(row.get("Question") or row.get("question") or ""),
            "choices": choices,
            "gold": choices.index(correct) if correct in choices else 0,
        })
    return items


# Built after all loader helpers are defined (name resolution is at call time).
BENCHMARK_ADAPTERS = _build_adapters()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_benchmark(
    model: ChatModel,
    name: str,
    *,
    n_samples: int = 100,
    max_tokens: int = 2048,
) -> dict:
    """Evaluate one model on one benchmark; return accuracy + per-item records."""
    adapter = BENCHMARK_ADAPTERS[name]
    items = adapter.load(n_samples)
    correct = 0
    records = []
    for it in items:
        prompt = adapter.render(it)
        # Greedy decoding for capability eval (temperature 0) — we want the
        # model's best answer, not a temperature-1 sample. See DESIGN.md.
        res = model.generate([{"role": "user", "content": prompt}],
                             temperature=0.0, max_tokens=max_tokens)
        ok = bool(adapter.score(res.text, it))
        correct += ok
        records.append({"prompt": prompt, "answer": res.text, "correct": ok})
    n = len(items)
    return {"benchmark": name, "n": n,
            "accuracy": (correct / n if n else float("nan")),
            "records": records}


def run_capabilities(
    model_keys: Sequence[str],
    *,
    benchmarks: Sequence[str] = tuple(config.CAPABILITY_BENCHMARKS),
    n_samples: int = 100,
    out_path: str | Path | None = None,
) -> Path:
    """Evaluate each model on each benchmark; persist accuracies."""
    out_path = Path(out_path) if out_path else storage.results_path(
        "capabilities/results.jsonl")
    done = storage.completed_keys(out_path)
    for key in model_keys:
        model = build_model(key)
        for name in benchmarks:
            uid = f"{key}|{name}"
            if uid in done:
                continue
            result = evaluate_benchmark(model, name, n_samples=n_samples)
            result.update({"uid": uid, "model": key})
            # Store accuracy + n in the main file; dump per-item records aside.
            storage.write_jsonl(
                storage.results_path(f"capabilities/items_{key}_{name}.jsonl"),
                result.pop("records"))
            storage.append_jsonl(out_path, result)
    return out_path


def compare_models(
    baseline_key: str,
    finetuned_keys: Sequence[str],
    *,
    path: str | Path | None = None,
) -> dict:
    """Tabulate accuracy deltas (finetuned - baseline) per benchmark."""
    path = Path(path) if path else storage.results_path("capabilities/results.jsonl")
    acc: dict[str, dict[str, float]] = {}
    for r in storage.read_jsonl(path):
        acc.setdefault(r["model"], {})[r["benchmark"]] = r["accuracy"]
    out = {"baseline": baseline_key, "baseline_accuracy": acc.get(baseline_key, {}),
           "deltas": {}}
    for key in finetuned_keys:
        out["deltas"][key] = {
            b: acc.get(key, {}).get(b, float("nan")) - v
            for b, v in acc.get(baseline_key, {}).items()}
    return out
