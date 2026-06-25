"""Capability-preservation benchmarks (Section 4.2, Figure 7).

We evaluate that DPO/SFT do not degrade capabilities on:
  - AIME + MATH subsets (Hendrycks et al.) : free-form numeric/expression answers
  - GPQA (Rein et al.)                      : 4-way multiple choice
  - BBH (Suzgun et al.)                     : mixed; scored by normalised exact match
  - TruthfulQA (Lin et al.)                 : MC1 multiple choice
  - EmoBench (Sabour et al.)                : multiple choice emotion understanding

Each benchmark is described by a :class:`BenchmarkSpec` declaring how to load it
(HF dataset id/config/split), how to render a prompt, and how to score. The runner
is generic: generate at temperature 0, extract an answer, compare to the gold
label. Dataset schemas vary across HF releases, so loaders are written defensively
and field access is centralised in small adapter functions (see DESIGN.md note on
benchmark fragility).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from ..config import OUTPUTS_DIR
from ..models import GenConfig, get_client

# -- answer extraction / scoring helpers ---------------------------------------
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_LINE = re.compile(r"(?:final answer|answer)\s*[:=]\s*(.+)", re.IGNORECASE)
_CHOICE = re.compile(r"\b([A-D])\b")


def _normalise_num(s: str) -> str:
    s = s.strip().strip(".$ ")
    s = s.replace(",", "").replace(" ", "")
    s = re.sub(r"\\(?:text|mathrm|left|right)\b", "", s)
    return s


def extract_numeric(text: str) -> str | None:
    m = _BOXED.search(text)
    if m:
        return _normalise_num(m.group(1))
    m = _ANSWER_LINE.search(text)
    if m:
        return _normalise_num(m.group(1).splitlines()[0])
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return _normalise_num(nums[-1]) if nums else None


def extract_choice(text: str) -> str | None:
    m = _ANSWER_LINE.search(text)
    if m:
        c = _CHOICE.search(m.group(1))
        if c:
            return c.group(1)
    # fall back to last standalone letter in the whole response
    cands = _CHOICE.findall(text)
    return cands[-1] if cands else None


def score_numeric(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    return _normalise_num(pred) == _normalise_num(gold)


def score_choice(pred: str | None, gold: str) -> bool:
    return pred is not None and pred.upper() == gold.upper()


# -- benchmark specification ---------------------------------------------------
@dataclass
class BenchmarkSpec:
    name: str
    loader: Callable[[], list[dict]]   # -> list of {"prompt", "gold", "type"}
    answer_type: str                    # numeric | choice
    max_tokens: int = 1024
    n_limit: int | None = None          # cap examples for cheaper runs


def _mc_prompt(question: str, choices: list[str]) -> str:
    letters = ["A", "B", "C", "D", "E", "F"][: len(choices)]
    body = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    return (
        f"{question}\n\n{body}\n\n"
        "Reason briefly, then end with 'Answer: <letter>'."
    )


def _math_prompt(problem: str) -> str:
    return (
        f"Solve the following problem. Put your final answer in \\boxed{{}}.\n\n{problem}"
    )


# -- dataset loaders (defensive; degrade to [] if unavailable) -----------------
def _safe_load(fn) -> list[dict]:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        print(f"[capabilities] loader failed ({exc}); skipping benchmark.")
        return []


def load_math() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for ex in ds:
        out.append({"prompt": _math_prompt(ex["problem"]),
                    "gold": _normalise_num(str(ex["answer"])), "type": "numeric"})
    return out


def load_aime() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    return [{"prompt": _math_prompt(ex["Problem"]),
             "gold": _normalise_num(str(ex["Answer"])), "type": "numeric"} for ex in ds]


def load_gpqa() -> list[dict]:
    import random

    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = random.Random(0)  # deterministic answer placement
    letters = ["A", "B", "C", "D"]
    out = []
    for ex in ds:
        choices = [ex["Correct Answer"], ex["Incorrect Answer 1"],
                   ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[j] for j in order]
        gold_pos = order.index(0)  # where the correct answer (index 0) landed
        out.append({"prompt": _mc_prompt(ex["Question"], shuffled),
                    "gold": letters[gold_pos], "type": "choice"})
    return out


def load_bbh() -> list[dict]:
    from datasets import load_dataset

    # A representative multiple-choice BBH task.
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
    out = []
    for ex in ds:
        out.append({"prompt": f"{ex['input']}\n\nEnd with 'Answer: <text>'.",
                    "gold": _normalise_num(ex["target"].strip("()")), "type": "numeric"})
    return out


def load_truthfulqa() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    out = []
    for ex in ds:
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        gold_idx = labels.index(1)
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"][: len(choices)]
        out.append({"prompt": _mc_prompt(ex["question"], choices),
                    "gold": letters[gold_idx], "type": "choice"})
    return out


def load_emobench() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("EmoBench/EmoBench", split="test")
    out = []
    for ex in ds:
        choices = ex.get("choices") or ex.get("options")
        q = ex.get("question") or ex.get("scenario")
        gold = ex.get("answer") or ex.get("label")
        if not choices:
            continue
        letters = ["A", "B", "C", "D"][: len(choices)]
        gold_letter = gold if isinstance(gold, str) and gold in letters else letters[int(gold)]
        out.append({"prompt": _mc_prompt(q, choices), "gold": gold_letter, "type": "choice"})
    return out


DEFAULT_BENCHMARKS = [
    BenchmarkSpec("MATH", lambda: _safe_load(load_math), "numeric", n_limit=200),
    BenchmarkSpec("AIME", lambda: _safe_load(load_aime), "numeric"),
    BenchmarkSpec("GPQA", lambda: _safe_load(load_gpqa), "choice"),
    BenchmarkSpec("BBH", lambda: _safe_load(load_bbh), "numeric"),
    BenchmarkSpec("TruthfulQA", lambda: _safe_load(load_truthfulqa), "choice", n_limit=300),
    BenchmarkSpec("EmoBench", lambda: _safe_load(load_emobench), "choice"),
]


def run_benchmark(model: str, spec: BenchmarkSpec) -> dict:
    client = get_client(model)
    cfg = GenConfig(temperature=0.0, max_tokens=spec.max_tokens)
    items = spec.loader()
    if spec.n_limit:
        items = items[: spec.n_limit]
    correct = 0
    for ex in tqdm(items, desc=f"{spec.name}:{model}"):
        out = client.generate([{"role": "user", "content": ex["prompt"]}], cfg)
        if spec.answer_type == "numeric":
            ok = score_numeric(extract_numeric(out), ex["gold"])
        else:
            ok = score_choice(extract_choice(out), ex["gold"])
        correct += int(ok)
    n = len(items)
    return {"benchmark": spec.name, "model": model, "n": n,
            "accuracy": (correct / n) if n else float("nan")}


def run_capabilities(
    models: list[str], *, benchmarks: list[BenchmarkSpec] | None = None,
    out_dir: Path | None = None,
) -> Path:
    benchmarks = benchmarks or DEFAULT_BENCHMARKS
    out_dir = out_dir or (OUTPUTS_DIR / "capabilities")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [run_benchmark(m, b) for m in models for b in benchmarks]
    path = out_dir / "results.json"
    path.write_text(json.dumps(results, indent=2))
    return path
