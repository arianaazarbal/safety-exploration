"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Goal: confirm DPO/SFT do not degrade capabilities. We evaluate the vanilla and
finetuned Gemma variants on standard benchmarks and compare accuracy.

Implemented as a small generic harness with per-benchmark adapters that define
(prompt, gold_answer, answer_extractor). The set mirrors the paper:
  * AIME / MATH : integer / \\boxed{} answer matching
  * GPQA / BBH  : multiple-choice letter matching
  * TruthfulQA  : multiple-choice (MC1) accuracy
  * EmoBench    : emotion-understanding multiple-choice accuracy

These are accuracy proxies; exact paper splits/few-shot setups are not fully
specified, so we use zero-shot with deterministic decoding and document the
choice (see DESIGN.md). The comparative claim ("no reduction") is robust to the
exact harness as long as vanilla and finetuned models are scored identically.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from tqdm import tqdm

from ..config import Config, load_config
from ..models import ChatMessage, GenerationConfig, get_client

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_LETTER_RE = re.compile(r"\b([A-D])\b")
_FINAL_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


@dataclass
class Example:
    prompt: str
    gold: str
    kind: str           # "boxed" | "mcq" | "integer"
    choices: List[str] | None = None


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #
def _extract(kind: str, text: str) -> str | None:
    if kind == "boxed":
        m = _BOXED_RE.findall(text)
        if m:
            return m[-1].strip()
        nums = _FINAL_NUM_RE.findall(text)
        return nums[-1] if nums else None
    if kind == "integer":
        nums = _FINAL_NUM_RE.findall(text)
        return nums[-1] if nums else None
    if kind == "mcq":
        # Prefer an explicit "Answer: X"; else last standalone A-D.
        m = re.findall(r"[Aa]nswer\s*[:\-]?\s*([A-D])", text)
        if m:
            return m[-1].upper()
        letters = _LETTER_RE.findall(text)
        return letters[-1].upper() if letters else None
    return None


def _normalise_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


# --------------------------------------------------------------------------- #
# Dataset adapters -> list[Example]
# --------------------------------------------------------------------------- #
def _mcq_prompt(question: str, choices: List[str]) -> str:
    letters = "ABCD"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{body}\n\nThink step by step, then end with "
            f"'Answer: <letter>'.")


def _load_examples(name: str, spec: dict, *, limit: int | None) -> List[Example]:
    from datasets import load_dataset

    ex: List[Example] = []
    if name in ("aime", "math"):
        ds = load_dataset(spec["dataset"], split="test") if name == "math" \
            else load_dataset(spec["dataset"], split="train")
        n = spec.get("subset_n", limit or len(ds))
        for row in ds.select(range(min(n, len(ds)))):
            q = row.get("problem") or row.get("question")
            ans = str(row.get("answer") or row.get("solution"))
            gold = _BOXED_RE.findall(ans)
            ex.append(Example(
                prompt=f"{q}\n\nProvide your final answer in \\boxed{{}}.",
                gold=_normalise_num(gold[-1] if gold else ans),
                kind="boxed",
            ))
    elif name == "gpqa":
        ds = load_dataset(spec["dataset"], spec.get("config"), split="train")
        n = spec.get("subset_n", limit or len(ds))
        for row in ds.select(range(min(n, len(ds)))):
            choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                       row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            ex.append(Example(_mcq_prompt(row["Question"], choices),
                              gold="A", kind="mcq", choices=choices))
    elif name == "bbh":
        ds = load_dataset(spec["dataset"], split="test")
        n = spec.get("subset_n", limit or len(ds))
        for row in ds.select(range(min(n, len(ds)))):
            ex.append(Example(f"{row['input']}\n\nEnd with 'Answer: <answer>'.",
                              gold=str(row["target"]).strip("()"), kind="mcq"))
    elif name == "truthfulqa":
        ds = load_dataset(spec["dataset"], spec.get("config"), split="validation")
        n = limit or len(ds)
        for row in ds.select(range(min(n, len(ds)))):
            mc1 = row["mc1_targets"]
            choices = mc1["labels"]
            opts = mc1["choices"]
            correct_idx = choices.index(1)
            ex.append(Example(
                _mcq_prompt(row["question"], opts),
                gold="ABCD"[correct_idx] if correct_idx < 4 else "A",
                kind="mcq", choices=opts,
            ))
    elif name == "emobench":
        ds = load_dataset(spec["dataset"], split="test")
        n = limit or len(ds)
        for row in ds.select(range(min(n, len(ds)))):
            q = row.get("question") or row.get("scenario") or ""
            opts = row.get("choices") or row.get("options") or []
            gold = row.get("answer") or row.get("label")
            ex.append(Example(_mcq_prompt(q, list(opts)),
                              gold=str(gold), kind="mcq", choices=list(opts)))
    else:
        raise ValueError(f"Unknown benchmark '{name}'")
    return ex


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_benchmark(
    model_name: str, benchmark: str, *, limit: int | None = None,
    cfg: Config | None = None,
) -> dict:
    cfg = cfg or load_config()
    spec = cfg.eval["capabilities"]["benchmarks"][benchmark]
    client = get_client(model_name)
    gen = GenerationConfig(temperature=0.0, max_new_tokens=2048, thinking=False)

    examples = _load_examples(benchmark, spec, limit=limit)
    correct = 0
    for exmpl in tqdm(examples, desc=f"{model_name}/{benchmark}"):
        out = client.generate([ChatMessage("user", exmpl.prompt)], gen)
        pred = _extract(exmpl.kind, out)
        if exmpl.kind in ("boxed", "integer"):
            pred = _normalise_num(pred)
        if pred is not None and str(pred).strip().lower() == str(exmpl.gold).strip().lower():
            correct += 1
    n = len(examples)
    return {"benchmark": benchmark, "model": model_name, "n": n,
            "accuracy": correct / n if n else float("nan")}


def run_capability_suite(
    model_name: str, benchmarks: List[str] | None = None,
    *, cfg: Config | None = None,
) -> Path:
    import json
    cfg = cfg or load_config()
    benchmarks = benchmarks or list(cfg.eval["capabilities"]["benchmarks"].keys())
    results = {b: evaluate_benchmark(model_name, b, cfg=cfg) for b in benchmarks}
    out_dir = cfg.path("outputs_dir") / "capabilities"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}.json"
    path.write_text(json.dumps(results, indent=2))
    return path
