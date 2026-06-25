"""Capability benchmarks to confirm finetuning does not degrade the model
(Section 4.2): AIME, MATH subset, GPQA, BBH, TruthfulQA, and EmoBench.

Each benchmark is reduced to a (question -> answer) accuracy task with a
benchmark-appropriate answer extractor. The harness is deliberately small and
uniform; the goal is a *relative* comparison (vanilla vs DPO vs SFT), matching
the paper's claim of "no reductions in scores", not to reproduce absolute
leaderboard numbers.

All datasets load from HuggingFace; subset sizes are capped via ``limit`` so the
suite is tractable. Greedy decoding (temperature 0) is used for grading.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from config import RESULTS_DIR
from src.models import load_model
from src.models.base import Message


@dataclass
class BenchSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    limit: int
    kind: str   # "numeric" | "mcq" | "freeform_mcq"


BENCHMARKS = [
    BenchSpec("AIME", "Maxwell-Jia/AIME_2024", None, "train", 30, "numeric"),
    BenchSpec("MATH", "HuggingFaceH4/MATH-500", None, "test", 200, "numeric"),
    BenchSpec("GPQA", "Idavidrein/gpqa", "gpqa_diamond", "train", 198, "mcq"),
    BenchSpec("BBH", "lukaemon/bbh", "boolean_expressions", "test", 250, "freeform_mcq"),
    BenchSpec("TruthfulQA", "truthful_qa", "multiple_choice", "validation", 200, "mcq"),
    BenchSpec("EmoBench", "Sahandfer/EmoBench", "EA", "test", 200, "mcq"),
]

ANSWER_INSTRUCTION = (
    "\n\nThink step by step, then end your reply with a line of the form "
    "'ANSWER: <answer>'."
)


def _extract_answer(text: str) -> str:
    m = re.findall(r"ANSWER:\s*(.+)", text)
    if m:
        return m[-1].strip().strip(".").strip()
    # boxed fallback for math
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _grade(pred: str, gold: str, kind: str) -> bool:
    p, g = _normalise(pred), _normalise(gold)
    if kind == "numeric":
        # Compare the trailing number if present.
        pn = re.findall(r"-?\d+\.?\d*", pred.replace(",", ""))
        gn = re.findall(r"-?\d+\.?\d*", str(gold).replace(",", ""))
        if pn and gn:
            try:
                return abs(float(pn[-1]) - float(gn[-1])) < 1e-3
            except ValueError:
                pass
        return p == g
    # MCQ: gold may be a letter or the full option; accept either containment.
    return g == p or g in p or p in g


def _load_items(spec: BenchSpec) -> list[dict]:
    """Return a uniform list of {question, answer} dicts. Wrapped in try/except
    so a single dataset's schema drift doesn't kill the whole suite."""
    from datasets import load_dataset

    ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split)
    ds = ds.select(range(min(spec.limit, len(ds))))
    items = []
    for row in ds:
        q, a = _row_to_qa(spec, row)
        if q and a is not None:
            items.append({"question": q, "answer": a})
    return items


def _row_to_qa(spec: BenchSpec, row: dict):
    """Best-effort field mapping per benchmark (schemas vary; see DESIGN.md)."""
    if spec.name == "AIME":
        return row.get("Problem") or row.get("question"), row.get("Answer") or row.get("answer")
    if spec.name == "MATH":
        return row.get("problem"), row.get("answer")
    if spec.name == "GPQA":
        q = row.get("Question")
        correct = row.get("Correct Answer")
        opts = [row.get(f"Incorrect Answer {i}") for i in (1, 2, 3)] + [correct]
        opts = [o for o in opts if o]
        letters = "ABCD"
        body = q + "\n" + "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
        # answer letter is position of correct in opts
        return body, letters[opts.index(correct)]
    if spec.name == "BBH":
        return row.get("input"), row.get("target")
    if spec.name == "TruthfulQA":
        q = row.get("question")
        mc = row.get("mc1_targets", {})
        choices = mc.get("choices", [])
        labels = mc.get("labels", [])
        letters = "ABCDEFGH"
        body = q + "\n" + "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
        gold = letters[labels.index(1)] if 1 in labels else None
        return body, gold
    if spec.name == "EmoBench":
        q = row.get("Scenario", "") + "\n" + row.get("Question", "")
        choices = row.get("Choices") or row.get("choices") or []
        letters = "ABCD"
        body = q + "\n" + "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
        gold = row.get("Label") or row.get("Answer")
        return body, gold
    return None, None


def run_benchmark(model, spec: BenchSpec) -> dict:
    try:
        items = _load_items(spec)
    except Exception as e:  # noqa: BLE001
        return {"benchmark": spec.name, "error": str(e), "accuracy": None, "n": 0}
    correct = 0
    for it in tqdm(items, desc=spec.name):
        prompt = it["question"] + ANSWER_INSTRUCTION
        out = model.generate(
            [Message("user", prompt)], temperature=0.0, max_new_tokens=2048
        )
        if _grade(_extract_answer(out), str(it["answer"]), spec.kind):
            correct += 1
    n = len(items)
    return {"benchmark": spec.name, "accuracy": correct / n if n else None, "n": n}


def run_all(target_spec, *, adapter_path=None, out_path: Path | None = None,
            hf_kwargs: dict | None = None) -> Path:
    label = target_spec.name + ("+adapter" if adapter_path else "")
    out_path = out_path or (RESULTS_DIR / f"capabilities_{label}.json")
    model = (
        load_model(target_spec, adapter_path=adapter_path, **(hf_kwargs or {}))
        if target_spec.backend == "hf" else load_model(target_spec)
    )
    results = [run_benchmark(model, spec) for spec in BENCHMARKS]
    model.close()
    out_path.write_text(json.dumps({"model": label, "results": results}, indent=2))
    return out_path
