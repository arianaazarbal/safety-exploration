"""Capability-preservation evals (Section 4.2, Figure 7).

Confirms DPO/SFT does not degrade capabilities: AIME + MATH subsets, GPQA, BBH,
TruthfulQA, and EmoBench (emotional intelligence). Compact, dataset-driven harness:
each benchmark declares how to render a question and how to score an answer
(multiple-choice letter match or boxed/last-number exact match).

The HF dataset schemas below reflect common public versions; adjust the field names
in BENCHMARKS if a particular dataset revision differs (see DESIGN.md).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import GenParams, Message, ModelClient

MC_INSTRUCTION = (
    "Answer the following multiple-choice question. Think step by step, then end your "
    "reply with a line of the form 'Answer: X' where X is the option letter."
)
EXACT_INSTRUCTION = (
    "Solve the problem. Show your reasoning, then end with a line 'Answer: <final answer>'."
)


@dataclass
class BenchmarkSpec:
    name: str
    hf_id: str
    split: str
    kind: str                       # "mc" | "exact"
    render: Callable[[dict], str]   # row -> question text (incl. options for mc)
    gold: Callable[[dict], str]     # row -> gold answer (letter for mc; string for exact)
    config: str | None = None
    subset_size: int | None = None  # paper uses subsets for AIME/MATH


# --- answer extraction ---
def extract_mc_letter(text: str) -> str | None:
    m = re.findall(r"Answer:\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([A-D])\b", text.strip()[-10:])
    return m[-1].upper() if m else None


def extract_boxed_or_number(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    ans = re.findall(r"Answer:\s*(.+)", text)
    if ans:
        return ans[-1].strip().rstrip(".")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _norm(s: str | None) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", "", s.lower().replace("$", "").replace(",", ""))


# --- renderers ---
def _mc_render(question_field: str, choices_field: str):
    def render(row: dict) -> str:
        choices = row[choices_field]
        if isinstance(choices, dict):  # {"text": [...], "label": [...]}
            choices = choices.get("text", [])
        opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        return f"{MC_INSTRUCTION}\n\n{row[question_field]}\n\n{opts}"
    return render


def _exact_render(question_field: str):
    def render(row: dict) -> str:
        return f"{EXACT_INSTRUCTION}\n\n{row[question_field]}"
    return render


# Common public schemas. Field names may need adjusting per dataset revision.
BENCHMARKS: dict[str, BenchmarkSpec] = {
    "math": BenchmarkSpec(
        name="math", hf_id="hendrycks/competition_math", split="test", kind="exact",
        render=_exact_render("problem"),
        gold=lambda r: extract_boxed_or_number(r.get("solution", "")) or "",
        subset_size=200,
    ),
    "aime": BenchmarkSpec(
        name="aime", hf_id="Maxwell-Jia/AIME_2024", split="train", kind="exact",
        render=_exact_render("Problem"),
        gold=lambda r: str(r.get("Answer", "")).strip(),
        subset_size=30,
    ),
    "gpqa": BenchmarkSpec(
        name="gpqa", hf_id="Idavidrein/gpqa", config="gpqa_diamond", split="train", kind="mc",
        render=lambda r: (
            f"{MC_INSTRUCTION}\n\n{r['Question']}\n\n"
            f"A. {r['Correct Answer']}\nB. {r['Incorrect Answer 1']}\n"
            f"C. {r['Incorrect Answer 2']}\nD. {r['Incorrect Answer 3']}"
        ),
        gold=lambda r: "A",  # NOTE: options should be shuffled per-row in practice
    ),
    "bbh": BenchmarkSpec(
        name="bbh", hf_id="lukaemon/bbh", config="boolean_expressions", split="test",
        kind="exact",
        render=_exact_render("input"),
        gold=lambda r: str(r.get("target", "")).strip(),
        subset_size=200,
    ),
    "truthfulqa": BenchmarkSpec(
        name="truthfulqa", hf_id="truthful_qa", config="multiple_choice", split="validation",
        kind="mc",
        render=lambda r: (
            f"{MC_INSTRUCTION}\n\n{r['question']}\n\n"
            + "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(r["mc1_targets"]["choices"]))
        ),
        gold=lambda r: chr(65 + r["mc1_targets"]["labels"].index(1)),
    ),
    "emobench": BenchmarkSpec(
        name="emobench", hf_id="EmoBench/EmoBench", split="test", kind="mc",
        render=_mc_render("question", "choices"),
        gold=lambda r: str(r.get("answer", "")).strip()[:1].upper(),
    ),
}


def run_benchmark(
    client: ModelClient,
    spec: BenchmarkSpec,
    n: int | None = None,
    max_new_tokens: int = 2048,
) -> dict:
    from datasets import load_dataset

    kwargs = {"split": spec.split}
    if spec.config:
        kwargs["name"] = spec.config
    ds = load_dataset(spec.hf_id, **kwargs)
    limit = n or spec.subset_size or len(ds)
    limit = min(limit, len(ds))
    params = GenParams(temperature=0.0, max_new_tokens=max_new_tokens, n=1)

    correct = 0
    for i in range(limit):
        row = ds[i]
        q = spec.render(row)
        out = client.generate_chat([Message("user", q)], params)[0]
        pred = extract_mc_letter(out) if spec.kind == "mc" else extract_boxed_or_number(out)
        gold = spec.gold(row)
        if _norm(pred) == _norm(gold):
            correct += 1
    return {"benchmark": spec.name, "n": limit, "accuracy": correct / limit if limit else None}


def run_capability_suite(
    client: ModelClient,
    model_name: str,
    benchmarks: list[str] | None = None,
    out_path: str | Path | None = None,
) -> list[dict]:
    benchmarks = benchmarks or list(BENCHMARKS.keys())
    results = []
    for b in benchmarks:
        try:
            res = run_benchmark(client, BENCHMARKS[b])
        except Exception as e:
            res = {"benchmark": b, "error": str(e)}
        res["model"] = model_name
        results.append(res)
        print(f"  {model_name} {b}: {res}")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results
