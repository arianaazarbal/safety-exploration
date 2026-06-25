"""Capability-preservation benchmark runner (Section 4.2 / Figure 7).

Each benchmark is described by a ``BenchmarkSpec``: how to load examples from a
HuggingFace dataset, how to format the prompt, and how to grade. We evaluate the
vanilla and finetuned Gemma variants and check that accuracy does not drop
(the paper reports "no reductions in scores").

Dataset ids and configs are defaults that match the commonly-used public
releases; they are overridable. EmoBench is included to check the emotion-
capability claim ("DPO does not degrade EmoBench performance").
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from ..config import RESULTS_DIR, get_participant
from ..models import build_client
from ..models.base import ChatClient
from ..utils import write_jsonl
from . import graders


@dataclass
class BenchmarkSpec:
    name: str
    hf_id: str
    config: str | None
    split: str
    format_fn: Callable[[dict[str, Any]], str]
    grade_fn: Callable[[str, dict[str, Any]], bool]
    subset: int | None = 200  # cap examples ("AIME and MATH subsets")


# --------------------------- formatting helpers --------------------------- #
def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{opts}\n\n"
        "Answer with the single letter (A, B, C, or D) on the final line as "
        "'Answer: <letter>'."
    )


# --------------------------- per-benchmark adapters ----------------------- #
def _fmt_aime(ex: dict[str, Any]) -> str:
    q = ex.get("problem") or ex.get("question") or ex.get("Problem")
    return f"Solve the problem. Give the final integer answer as 'Answer: <n>'.\n\n{q}"


def _grade_aime(out: str, ex: dict[str, Any]) -> bool:
    gold = ex.get("answer") or ex.get("Answer") or ex.get("solution")
    return graders.grade_numeric(out, gold)


def _fmt_math(ex: dict[str, Any]) -> str:
    return f"Solve the problem and put the final answer in \\boxed{{}}.\n\n{ex['problem']}"


def _grade_math(out: str, ex: dict[str, Any]) -> bool:
    return graders.grade_boxed(out, ex.get("solution", ""))


def _fmt_gpqa(ex: dict[str, Any]) -> str:
    # GPQA stores correct + 3 incorrect answers; we present them in fixed order.
    choices = [
        ex["Correct Answer"],
        ex["Incorrect Answer 1"],
        ex["Incorrect Answer 2"],
        ex["Incorrect Answer 3"],
    ]
    return _mc_prompt(ex["Question"], choices)


def _grade_gpqa(out: str, ex: dict[str, Any]) -> bool:
    return graders.grade_mc(out, "A")  # correct answer is option A by construction


def _fmt_bbh(ex: dict[str, Any]) -> str:
    return f"{ex['input']}\n\nGive only the final answer on the last line as 'Answer: <answer>'."


def _grade_bbh(out: str, ex: dict[str, Any]) -> bool:
    target = str(ex.get("target", "")).strip().strip("()")
    low = out.lower()
    return target.lower() in low.splitlines()[-1] if low.strip() else False


def _fmt_truthfulqa(ex: dict[str, Any]) -> str:
    choices = ex["mc1_targets"]["choices"]
    return _mc_prompt(ex["question"], choices)


def _grade_truthfulqa(out: str, ex: dict[str, Any]) -> bool:
    # In TruthfulQA mc1, the first listed choice is the correct one.
    labels = ex["mc1_targets"]["labels"]
    correct_idx = labels.index(1) if 1 in labels else 0
    return graders.grade_mc(out, chr(65 + correct_idx))


def _fmt_emobench(ex: dict[str, Any]) -> str:
    q = ex.get("question") or ex.get("scenario") or ex.get("Scenario", "")
    choices = ex.get("choices") or ex.get("options") or []
    return _mc_prompt(q, list(choices))


def _grade_emobench(out: str, ex: dict[str, Any]) -> bool:
    ans = ex.get("answer") or ex.get("label")
    if isinstance(ans, int):
        return graders.grade_mc(out, chr(65 + ans))
    return graders.grade_mc(out, str(ans))


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "aime": BenchmarkSpec("aime", "Maxwell-Jia/AIME_2024", None, "train", _fmt_aime, _grade_aime, None),
    "math": BenchmarkSpec("math", "hendrycks/competition_math", None, "test", _fmt_math, _grade_math, 200),
    "gpqa": BenchmarkSpec("gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train", _fmt_gpqa, _grade_gpqa, None),
    "bbh": BenchmarkSpec("bbh", "lukaemon/bbh", "boolean_expressions", "test", _fmt_bbh, _grade_bbh, 200),
    "truthfulqa": BenchmarkSpec("truthfulqa", "truthful_qa", "multiple_choice", "validation", _fmt_truthfulqa, _grade_truthfulqa, 200),
    "emobench": BenchmarkSpec("emobench", "Sahandfer/EmoBench", None, "test", _fmt_emobench, _grade_emobench, 200),
}


def _load_examples(spec: BenchmarkSpec) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(spec.hf_id, spec.config, split=spec.split)
    rows = list(ds)
    if spec.subset:
        rows = rows[: spec.subset]
    return rows


def run_benchmark(
    spec: BenchmarkSpec,
    client: ChatClient,
    *,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    try:
        examples = _load_examples(spec)
    except Exception as exc:  # dataset unavailable/gated
        return {"benchmark": spec.name, "error": str(exc), "accuracy": None, "n": 0}

    correct = 0
    details = []
    for ex in tqdm(examples, desc=f"{client.name}:{spec.name}"):
        prompt = spec.format_fn(ex)
        out = client.generate(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        ok = bool(spec.grade_fn(out, ex))
        correct += int(ok)
        details.append({"prompt": prompt, "output": out, "correct": ok})
    return {
        "benchmark": spec.name,
        "accuracy": correct / len(examples) if examples else None,
        "n": len(examples),
        "details": details,
    }


def run_all(
    model_name: str,
    *,
    adapter_path: str | None = None,
    benchmarks: list[str] | None = None,
    output_subdir: str | None = None,
) -> Path:
    spec = get_participant(model_name)
    client_kwargs = {"adapter_path": adapter_path} if adapter_path else {}
    client = build_client(spec, **client_kwargs)
    names = benchmarks or list(BENCHMARKS.keys())

    out_dir = RESULTS_DIR / "capabilities"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = output_subdir or client.name.replace("/", "__")

    summary = []
    for n in names:
        res = run_benchmark(BENCHMARKS[n], client)
        write_jsonl(out_dir / f"{label}__{n}.jsonl", res.get("details", []) or [])
        summary.append({k: v for k, v in res.items() if k != "details"})

    write_jsonl(out_dir / f"{label}__summary.jsonl", summary)
    client.close()
    return out_dir
