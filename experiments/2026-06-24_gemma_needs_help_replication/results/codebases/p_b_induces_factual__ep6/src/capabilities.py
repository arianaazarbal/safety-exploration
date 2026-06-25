"""Section 4.2 / Figure 7: capability-preservation benchmarks.

Confirms DPO/SFT do not degrade capabilities: AIME + MATH (subsets), GPQA, BBH,
TruthfulQA, and EmoBench (emotion capability). This is a lightweight,
self-contained accuracy harness over HuggingFace datasets -- the paper does not
publish exact prompt formats, so we use simple zero-shot prompts with regex answer
extraction and document this gap in DESIGN.md. The point of the replication is the
*relative* comparison (vanilla vs DPO vs SFT), which is robust to harness details.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import config
from .models import get_model


@dataclass
class BenchmarkSpec:
    key: str
    hf_dataset: str
    hf_config: str | None
    split: str
    question_field: str
    answer_field: str
    kind: str           # "exact_letter" | "exact_number" | "mcq" | "truthfulqa_mc"
    instruction: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "aime": BenchmarkSpec(
        "aime", "Maxwell-Jia/AIME_2024", None, "train", "Problem", "Answer",
        "exact_number",
        "Solve the problem. End your response with 'Final Answer: <integer>'.",
    ),
    "math": BenchmarkSpec(
        "math", "HuggingFaceH4/MATH-500", None, "test", "problem", "answer",
        "exact_number",
        "Solve the problem. End your response with 'Final Answer: <answer>'.",
    ),
    "gpqa": BenchmarkSpec(
        "gpqa", "Idavidrein/gpqa", "gpqa_main", "train", "Question", "Correct Answer",
        "mcq",
        "Answer the multiple-choice question. End with 'Final Answer: <letter>'.",
    ),
    "bbh": BenchmarkSpec(
        "bbh", "lukaemon/bbh", "boolean_expressions", "test", "input", "target",
        "exact_letter",
        "Answer the question. End with 'Final Answer: <answer>'.",
    ),
    "truthfulqa": BenchmarkSpec(
        "truthfulqa", "truthful_qa", "multiple_choice", "validation",
        "question", "mc1_targets", "truthfulqa_mc",
        "Choose the single best answer. End with 'Final Answer: <letter>'.",
    ),
    "emobench": BenchmarkSpec(
        "emobench", "Sahandfer/EmoBench", None, "test", "question", "answer", "mcq",
        "Answer the multiple-choice question. End with 'Final Answer: <letter>'.",
    ),
}

_FINAL_ANSWER_RE = re.compile(r"final answer\s*[:\-]?\s*(.+?)\s*$",
                              re.IGNORECASE | re.DOTALL)
_LETTER_RE = re.compile(r"\b([A-E])\b")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_final(text: str) -> str:
    m = _FINAL_ANSWER_RE.search(text.strip())
    return (m.group(1).strip() if m else text.strip().splitlines()[-1] if text.strip() else "")


def _grade(kind: str, model_text: str, gold) -> bool:
    pred = _extract_final(model_text)
    if kind == "exact_number":
        pm = _NUMBER_RE.findall(pred)
        gm = _NUMBER_RE.findall(str(gold))
        return bool(pm and gm and pm[-1] == gm[-1])
    if kind in ("exact_letter", "mcq", "truthfulqa_mc"):
        lm = _LETTER_RE.findall(pred.upper())
        return bool(lm and lm[-1] == str(gold).upper())
    return pred.strip().lower() == str(gold).strip().lower()


def _format_question(spec: BenchmarkSpec, row) -> tuple[str, str]:
    """Return (prompt_text, gold_answer) handling MCQ option shuffling."""
    q = str(row[spec.question_field])
    if spec.kind == "mcq" and "choices" in row:
        opts = row["choices"]
        letters = "ABCDE"
        lines = [f"{letters[i]}. {o}" for i, o in enumerate(opts)]
        gold = letters[int(row.get("answer", 0))] if isinstance(row.get("answer"), int) else row[spec.answer_field]
        return q + "\n" + "\n".join(lines), str(gold)
    if spec.kind == "truthfulqa_mc":
        targets = row[spec.answer_field]
        choices = targets["choices"]
        labels = targets["labels"]
        letters = "ABCDE"
        lines = [f"{letters[i]}. {c}" for i, c in enumerate(choices)]
        gold = letters[labels.index(1)]
        return q + "\n" + "\n".join(lines), gold
    return q, str(row[spec.answer_field])


def run_benchmark(model_key: str, bench_key: str, *, limit: int = 100,
                  seed: int = 0) -> dict:
    spec = BENCHMARKS[bench_key]
    model = get_model(model_key)
    from datasets import load_dataset

    ds_kwargs = {} if spec.hf_config is None else {"name": spec.hf_config}
    ds = load_dataset(spec.hf_dataset, split=spec.split, **ds_kwargs)
    n = min(limit, len(ds))

    correct = 0
    for i in range(n):
        row = ds[i]
        prompt_text, gold = _format_question(spec, row)
        messages = [{"role": "user", "content": f"{spec.instruction}\n\n{prompt_text}"}]
        out = model.generate(messages, temperature=0.0, max_new_tokens=2048)
        if _grade(spec.kind, out, gold):
            correct += 1
    return {"model_key": model_key, "benchmark": bench_key,
            "n": n, "correct": correct, "accuracy": correct / n if n else 0.0}


def run_capabilities(
    *,
    model_keys: list[str] | None = None,
    benchmarks: tuple[str, ...] = config.CAPABILITY_BENCHMARKS,
    limit: int = 100,
    out_path: Path | None = None,
) -> Path:
    model_keys = model_keys or config.CAPABILITY_MODELS
    out_path = out_path or (config.RESULTS_DIR / "capabilities.jsonl")
    with out_path.open("w") as fh:
        for mkey in model_keys:
            for bkey in benchmarks:
                try:
                    res = run_benchmark(mkey, bkey, limit=limit)
                except Exception as exc:  # noqa: BLE001
                    res = {"model_key": mkey, "benchmark": bkey, "error": repr(exc)}
                fh.write(json.dumps(res) + "\n")
                print(f"[capabilities] {mkey} / {bkey}: "
                      f"{res.get('accuracy', res.get('error'))}")
    return out_path
