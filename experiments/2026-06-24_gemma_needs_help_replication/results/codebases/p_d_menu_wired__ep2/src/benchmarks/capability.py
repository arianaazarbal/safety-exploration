"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Verifies the DPO finetune does not degrade capabilities by comparing the
vanilla instruct model against the DPO model on:
  * AIME + MATH subsets (Hendrycks et al., 2021) - numeric answers
  * GPQA (Rein et al., 2023)                      - multiple choice
  * BBH (Suzgun et al., 2022)                      - mixed
  * TruthfulQA (Lin et al., 2022)                  - multiple choice
  * EmoBench (Sabour et al., 2024)                 - emotion-capability MC

Each benchmark is run at temperature 0 (capability, not propensity) with a
final-answer extractor. Dataset ids / splits are best-effort; if a dataset is
unavailable the runner reports it as skipped rather than crashing, so the
pipeline stays runnable offline.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from config import PATHS
from src.models import load_subject
from src.models.base import Conversation


@dataclass
class BenchmarkSpec:
    name: str
    hf_id: str
    split: str
    answer_type: str            # "numeric" | "mc"
    question_key: str
    answer_key: str
    choices_key: str | None = None
    config: str | None = None


BENCHMARKS = {
    "math": BenchmarkSpec("math", "HuggingFaceH4/MATH-500", "test", "numeric",
                          "problem", "answer"),
    "aime": BenchmarkSpec("aime", "HuggingFaceH4/aime_2024", "train", "numeric",
                          "problem", "answer"),
    "gpqa": BenchmarkSpec("gpqa", "Idavidrein/gpqa", "train", "mc",
                          "Question", "Correct Answer", config="gpqa_diamond"),
    "bbh": BenchmarkSpec("bbh", "lukaemon/bbh", "test", "mc", "input", "target",
                         config="logical_deduction_three_objects"),
    "truthfulqa": BenchmarkSpec("truthfulqa", "truthful_qa", "validation", "mc",
                                "question", "mc1_targets", config="multiple_choice"),
    "emobench": BenchmarkSpec("emobench", "Sahandfer/EmoBench", "test", "mc",
                              "question", "answer", choices_key="choices"),
}

_INSTR = {
    "numeric": "Solve the problem. End your reply with 'ANSWER: <final answer>'.",
    "mc": "Answer the multiple-choice question. End your reply with 'ANSWER: <letter>'.",
}


def _extract_answer(text: str) -> str:
    m = re.search(r"ANSWER:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    # Fallback: last non-empty line.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def run_benchmark(
    subject_key: str,
    benchmark: str,
    *,
    adapter_path: str | None = None,
    n: int = 100,
    out_path: str | None = None,
) -> dict:
    spec = BENCHMARKS[benchmark]
    try:
        from datasets import load_dataset

        ds = (
            load_dataset(spec.hf_id, spec.config, split=spec.split)
            if spec.config
            else load_dataset(spec.hf_id, split=spec.split)
        )
    except Exception as e:
        return {"benchmark": benchmark, "skipped": True, "reason": str(e)}

    client = load_subject(subject_key, adapter_path=adapter_path)

    correct, total = 0, 0
    for row in ds.select(range(min(n, len(ds)))):
        q = row[spec.question_key]
        prompt = f"{_INSTR[spec.answer_type]}\n\n{q}"
        if spec.choices_key and row.get(spec.choices_key):
            opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(row[spec.choices_key]))
            prompt += f"\n\nOptions:\n{opts}"
        resp = client.generate(
            Conversation().user(prompt), max_tokens=1024, temperature=0.0
        )
        pred = _extract_answer(resp.text)
        gold = row[spec.answer_key]
        if isinstance(gold, dict):  # truthfulqa mc1_targets style
            labels = gold.get("labels", [])
            choices = gold.get("choices", [])
            gold = next((c for c, l in zip(choices, labels) if l == 1), "")
        np_pred, np_gold = _norm(pred), _norm(str(gold))
        # Count correct when the (non-empty) normalized prediction and gold
        # answer contain one another — tolerant of formatting / extra wording.
        if np_pred and np_gold and (np_pred in np_gold or np_gold in np_pred):
            correct += 1
        total += 1

    acc = correct / total if total else float("nan")
    result = {"benchmark": benchmark, "subject": subject_key,
              "adapter": adapter_path, "n": total, "accuracy": acc}
    out_path = out_path or os.path.join(
        PATHS.results, f"bench_{benchmark}_{subject_key}{'_dpo' if adapter_path else ''}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result
