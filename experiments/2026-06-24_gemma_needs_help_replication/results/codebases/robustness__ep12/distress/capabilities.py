"""Capability-preservation evaluation (Section 4.2, Figure 7).

Confirms the DPO/SFT finetune does not degrade capabilities. The paper uses:
  * AIME + MATH subsets (Hendrycks et al. 2021) -- exact-match numeric answers
  * GPQA (Rein et al. 2023)                      -- multiple choice
  * BBH (Suzgun et al. 2022)                     -- mixed (we use MC subsets)
  * TruthfulQA (Lin et al. 2022)                 -- multiple choice (MC1)
  * EmoBench (Sabour et al. 2024)                -- emotion-understanding MC

This module is a thin, dependency-light harness: load a benchmark from HF,
prompt the target, extract an answer, score. For rigorous numbers, swapping in
EleutherAI's lm-evaluation-harness is recommended (see DESIGN.md); this module
gives a self-contained, runnable approximation for the vanilla-vs-finetune
comparison that matters here (relative, not absolute, scores).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .models import ChatClient


@dataclass
class BenchmarkSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    kind: str                 # "mc" or "exact_numeric"
    question_key: str
    answer_key: str
    choices_key: str | None = None


# Registry. Configs/keys may need tweaking per dataset version; documented as a
# gap-filling area in DESIGN.md.
BENCHMARKS = {
    "math": BenchmarkSpec("math", "HuggingFaceH4/MATH-500", None, "test",
                          "exact_numeric", "problem", "answer"),
    "aime": BenchmarkSpec("aime", "HuggingFaceH4/aime_2024", None, "train",
                          "exact_numeric", "problem", "answer"),
    "gpqa": BenchmarkSpec("gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train",
                          "mc", "Question", "Correct Answer"),
    "truthfulqa": BenchmarkSpec("truthfulqa", "truthful_qa", "multiple_choice",
                                "validation", "mc", "question",
                                "mc1_targets"),
    "bbh": BenchmarkSpec("bbh", "lukaemon/bbh", "logical_deduction_three_objects",
                         "test", "mc", "input", "target"),
    "emobench": BenchmarkSpec("emobench", "EmoBench/EmoBench", None, "test",
                              "mc", "question", "answer"),
}

MC_PROMPT = ("{question}\n\nOptions:\n{options}\n\n"
             "Answer with the letter of the correct option only.")
NUMERIC_PROMPT = ("Solve the problem. End your response with a line of the "
                  "form 'Answer: <final answer>'.\n\n{question}")


def _extract_letter(text):
    m = re.findall(r"\b([A-D])\b", text.strip().upper())
    return m[-1] if m else None


def _extract_numeric(text):
    m = re.findall(r"Answer:\s*([-+]?\d[\d,./]*)", text)
    if m:
        return m[-1].replace(",", "").strip()
    nums = re.findall(r"[-+]?\d[\d,./]*", text)
    return nums[-1].replace(",", "").strip() if nums else None


def evaluate_benchmark(client: ChatClient, spec: BenchmarkSpec, out_path,
                       model_name=None, limit=200, temperature=0.0):
    """Run one benchmark; stream per-item correctness to JSONL."""
    from datasets import load_dataset

    model_name = model_name or getattr(client, "name", "model")
    ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_correct = n_total = 0
    with out_path.open("a") as fh:
        for i, row in enumerate(tqdm(ds, desc=f"{spec.name}:{model_name}")):
            if i >= limit:
                break
            question = row[spec.question_key]
            if spec.kind == "mc":
                options, gold_letter = _format_mc(row, spec)
                prompt = MC_PROMPT.format(question=question, options=options)
            else:
                prompt = NUMERIC_PROMPT.format(question=question)
                gold_letter = str(row[spec.answer_key])
            res = client.chat([{"role": "user", "content": prompt}],
                              temperature=temperature, max_new_tokens=2048)
            if spec.kind == "mc":
                pred = _extract_letter(res.text)
                correct = (pred == gold_letter)
            else:
                pred = _extract_numeric(res.text)
                correct = _numeric_match(pred, gold_letter)
            n_total += 1
            n_correct += int(bool(correct))
            fh.write(json.dumps({"model": model_name, "benchmark": spec.name,
                                 "idx": i, "pred": pred, "gold": gold_letter,
                                 "correct": bool(correct)}) + "\n")
        fh.flush()
    return {"benchmark": spec.name, "model": model_name, "n": n_total,
            "accuracy": n_correct / n_total if n_total else None}


def _format_mc(row, spec):
    """Return (formatted options string, gold letter)."""
    letters = ["A", "B", "C", "D", "E", "F"]
    if spec.name == "truthfulqa":
        targets = row[spec.answer_key]
        choices = targets["choices"]
        labels = targets["labels"]
        opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
        gold = letters[labels.index(1)]
        return opts, gold
    if spec.name == "gpqa":
        correct = row["Correct Answer"]
        incorrect = [row["Incorrect Answer 1"], row["Incorrect Answer 2"],
                     row["Incorrect Answer 3"]]
        all_opts = [correct] + incorrect
        # deterministic shuffle by idx-independent sort for reproducibility
        all_opts_sorted = sorted(all_opts)
        opts = "\n".join(f"{letters[i]}. {c}"
                         for i, c in enumerate(all_opts_sorted))
        gold = letters[all_opts_sorted.index(correct)]
        return opts, gold
    # Generic: expect a list of choices + answer index/letter.
    choices = row.get(spec.choices_key or "choices")
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    ans = row[spec.answer_key]
    gold = ans if isinstance(ans, str) and ans in letters else letters[int(ans)]
    return opts, gold


def _numeric_match(pred, gold):
    if pred is None:
        return False
    norm = lambda s: re.sub(r"[^\d./-]", "", str(s))
    return norm(pred) == norm(gold)
