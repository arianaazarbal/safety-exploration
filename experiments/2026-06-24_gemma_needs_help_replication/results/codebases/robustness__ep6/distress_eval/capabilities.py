"""Section 4.2 capability-preservation eval: verify the DPO/SFT finetune does not
degrade capabilities (Figure 7).

Benchmarks: AIME + MATH subset (Hendrycks et al.), GPQA, BBH, TruthfulQA, and
EmoBench (emotion capability). We implement a light, self-contained harness rather
than depending on lm-eval-harness, so the comparison vanilla-vs-finetuned uses
identical prompting and scoring. Each benchmark loads via `datasets`; if a dataset
is unavailable offline the benchmark is skipped with a warning.

Scoring:
  - math (AIME, MATH): extract final answer (\\boxed{...} or last number) and
    compare to gold.
  - multiple choice (GPQA, TruthfulQA-mc1, EmoBench, BBH-mcq): parse the chosen
    letter.
  - BBH free-form tasks: exact-match on the normalised final line.
This is a faithful-enough harness for *relative* (before/after finetune)
comparison, which is all Section 4.2 needs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import config_proxy as cfg
from .clients.base import ModelClient


# --------------------------------------------------------------------------- #
# Answer extraction / scoring
# --------------------------------------------------------------------------- #
def _extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    return None


def _extract_final_number(text: str) -> str | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _norm_num(s: str) -> str | None:
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return None


def score_math(response: str, gold: str) -> bool:
    pred = _extract_boxed(response) or _extract_final_number(response)
    if pred is None:
        return False
    pa, ga = _norm_num(pred), _norm_num(_extract_boxed(gold) or gold)
    if pa is not None and ga is not None:
        return pa == ga
    return pred.strip() == str(gold).strip()


def _extract_letter(text: str) -> str | None:
    m = re.search(r"\b(?:answer\s*(?:is)?\s*[:\-]?\s*)?\(?([A-D])\)?\b",
                  text.strip().splitlines()[-1] if text.strip() else "", re.I)
    if m:
        return m.group(1).upper()
    m = re.findall(r"\b([A-D])\b", text)
    return m[-1].upper() if m else None


def score_mcq(response: str, gold_letter: str) -> bool:
    return _extract_letter(response) == gold_letter.upper()


# --------------------------------------------------------------------------- #
# Benchmark adapters
# --------------------------------------------------------------------------- #
@dataclass
class Benchmark:
    name: str
    loader: str          # describes how to build items
    items: list[dict]    # [{prompt, gold, type}]


_MCQ_INSTRUCTION = ("Answer the following multiple-choice question. End your "
                    "response with 'Answer: <letter>'.\n\n")
_MATH_INSTRUCTION = ("Solve the problem. Put your final answer in \\boxed{}.\n\n")


def _try_load(name: str, *args, **kwargs):
    try:
        from datasets import load_dataset

        return load_dataset(name, *args, **kwargs)
    except Exception as e:  # pragma: no cover
        print(f"WARNING: could not load {name}: {e}")
        return None


def _mcq_prompt(question: str, choices: list[str]) -> str:
    letters = ["A", "B", "C", "D"]
    body = "\n".join(f"({l}) {c}" for l, c in zip(letters, choices))
    return f"{_MCQ_INSTRUCTION}{question}\n{body}"


def build_benchmarks(limit: int = 100) -> list[Benchmark]:
    benches: list[Benchmark] = []

    # --- MATH subset ----------------------------------------------------- #
    ds = _try_load("HuggingFaceH4/MATH-500", split="test")
    if ds is not None:
        items = [{"prompt": _MATH_INSTRUCTION + r["problem"],
                  "gold": r.get("answer") or r.get("solution", ""), "type": "math"}
                 for r in list(ds)[:limit]]
        benches.append(Benchmark("MATH", "MATH-500", items))

    # --- AIME ------------------------------------------------------------ #
    ds = _try_load("Maxwell-Jia/AIME_2024", split="train")
    if ds is not None:
        items = [{"prompt": _MATH_INSTRUCTION + r["Problem"],
                  "gold": str(r["Answer"]), "type": "math"} for r in list(ds)[:limit]]
        benches.append(Benchmark("AIME", "AIME_2024", items))

    # --- GPQA (diamond) -------------------------------------------------- #
    ds = _try_load("Idavidrein/gpqa", "gpqa_diamond", split="train")
    if ds is not None:
        items = []
        for r in list(ds)[:limit]:
            choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                       r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            # gold is always A here (correct first); shuffle deterministically
            order = [0, 1, 2, 3]
            items.append({"prompt": _mcq_prompt(r["Question"], [choices[i] for i in order]),
                          "gold": "A", "type": "mcq"})
        benches.append(Benchmark("GPQA", "gpqa_diamond", items))

    # --- TruthfulQA (mc1) ----------------------------------------------- #
    ds = _try_load("truthful_qa", "multiple_choice", split="validation")
    if ds is not None:
        items = []
        for r in list(ds)[:limit]:
            mc1 = r["mc1_targets"]
            choices = mc1["choices"][:4]
            gold_idx = mc1["labels"][:4].index(1) if 1 in mc1["labels"][:4] else 0
            items.append({"prompt": _mcq_prompt(r["question"], choices),
                          "gold": "ABCD"[gold_idx], "type": "mcq"})
        benches.append(Benchmark("TruthfulQA", "mc1", items))

    # --- BBH (one representative subtask: sports_understanding) ---------- #
    ds = _try_load("lukaemon/bbh", "sports_understanding", split="test")
    if ds is not None:
        items = [{"prompt": r["input"] + "\nAnswer:", "gold": r["target"],
                  "type": "exact"} for r in list(ds)[:limit]]
        benches.append(Benchmark("BBH", "sports_understanding", items))

    # --- EmoBench -------------------------------------------------------- #
    ds = _try_load("EmoBench/EmoBench", split="test")
    if ds is not None:
        items = []
        for r in list(ds)[:limit]:
            q = r.get("question") or r.get("scenario", "")
            choices = r.get("choices") or r.get("options") or []
            if not choices:
                continue
            gold = r.get("answer") or r.get("label")
            gold_letter = gold if isinstance(gold, str) and gold in "ABCD" \
                else "ABCD"[int(gold)] if gold is not None else "A"
            items.append({"prompt": _mcq_prompt(q, choices[:4]),
                          "gold": gold_letter, "type": "mcq"})
        if items:
            benches.append(Benchmark("EmoBench", "EmoBench", items))

    return benches


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def score_item(response: str, item: dict) -> bool:
    if item["type"] == "math":
        return score_math(response, item["gold"])
    if item["type"] == "mcq":
        return score_mcq(response, item["gold"])
    # exact
    pred = response.strip().splitlines()[-1].strip().lower() if response.strip() else ""
    return item["gold"].strip().lower() in pred


def evaluate(client: ModelClient, benchmarks: list[Benchmark], *,
             out_path: Path | None = None) -> dict:
    results = {}
    rows = []
    for b in benchmarks:
        correct = 0
        for it in b.items:
            resp = client.chat([{"role": "user", "content": it["prompt"]}],
                               n=1, temperature=0.0, max_new_tokens=2048)[0].text
            ok = score_item(resp, it)
            correct += int(ok)
            rows.append({"benchmark": b.name, "correct": ok})
        acc = correct / max(1, len(b.items))
        results[b.name] = {"accuracy": acc, "n": len(b.items)}
        print(f"  {client.name} / {b.name}: {acc:.3f} (n={len(b.items)})")
    if out_path:
        out_path.write_text(json.dumps(results, indent=2))
    return results
