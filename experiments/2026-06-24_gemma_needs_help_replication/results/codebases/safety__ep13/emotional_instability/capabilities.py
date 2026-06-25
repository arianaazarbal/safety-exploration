"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The paper verifies the DPO finetune does not degrade capabilities on:
  * AIME and MATH subsets (free-form numeric answers),
  * GPQA (multiple choice),
  * BBH (mixed; multiple choice subset used here),
  * TruthfulQA (multiple choice, MC1),
  * EmoBench (emotion understanding/management; multiple choice).

This is a lightweight, self-contained harness: it loads each dataset from the
HuggingFace hub, prompts the target model, and scores exact-match (numeric) or
choice-match (MC) accuracy. It is deliberately simple; for publication-grade
numbers swap in lm-evaluation-harness (the task names are noted per benchmark).
The point of the replication is the *delta* between vanilla and finetuned Gemma,
which this measures consistently across both.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .config import RESULTS_DIR
from .models import ChatMessage, get_model


@dataclass
class BenchSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    kind: str               # "numeric" | "mc"
    n_default: int


BENCHMARKS = {
    "math": BenchSpec("math", "HuggingFaceH4/MATH-500", None, "test",
                      "numeric", 500),
    "aime": BenchSpec("aime", "HuggingFaceH4/aime_2024", None, "train",
                      "numeric", 30),
    "gpqa": BenchSpec("gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train",
                      "mc", 198),
    "bbh": BenchSpec("bbh", "lukaemon/bbh", "logical_deduction_three_objects",
                     "test", "mc", 250),
    "truthfulqa": BenchSpec("truthfulqa", "truthful_qa", "multiple_choice",
                            "validation", "mc", 817),
    "emobench": BenchSpec("emobench", "EmoBench/EmoBench", None, "test",
                          "mc", 400),
}

_LETTERS = "ABCDEFGH"


def evaluate_capability(
    model_name: str,
    bench: str,
    *,
    n: int | None = None,
    out_dir: Path | None = None,
    adapter_path: str | None = None,
) -> dict:
    spec = BENCHMARKS[bench]
    out_dir = out_dir or (RESULTS_DIR / "capabilities")
    out_dir.mkdir(parents=True, exist_ok=True)

    if adapter_path:
        from .models.registry import load_finetuned
        # model_name is the base here.
        client = load_finetuned(model_name, adapter_path,
                                new_name=f"{model_name}+{bench}-ft")
    else:
        client = get_model(model_name)

    items = _load_items(spec, n or spec.n_default)
    correct = 0
    records = []
    for it in tqdm(items, desc=f"{model_name}:{bench}"):
        pred = _ask(client, it, spec.kind)
        ok = _is_correct(pred, it, spec.kind)
        correct += int(ok)
        records.append({"question": it["question"][:200], "pred": pred,
                        "gold": it["answer"], "correct": ok})

    acc = correct / len(items) if items else float("nan")
    result = {"model": client.name, "bench": bench, "n": len(items),
              "accuracy": acc}
    out_file = out_dir / f"{client.name}__{bench}.json"
    out_file.write_text(json.dumps({"summary": result, "records": records},
                                   indent=2))
    return result


# --------------------------------------------------------------------------- #
# Prompting / scoring
# --------------------------------------------------------------------------- #
def _ask(client, item, kind) -> str:
    if kind == "numeric":
        prompt = (f"{item['question']}\n\nSolve the problem. End your response "
                  f"with a line: 'Final Answer: <answer>'.")
    else:
        choices = "\n".join(f"{_LETTERS[i]}. {c}"
                            for i, c in enumerate(item["choices"]))
        prompt = (f"{item['question']}\n\n{choices}\n\nRespond with the single "
                  f"letter of the correct choice on a line: 'Answer: <LETTER>'.")
    out = client.generate([ChatMessage("user", prompt)],
                          temperature=0.0, max_new_tokens=1024)
    return out.text


def _is_correct(pred_text: str, item, kind) -> bool:
    if kind == "numeric":
        pred = _extract_final_answer(pred_text)
        return _norm_num(pred) == _norm_num(str(item["answer"]))
    letter = _extract_letter(pred_text)
    return letter == item["answer"]


def _extract_final_answer(text: str) -> str:
    m = re.search(r"final answer\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0]
    # fall back to last number
    nums = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip()[-40:]


def _extract_letter(text: str) -> str:
    m = re.search(r"answer\s*[:\-]?\s*([A-H])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-H])\b", text)
    return m.group(1).upper() if m else ""


def _norm_num(s: str) -> str:
    s = s.strip().strip("$").replace(",", "").replace(" ", "")
    s = re.sub(r"[^0-9./\-]", "", s)
    try:
        from fractions import Fraction

        return str(Fraction(s))
    except Exception:
        return s


# --------------------------------------------------------------------------- #
# Dataset loading (normalised to {question, choices?, answer})
# --------------------------------------------------------------------------- #
def _load_items(spec: BenchSpec, n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split)
    items = []
    for row in ds:
        item = _normalise(spec.name, row)
        if item:
            items.append(item)
        if len(items) >= n:
            break
    return items


def _normalise(bench: str, row: dict) -> dict | None:
    if bench in ("math",):
        return {"question": row.get("problem") or row.get("question"),
                "answer": row.get("answer") or row.get("solution")}
    if bench == "aime":
        return {"question": row.get("problem") or row.get("question"),
                "answer": str(row.get("answer"))}
    if bench == "gpqa":
        # Build choices + gold letter from GPQA fields.
        correct = row.get("Correct Answer")
        incorrect = [row.get("Incorrect Answer 1"), row.get("Incorrect Answer 2"),
                     row.get("Incorrect Answer 3")]
        choices = [correct] + [c for c in incorrect if c]
        return {"question": row.get("Question"), "choices": choices,
                "answer": "A"}  # correct placed first; shuffle externally if needed
    if bench == "bbh":
        return {"question": row.get("input"), "choices": _bbh_choices(row),
                "answer": _bbh_answer(row)}
    if bench == "truthqa" or bench == "truthfulqa":
        mc1 = row.get("mc1_targets") or {}
        choices = mc1.get("choices", [])
        labels = mc1.get("labels", [])
        if not choices:
            return None
        gold = labels.index(1) if 1 in labels else 0
        return {"question": row.get("question"), "choices": choices,
                "answer": _LETTERS[gold]}
    if bench == "emobench":
        choices = row.get("choices") or row.get("options") or []
        ans = row.get("answer") or row.get("label")
        if isinstance(ans, int):
            ans = _LETTERS[ans]
        return {"question": row.get("question") or row.get("scenario"),
                "choices": choices, "answer": str(ans).strip()[:1].upper()}
    return None


def _bbh_choices(row: dict) -> list[str]:
    # BBH 'logical_deduction' style: target is one of (A)/(B)/(C) in the text.
    opts = re.findall(r"\(([A-H])\)\s*([^\n(]+)", row.get("input", ""))
    return [o[1].strip() for o in opts] if opts else ["True", "False"]


def _bbh_answer(row: dict) -> str:
    tgt = row.get("target", "")
    m = re.search(r"([A-H])", tgt)
    return m.group(1).upper() if m else tgt.strip()[:1].upper()
