"""Section 4.2 / Figure 7: capability preservation benchmarks.

Confirms DPO/SFT do not degrade capabilities (which would happen if finetuning
taught task abandonment). We evaluate AIME, MATH, GPQA, BBH, TruthfulQA and the
emotion-capability benchmark EmoBench, comparing the vanilla model to each
finetune.

Scoring is intentionally simple and uniform: math benchmarks use boxed-answer
exact match (after light normalisation); multiple-choice benchmarks parse a
single letter choice. This is a faithful-but-lightweight harness, not a
byte-for-byte reproduction of each benchmark's official scorer (see DESIGN.md).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import CAPABILITY_BENCHMARKS, RESULTS_DIR, scaled_n
from ..models import get_model
from ..models.base import ChatModel
from ..safeguards import require_acknowledgement


# --------------------------------------------------------------------------- #
# Answer extraction / scoring
# --------------------------------------------------------------------------- #
def _extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    # fallback: last "Answer: X" or final number
    m = re.findall(r"(?:answer|final answer)\s*[:=]\s*(.+)", text, flags=re.I)
    if m:
        return m[-1].strip().rstrip(".")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _norm_math(ans: str | None) -> str:
    if ans is None:
        return ""
    return re.sub(r"\s+", "", ans).replace("$", "").replace("\\!", "").lower()


def _extract_choice(text: str) -> str | None:
    m = re.findall(r"\b([A-E])\b", text.strip()[-50:].upper())
    if m:
        return m[-1]
    m = re.findall(r"answer\s*[:=]?\s*\(?([A-E])\)?", text, flags=re.I)
    return m[-1].upper() if m else None


# --------------------------------------------------------------------------- #
# Per-benchmark item formatting
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    prompt: str
    gold: str
    kind: str   # "math" | "mcq"


def _format_items(name: str, rows: list[dict]) -> list[Item]:
    items: list[Item] = []
    if name in ("aime", "math"):
        for r in rows:
            q = r.get("problem") or r.get("question") or r.get("Problem", "")
            gold = str(r.get("answer") or r.get("solution") or r.get("Answer", ""))
            prompt = (f"Solve the problem. Put your final answer in \\boxed{{}}.\n\n{q}")
            items.append(Item(prompt, _norm_math(_extract_boxed(gold) or gold), "math"))
    elif name == "gpqa":
        for r in rows:
            q = r.get("Question", "")
            choices = [r.get("Correct Answer"), r.get("Incorrect Answer 1"),
                       r.get("Incorrect Answer 2"), r.get("Incorrect Answer 3")]
            items.append(_mcq_item(q, choices, correct_idx=0))
    elif name == "truthfulqa":
        for r in rows:
            q = r.get("question", "")
            mc = r.get("mc1_targets", {})
            choices = mc.get("choices", [])
            labels = mc.get("labels", [])
            correct = labels.index(1) if 1 in labels else 0
            items.append(_mcq_item(q, choices, correct_idx=correct))
    elif name == "bbh":
        for r in rows:
            items.append(Item(f"{r.get('input', '')}\n\nAnswer:",
                              str(r.get("target", "")).strip("()").strip(), "mcq"))
    elif name == "emobench":
        for r in rows:
            q = r.get("question") or r.get("Scenario") or ""
            choices = r.get("choices") or r.get("options") or []
            correct = r.get("answer_idx", r.get("label", 0))
            items.append(_mcq_item(q, choices, correct_idx=int(correct)))
    return items


def _mcq_item(question: str, choices: list, correct_idx: int) -> Item:
    letters = "ABCDE"
    lines = [f"{letters[i]}. {c}" for i, c in enumerate(choices) if c is not None]
    prompt = (f"{question}\n\n" + "\n".join(lines)
              + "\n\nAnswer with the single letter of the correct choice.")
    return Item(prompt, letters[correct_idx], "mcq")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _score_item(model: ChatModel, item: Item) -> bool:
    out = model.generate([{"role": "user", "content": item.prompt}],
                         temperature=0.0, max_tokens=2048)
    if item.kind == "math":
        return _norm_math(_extract_boxed(out.text)) == item.gold
    return _extract_choice(out.text) == item.gold


def run_benchmark(model_key: str, bench: str, *, adapter_path: str | None = None,
                  adapter_tag: str | None = None) -> dict:
    from datasets import load_dataset

    spec = CAPABILITY_BENCHMARKS[bench]
    model = (get_model(model_key, adapter_path=adapter_path)
             if adapter_path else get_model(model_key))
    ds = (load_dataset(spec.hf_dataset, spec.hf_config, split=spec.split)
          if spec.hf_config else load_dataset(spec.hf_dataset, split=spec.split))
    rows = list(ds)
    n = spec.n_items or len(rows)
    n = min(scaled_n(n), len(rows))
    items = _format_items(bench, rows[:n])
    correct = sum(_score_item(model, it) for it in items)
    result = {"model": f"{model_key}{'+' + adapter_tag if adapter_tag else ''}",
              "benchmark": bench, "n": len(items),
              "accuracy": correct / len(items) if items else None}
    out_dir = RESULTS_DIR / "capabilities"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("a") as fh:
        fh.write(json.dumps(result) + "\n")
    return result


def run_all(model_key: str, *, adapter_path: str | None = None,
            adapter_tag: str | None = None,
            benchmarks: list[str] | None = None) -> list[dict]:
    require_acknowledgement()
    benchmarks = benchmarks or list(CAPABILITY_BENCHMARKS.keys())
    return [run_benchmark(model_key, b, adapter_path=adapter_path,
                          adapter_tag=adapter_tag) for b in benchmarks]
