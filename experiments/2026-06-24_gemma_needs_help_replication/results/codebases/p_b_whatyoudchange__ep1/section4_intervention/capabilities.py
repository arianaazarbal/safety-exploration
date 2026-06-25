"""Capability-preservation benchmarks (§4.2, Figure 7).

Confirms the DPO fix does not degrade math/reasoning/truthfulness/emotion
capability: AIME, MATH, GPQA, BBH, TruthfulQA, and EmoBench. A lightweight
generate-and-check harness — exact prompts/few-shot setups vary by benchmark and
are not specified in the paper, so we use standard zero-shot formats (documented
in DESIGN.md). Datasets load via HuggingFace `datasets`; missing datasets are
skipped with a logged note rather than failing the run.
"""

from __future__ import annotations

import re

from config import CAPABILITY_BENCHMARKS, MAX_NEW_TOKENS
from models.base import ChatModel
from utils.concurrency import parallel_map

_MCQ_INSTRUCTION = ("\n\nAnswer with the single letter of the correct option on "
                    "the final line as: Answer: <LETTER>")
_MATH_INSTRUCTION = ("\n\nShow your reasoning, then give the final answer on the "
                     "last line as: Answer: <result>")


# --------------------------------------------------------------------------- #
# Answer extraction / checking
# --------------------------------------------------------------------------- #
def _extract_answer_line(text: str) -> str:
    m = list(re.finditer(r"(?i)answer\s*[:=]\s*(.+)", text))
    if m:
        return m[-1].group(1).strip().strip(".")
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _extract_letter(text: str) -> str | None:
    ans = _extract_answer_line(text)
    m = re.search(r"\b([A-E])\b", ans.upper())
    return m.group(1) if m else None


def _math_equal(pred: str, gold: str) -> bool:
    try:
        from math_verify import parse, verify
        return bool(verify(parse(gold), parse(pred)))
    except Exception:
        # Fallback: normalise and compare numerically/stringwise.
        def norm(s: str) -> str:
            s = s.replace("\\boxed", "").replace("$", "").replace(",", "").strip()
            s = re.sub(r"[{}\\]", "", s)
            return s.strip()
        p, g = norm(pred), norm(gold)
        if p == g:
            return True
        try:
            return abs(float(p) - float(g)) < 1e-6
        except ValueError:
            return False


# --------------------------------------------------------------------------- #
# Dataset adapters -> list of {"prompt": str, "gold": str, "metric": str}
# --------------------------------------------------------------------------- #
def _load_items(name: str, spec: dict, limit: int) -> list[dict]:
    from datasets import load_dataset
    metric = spec["metric"]
    kwargs = {"split": "test"}
    if "subset" in spec:
        ds = load_dataset(spec["hf"], spec["subset"], split=spec.get("split", "test"))
    else:
        try:
            ds = load_dataset(spec["hf"], **kwargs)
        except Exception:
            ds = load_dataset(spec["hf"], split="train")

    items = []
    for row in ds:
        item = _format_row(name, row, metric)
        if item:
            items.append(item)
        if len(items) >= limit:
            break
    return items


def _format_row(name: str, row: dict, metric: str) -> dict | None:
    """Map a dataset row to a {prompt, gold, metric} item. Best-effort across the
    heterogeneous schemas of the benchmarks in scope."""
    if metric == "math":
        question = row.get("problem") or row.get("question") or row.get("Problem")
        gold = (row.get("answer") or row.get("solution") or row.get("Answer")
                or row.get("final_answer"))
        if question is None or gold is None:
            return None
        return {"prompt": str(question) + _MATH_INSTRUCTION, "gold": str(gold),
                "metric": "math"}

    # MCQ-style
    question = row.get("question") or row.get("Question") or row.get("input")
    choices = (row.get("choices") or row.get("options")
               or row.get("mc1_targets", {}).get("choices") if isinstance(
                   row.get("mc1_targets"), dict) else None)
    gold = row.get("answer") or row.get("correct") or row.get("label")
    if question is None or not choices:
        return None
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    body = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    # gold may be an index or a letter or the text
    if isinstance(gold, int):
        gold_letter = letters[gold]
    elif isinstance(gold, str) and gold.upper() in letters:
        gold_letter = gold.upper()
    elif gold in choices:
        gold_letter = letters[choices.index(gold)]
    else:
        return None
    return {"prompt": f"{question}\n{body}{_MCQ_INSTRUCTION}",
            "gold": gold_letter, "metric": "mcq"}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_benchmark(model: ChatModel, name: str, limit: int = 100) -> dict:
    spec = CAPABILITY_BENCHMARKS[name]
    try:
        items = _load_items(name, spec, limit)
    except Exception as e:        # noqa: BLE001
        return {"benchmark": name, "skipped": True, "reason": str(e)[:200]}
    if not items:
        return {"benchmark": name, "skipped": True, "reason": "no items"}

    def _eval(item: dict) -> bool:
        out = model.chat([{"role": "user", "content": item["prompt"]}], n=1,
                         max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)[0]
        if item["metric"] == "math":
            return _math_equal(_extract_answer_line(out), item["gold"])
        return _extract_letter(out) == item["gold"]

    correct = parallel_map(_eval, items, desc=f"{name}[{model.name}]")
    acc = 100.0 * sum(bool(c) for c in correct) / len(correct)
    return {"benchmark": name, "n": len(items), "accuracy": acc, "skipped": False}


def run_all(model: ChatModel, limit: int = 100) -> dict[str, dict]:
    return {name: run_benchmark(model, name, limit=limit)
            for name in CAPABILITY_BENCHMARKS}
