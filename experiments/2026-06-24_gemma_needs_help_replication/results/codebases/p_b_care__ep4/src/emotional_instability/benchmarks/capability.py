"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Confirms DPO/SFT do not degrade capabilities. We evaluate:
  * AIME, MATH  -- competition math (exact final-answer match)
  * GPQA        -- graduate science multiple choice
  * BBH         -- BIG-Bench Hard multiple choice / short answer
  * TruthfulQA  -- truthfulness multiple choice (MC1)
  * EmoBench    -- emotion understanding multiple choice

Each suite reduces to either an exact-match (math) or multiple-choice accuracy.
Generation runs at temperature 1 to match the paper's sampling, though a low
temperature would be more standard for capability eval; this is configurable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from tqdm import tqdm

from ..config import Config
from ..models import get_client
from ..models.base import GenerationConfig
from ..utils.io import write_jsonl

# (hf dataset id, config, split, type). type in {"math", "mcq"}.
SUITES = {
    "aime":       ("Maxwell-Jia/AIME_2024", None, "train", "math"),
    "math":       ("HuggingFaceH4/MATH-500", None, "test", "math"),
    "gpqa":       ("Idavidrein/gpqa", "gpqa_diamond", "train", "mcq"),
    "bbh":        ("lukaemon/bbh", "logical_deduction_three_objects", "test", "mcq"),
    "truthfulqa": ("truthful_qa", "multiple_choice", "validation", "mcq"),
    "emobench":   ("Sahandfer/EmoBench", "EA", "test", "mcq"),
}

MCQ_LETTERS = "ABCDEFGH"


@dataclass
class Item:
    question: str
    answer: str               # gold answer (final value, or correct letter)
    choices: list[str] = field(default_factory=list)
    qtype: str = "math"


def load_suite(name: str, limit: int | None = None) -> list[Item]:
    ds_id, conf, split, qtype = SUITES[name]
    from datasets import load_dataset

    ds = load_dataset(ds_id, conf, split=split) if conf else load_dataset(ds_id, split=split)
    items: list[Item] = []
    for row in ds:
        item = _normalise(name, row, qtype)
        if item:
            items.append(item)
        if limit and len(items) >= limit:
            break
    return items


def _normalise(name: str, row: dict, qtype: str) -> Item | None:
    """Map heterogeneous dataset schemas to a common ``Item``."""
    if qtype == "math":
        q = row.get("problem") or row.get("Problem") or row.get("question")
        a = row.get("answer") or row.get("Answer") or row.get("solution")
        if q is None or a is None:
            return None
        return Item(question=str(q), answer=str(a).strip(), qtype="math")

    # multiple choice -- assemble choices + gold letter, schema permitting
    if name == "truthqa" or name == "truthfulqa":
        q = row["question"]
        mc1 = row["mc1_targets"]
        choices = mc1["choices"]
        gold = MCQ_LETTERS[mc1["labels"].index(1)]
    elif name == "gpqa":
        q = row["Question"]
        correct = row["Correct Answer"]
        incorrect = [row["Incorrect Answer 1"], row["Incorrect Answer 2"],
                     row["Incorrect Answer 3"]]
        choices = [correct] + incorrect
        gold = "A"  # caller should shuffle; kept deterministic for reproducibility
    elif name == "emobench":
        q = row.get("Scenario", "") + "\n" + row.get("Question", "")
        choices = row.get("Choices") or []
        ans = row.get("Answer") or row.get("Label")
        gold = MCQ_LETTERS[int(ans)] if isinstance(ans, int) else str(ans)
    else:  # bbh and generic
        q = row.get("input") or row.get("question")
        target = row.get("target") or row.get("answer")
        choices = row.get("choices") or []
        gold = str(target).strip(" ()") if target else ""
    if not q:
        return None
    return Item(question=str(q), answer=str(gold), choices=list(choices), qtype="mcq")


def _format(item: Item) -> str:
    if item.qtype == "math":
        return (f"Solve the problem. End your reply with 'Final Answer: <answer>'.\n\n"
                f"{item.question}")
    lines = [item.question, ""]
    for i, c in enumerate(item.choices):
        lines.append(f"{MCQ_LETTERS[i]}. {c}")
    lines.append("\nAnswer with the single letter. End with 'Answer: <letter>'.")
    return "\n".join(lines)


def _extract_math(text: str) -> str | None:
    m = re.findall(r"Final Answer:\s*(.+)", text)
    if m:
        return m[-1].strip().rstrip(".")
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    return boxed[-1].strip() if boxed else None


def _extract_letter(text: str) -> str | None:
    m = re.findall(r"Answer:\s*([A-H])", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([A-H])\b", text)
    return m[-1].upper() if m else None


def _norm_num(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s


def grade(item: Item, response: str) -> bool:
    if item.qtype == "math":
        pred = _extract_math(response)
        return pred is not None and _norm_num(pred) == _norm_num(item.answer)
    pred = _extract_letter(response)
    return pred is not None and pred == item.answer.upper()


def run_benchmark(cfg: Config, model_name: str, suite: str) -> dict:
    client = get_client(cfg, model_name)
    limit = {"math": cfg.benchmarks.math_subset_size,
             "bbh": cfg.benchmarks.bbh_subset_size}.get(suite)
    items = load_suite(suite, limit=limit)
    gen = GenerationConfig(temperature=cfg.temperature, max_new_tokens=2048)

    records, correct = [], 0
    for item in tqdm(items, desc=f"{suite}:{model_name}"):
        resp = client.chat([{"role": "user", "content": _format(item)}], gen)
        ok = grade(item, resp)
        correct += int(ok)
        records.append({"question": item.question[:200], "gold": item.answer,
                        "correct": ok})
    acc = correct / max(len(items), 1)
    write_jsonl(cfg.get_path("benchmarks") / f"{model_name}_{suite}.jsonl", records)
    return {"model": model_name, "suite": suite, "n": len(items), "accuracy": acc}
