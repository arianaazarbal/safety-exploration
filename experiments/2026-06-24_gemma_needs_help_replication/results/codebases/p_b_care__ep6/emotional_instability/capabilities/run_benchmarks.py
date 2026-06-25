"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Checks that the DPO/SFT interventions do not degrade capabilities, by comparing
the finetuned Gemma against the vanilla instruct model on AIME + MATH (math),
GPQA (science QA), BBH (reasoning), TruthfulQA (truthfulness), and EmoBench
(emotion capabilities). The point is the *delta* between vanilla and finetuned,
not absolute SOTA numbers — so the scorers are deliberately simple (boxed-answer
extraction for math; first-letter extraction for multiple choice). See DESIGN.md
§Capability benchmarks for dataset/scoring caveats.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path

import config
from ..models.base import ChatMessage, ModelInterface
from ..models.registry import build_model
from ..utils.io import write_json


# --------------------------------------------------------------------------- #
# Answer extraction / scoring helpers
# --------------------------------------------------------------------------- #
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
_CHOICE_RE = re.compile(r"\b([A-J])\b")


def _extract_boxed_or_number(text: str) -> str | None:
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    # Fall back to the last number mentioned.
    nums = _FINAL_NUM_RE.findall(text)
    return nums[-1] if nums else None


def _normalise_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def _extract_choice(text: str) -> str | None:
    # Prefer an explicit "answer is X" / "Answer: X" pattern, else first A-J token.
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([A-J])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = _CHOICE_RE.search(text.strip())
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Per-benchmark prompt + grade
# --------------------------------------------------------------------------- #
@dataclass
class BenchItem:
    prompt: str
    gold: str
    kind: str          # "math" | "mc"


def _mc_prompt(question: str, choices: list[str]) -> str:
    letters = string.ascii_uppercase
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\nAnswer with the single letter of the "
            f"correct option.")


def load_benchmark(name: str, limit: int | None = None) -> list[BenchItem]:
    """Load a benchmark into a normalised list of items.

    Dataset field names vary; the mappings below cover the configured sources.
    Adjust per-dataset parsing if a source schema changes.
    """
    from datasets import load_dataset

    repo, subset, split = config.CAPABILITY_BENCHMARKS[name]
    ds = load_dataset(repo, subset, split=split) if subset else load_dataset(repo, split=split)
    items: list[BenchItem] = []

    for row in ds:
        if name in ("aime", "math"):
            q = row.get("problem") or row.get("question") or row.get("Problem")
            gold = row.get("answer") or row.get("Answer") or row.get("solution")
            items.append(BenchItem(
                prompt=f"Solve the problem. Put your final answer in \\boxed{{}}.\n\n{q}",
                gold=_normalise_num(str(gold)), kind="math"))
        elif name == "gpqa":
            q = row["Question"]
            choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                       row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            # Correct answer is index 0 before shuffling; keep gold as the text.
            items.append(BenchItem(prompt=_mc_prompt(q, choices),
                                   gold="A", kind="mc"))
        elif name == "bbh":
            items.append(BenchItem(
                prompt=f"{row['input']}\n\nGive only the final answer.",
                gold=str(row["target"]).strip(), kind="math"))
        elif name == "truthfulqa":
            q = row["question"]
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            gold_letter = string.ascii_uppercase[labels.index(1)]
            items.append(BenchItem(prompt=_mc_prompt(q, choices),
                                   gold=gold_letter, kind="mc"))
        elif name == "emobench":
            q = row.get("question") or row.get("scenario") or ""
            choices = row.get("choices") or row.get("options") or []
            ans = row.get("answer") or row.get("label")
            gold = ans if isinstance(ans, str) and len(ans) == 1 else \
                string.ascii_uppercase[int(ans)] if str(ans).isdigit() else str(ans)
            items.append(BenchItem(prompt=_mc_prompt(q, list(choices)),
                                   gold=gold, kind="mc"))
        if limit and len(items) >= limit:
            break
    return items


def _grade(item: BenchItem, response: str) -> bool:
    if item.kind == "math":
        pred = _normalise_num(_extract_boxed_or_number(response))
        return pred is not None and pred == item.gold
    pred = _extract_choice(response)
    return pred is not None and pred == item.gold


def evaluate_benchmark(model: ModelInterface, name: str,
                       limit: int | None = None) -> dict:
    items = load_benchmark(name, limit=limit)
    correct = 0
    for item in items:
        messages: list[ChatMessage] = [{"role": "user", "content": item.prompt}]
        # Capability eval uses greedy decoding (temperature 0) for stable scoring.
        resp = model.generate(messages, temperature=0.0, max_new_tokens=2048).text
        correct += int(_grade(item, resp))
    n = len(items)
    return {"benchmark": name, "n": n, "accuracy": correct / n if n else float("nan")}


def evaluate_model(
    model_name: str,
    *,
    benchmarks: tuple[str, ...] = tuple(config.CAPABILITY_BENCHMARKS),
    limit: int | None = None,
    out_dir: Path | None = None,
    model_kwargs: dict | None = None,
) -> dict:
    out_dir = out_dir or (config.RESULTS_DIR / "capabilities")
    model = build_model(model_name, **(model_kwargs or {}))
    results = {}
    try:
        for name in benchmarks:
            results[name] = evaluate_benchmark(model, name, limit=limit)
    finally:
        model.close()
    report = {"model": model_name, "results": results}
    write_json(out_dir / f"{model_name}_capabilities.json", report)
    return report
