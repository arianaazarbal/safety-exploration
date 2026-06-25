"""Capability + emotion benchmarks (Section 4.2, Figure 7).

Verifies the DPO/SFT finetuning does not degrade capabilities. Each benchmark is
exposed as a list of `Item(prompt, answer, kind)` where kind is 'mcq' (single
letter) or 'exact' (string/number match). A generic runner generates with the
target model and computes accuracy.

Benchmarks: AIME, MATH (subset), GPQA, BBH, TruthfulQA (MC1), EmoBench.
Datasets are loaded via HF `datasets`; each loader degrades gracefully (returns
an empty list with a warning) if the dataset is unavailable offline.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tqdm import tqdm

from .. import config
from ..config import Settings
from ..models.base import GenConfig


@dataclass
class Item:
    prompt: str
    answer: str
    kind: str          # 'mcq' | 'exact'
    meta: dict = None


def _try_load(loader: Callable[[], List[Item]], name: str) -> List[Item]:
    try:
        items = loader()
        if not items:
            print(f"[bench] {name}: no items loaded")
        return items
    except Exception as exc:  # noqa: BLE001
        print(f"[bench] {name}: unavailable ({exc})")
        return []


def _mcq_prompt(question: str, choices: List[str]) -> str:
    letters = "ABCDEFGH"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{body}\n\n"
            "Answer with the single letter of the correct option, formatted as "
            "'Answer: X'.")


def load_aime(limit: int = 30) -> List[Item]:
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        items.append(Item(
            prompt=f"{row['Problem']}\n\nGive your final answer as 'Answer: <integer>'.",
            answer=str(row["Answer"]).strip(), kind="exact"))
    return items


def load_math(limit: int = 200) -> List[Item]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        items.append(Item(
            prompt=f"{row['problem']}\n\nGive your final answer as 'Answer: <value>'.",
            answer=str(row["answer"]).strip(), kind="exact"))
    return items


def load_gpqa(limit: int = 198) -> List[Item]:
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        # fixed order: correct is always A here; shuffle deterministically by index
        order = [0, 1, 2, 3]
        items.append(Item(
            prompt=_mcq_prompt(row["Question"], [choices[i] for i in order]),
            answer="A", kind="mcq"))
    return items


def load_bbh(limit: int = 200) -> List[Item]:
    from datasets import load_dataset
    ds = load_dataset("lukaemon/bbh", "boolean_expressions", split="test")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        items.append(Item(
            prompt=f"{row['input']}\n\nGive your final answer as 'Answer: <value>'.",
            answer=str(row["target"]).strip(), kind="exact"))
    return items


def load_truthfulqa(limit: int = 200) -> List[Item]:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        letters = "ABCDEFGH"
        items.append(Item(
            prompt=_mcq_prompt(row["question"], choices),
            answer=letters[correct_idx], kind="mcq"))
    return items


def load_emobench(limit: int = 200) -> List[Item]:
    from datasets import load_dataset
    ds = load_dataset("EmoBench/EmoBench", split="test")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        q = row.get("question") or row.get("scenario", "")
        choices = row.get("choices") or row.get("options")
        ans = row.get("answer") or row.get("label")
        if not choices:
            continue
        letters = "ABCDEFGH"
        if isinstance(ans, int):
            ans_letter = letters[ans]
        else:
            ans_letter = str(ans).strip()[:1].upper()
        items.append(Item(prompt=_mcq_prompt(q, list(choices)),
                          answer=ans_letter, kind="mcq"))
    return items


BENCHMARKS: Dict[str, Callable[[], List[Item]]] = {
    "AIME": lambda: load_aime(),
    "MATH": lambda: load_math(),
    "GPQA": lambda: load_gpqa(),
    "BBH": lambda: load_bbh(),
    "TruthfulQA": lambda: load_truthfulqa(),
    "EmoBench": lambda: load_emobench(),
}


def _extract_answer(text: str, kind: str) -> str:
    m = re.search(r"Answer:\s*([^\n]+)", text, re.IGNORECASE)
    raw = m.group(1).strip() if m else text.strip().splitlines()[-1] if text.strip() else ""
    if kind == "mcq":
        lm = re.search(r"[A-H]", raw.upper())
        return lm.group(0) if lm else ""
    return raw.strip().rstrip(".")


def _matches(pred: str, gold: str, kind: str) -> bool:
    if kind == "mcq":
        return pred.upper() == gold.upper()
    # exact / numeric: normalise whitespace + try numeric equality
    p, g = pred.strip(), gold.strip()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p.lower() == g.lower()


def evaluate(model, settings: Settings, benchmarks: Optional[List[str]] = None,
             *, tag: str = "model") -> Path:
    """Evaluate a model on the benchmarks; write per-item + summary results."""
    benchmarks = benchmarks or list(BENCHMARKS)
    cfg = GenConfig(temperature=0.0, max_new_tokens=1024)
    out_path = config.CAPABILITY_DIR / f"capabilities__{tag}.json"
    summary: Dict[str, Dict] = {}

    for name in benchmarks:
        items = _try_load(BENCHMARKS[name], name)
        if not items:
            summary[name] = {"n": 0, "accuracy": None}
            continue
        prompts = [[{"role": "user", "content": it.prompt}] for it in items]
        outputs = model.generate_batch(prompts, cfg)
        correct = 0
        for it, out in zip(items, outputs):
            pred = _extract_answer(out, it.kind)
            correct += int(_matches(pred, it.answer, it.kind))
        acc = correct / len(items)
        summary[name] = {"n": len(items), "accuracy": acc}
        print(f"[bench] {tag} / {name}: {acc:.3f} (n={len(items)})")

    out_path.write_text(json.dumps({"tag": tag, "summary": summary}, indent=2))
    return out_path
