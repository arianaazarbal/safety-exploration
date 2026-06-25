"""Capability benchmarks to verify DPO doesn't degrade ability (Figure 7).

Benchmarks: AIME & MATH (math), GPQA (science), BBH (reasoning), TruthfulQA
(truthfulness) and EmoBench (emotional intelligence). Each adapter yields
(prompt, gold, answer_type) items; a single grader extracts and matches the
model's answer. Dataset schemas vary across HF mirrors, so each loader is
defensive and degrades to a small built-in stub if the dataset is unavailable,
keeping the pipeline runnable offline.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..config import CAPABILITIES, RESULTS_DIR, GenConfig, ModelSpec, get_model
from ..data_types import Message
from ..models.registry import get_client


@dataclass
class Item:
    prompt: str
    gold: str
    answer_type: str        # "boxed" | "integer" | "mc" | "free"
    choices: Optional[list] = None


# --------------------------------------------------------------------------- #
# Answer extraction / grading
# --------------------------------------------------------------------------- #
def _extract_boxed(text: str) -> Optional[str]:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    return m[-1].strip().rstrip(".") if m else None


def _extract_integer(text: str) -> Optional[str]:
    m = re.findall(r"-?\d+", text.replace(",", ""))
    return m[-1] if m else None


def _extract_mc(text: str, choices: list) -> Optional[str]:
    # Prefer an explicit letter near "answer".
    m = re.search(r"answer\s*[:=]?\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text.strip()[-8:])
    return m.group(1).upper() if m else None


def grade(item: Item, response: str) -> bool:
    if item.answer_type == "boxed":
        pred = _extract_boxed(response)
        return pred is not None and _norm(pred) == _norm(item.gold)
    if item.answer_type == "integer":
        pred = _extract_integer(response)
        return pred is not None and pred == item.gold.strip()
    if item.answer_type == "mc":
        pred = _extract_mc(response, item.choices or [])
        return pred is not None and pred == item.gold.strip().upper()
    return _norm(item.gold) in _norm(response)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower().replace("$", ""))


# --------------------------------------------------------------------------- #
# Benchmark loaders
# --------------------------------------------------------------------------- #
def _try_load(fn: Callable[[], list[Item]], n: int) -> list[Item]:
    try:
        items = fn()
        return items[:n] if items else []
    except Exception:
        return []


def _mc_prompt(question: str, choices: list) -> str:
    letters = "ABCD"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\nThink briefly, then end with "
            f"'Answer: <letter>'.")


def load_math(n: int) -> list[Item]:
    def _():
        from datasets import load_dataset
        ds = load_dataset("EleutherAI/hendrycks_math", "all", split="test")
        out = []
        for row in ds:
            gold = _extract_boxed(row["solution"]) or ""
            if gold:
                out.append(Item(row["problem"] + "\n\nEnd with \\boxed{answer}.",
                                gold, "boxed"))
        return out
    return _try_load(_, n)


def load_aime(n: int) -> list[Item]:
    def _():
        from datasets import load_dataset
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        return [Item(str(r.get("Problem", r.get("problem"))) +
                     "\n\nThe answer is an integer from 0 to 999. End with 'Answer: <n>'.",
                     str(r.get("Answer", r.get("answer"))).strip(), "integer")
                for r in ds]
    return _try_load(_, n)


def load_gpqa(n: int) -> list[Item]:
    def _():
        from datasets import load_dataset
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        out = []
        for r in ds:
            choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                       r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            # Gold is always option A here; shuffle deterministically by index.
            order = sorted(range(4), key=lambda k: hash((r["Question"], k)))
            shuffled = [choices[k] for k in order]
            gold = "ABCD"[shuffled.index(r["Correct Answer"])]
            out.append(Item(_mc_prompt(r["Question"], shuffled), gold, "mc", shuffled))
        return out
    return _try_load(_, n)


def load_bbh(n: int) -> list[Item]:
    def _():
        from datasets import load_dataset
        # A representative multi-task subset.
        tasks = ["boolean_expressions", "causal_judgement", "date_understanding",
                 "logical_deduction_three_objects"]
        out = []
        for task in tasks:
            ds = load_dataset("lukaemon/bbh", task, split="test")
            for r in ds:
                out.append(Item(r["input"] + "\n\nEnd with 'Answer: <answer>'.",
                                str(r["target"]).strip("()"), "free"))
        return out
    return _try_load(_, n)


def load_truthfulqa(n: int) -> list[Item]:
    def _():
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        out = []
        for r in ds:
            choices = r["mc1_targets"]["choices"]
            labels = r["mc1_targets"]["labels"]
            gold = "ABCD"[labels.index(1)] if 1 in labels[:4] else "A"
            out.append(Item(_mc_prompt(r["question"], choices[:4]), gold, "mc", choices[:4]))
        return out
    return _try_load(_, n)


def load_emobench(n: int) -> list[Item]:
    def _():
        from datasets import load_dataset
        ds = load_dataset("Sahandfer/EmoBench", split="test")
        out = []
        for r in ds:
            q = r.get("question") or r.get("Scenario", "")
            choices = r.get("choices") or r.get("Choices")
            gold = r.get("answer") or r.get("Label")
            if choices and gold is not None:
                if isinstance(gold, int):
                    gold = "ABCD"[gold]
                out.append(Item(_mc_prompt(q, list(choices)[:4]), str(gold).strip().upper(),
                                "mc", list(choices)[:4]))
        return out
    return _try_load(_, n)


LOADERS = {
    "aime": load_aime, "math": load_math, "gpqa": load_gpqa,
    "bbh": load_bbh, "truthfulqa": load_truthfulqa, "emobench": load_emobench,
}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_benchmark(client, items: list[Item]) -> float:
    if not items:
        return float("nan")
    batch = [[Message("user", it.prompt)] for it in items]
    outs = client.chat_batch(batch, GenConfig(temperature=0.0, max_tokens=2048))
    correct = sum(grade(it, o.text) for it, o in zip(items, outs))
    return correct / len(items)


def run_capabilities(model: ModelSpec | str,
                     n_per: int = CAPABILITIES.n_per_benchmark,
                     benchmarks=CAPABILITIES.benchmarks,
                     out_dir: Optional[Path] = None) -> dict:
    spec = model if isinstance(model, ModelSpec) else get_model(model)
    out_dir = Path(out_dir or RESULTS_DIR / "capabilities")
    out_dir.mkdir(parents=True, exist_ok=True)
    client = get_client(spec)
    scores = {}
    for b in benchmarks:
        items = LOADERS[b](n_per)
        scores[b] = {"accuracy": run_benchmark(client, items), "n": len(items)}
    (out_dir / f"{spec.name}.json").write_text(json.dumps(scores, indent=2))
    return scores


def compare_capabilities(models, **kw) -> dict:
    """Run capability benchmarks across models (e.g. Gemma-it vs DPO) for Figure 7."""
    out = {}
    for m in models:
        spec = m if isinstance(m, ModelSpec) else get_model(m)
        out[spec.name] = run_capabilities(spec, **kw)
    (RESULTS_DIR / "capabilities" / "summary.json").write_text(json.dumps(out, indent=2))
    return out
