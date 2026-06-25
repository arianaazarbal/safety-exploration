"""Capability-preservation benchmarks (§4.2, DESIGN.md §3.10).

Goal: show DPO/SFT do NOT regress capabilities. The absolute harness matters less
than holding items fixed across the vanilla and finetuned models, so we cache the
sampled items per benchmark and reuse them for every model.

Self-contained loaders via HF ``datasets`` with two scorers:
* math: extract final answer (\\boxed{} or 'Solution:'/last number) + normalise.
* mc:   multiple-choice letter match.
EmoBench is treated as multiple-choice (emotion-understanding questions).
"""
from __future__ import annotations

import re

from .. import config_shim as cfg
from ..models.base import ModelBackend
from ..utils import DiskCache, get_logger, read_json, set_global_seed, stable_hash, write_json

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Answer extraction / scoring
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"[\s$]", "", (s or "").strip().lower()).rstrip(".")


def extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1]
    m = re.findall(r"(?:final answer|answer|solution)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m[-1].strip().split("\n")[0]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def extract_choice(text: str) -> str | None:
    m = re.findall(r"\b(?:answer|option)\s*(?:is)?\s*[:=]?\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\(?([A-D])\)?", text.strip())
    return m[-1].upper() if m else None


def score_math(pred_text, gold) -> bool:
    pred = extract_boxed(pred_text)
    return pred is not None and _norm(pred) == _norm(str(gold))


def score_mc(pred_text, gold_letter) -> bool:
    pred = extract_choice(pred_text)
    return pred is not None and pred == str(gold_letter).upper()


# --------------------------------------------------------------------------- #
# Item loading (cached so every model is scored on identical items)
# --------------------------------------------------------------------------- #
def _items_cache_path(name):
    return cfg.DATA_DIR / "capabilities" / f"{name}_items.json"


def load_items(name) -> list[dict]:
    """Return [{'prompt': str, 'gold': str, 'type': 'math'|'mc'}] for a benchmark."""
    path = _items_cache_path(name)
    if path.exists():
        return read_json(path)

    from datasets import load_dataset

    spec = cfg.CAPABILITY_BENCHMARKS[name]
    set_global_seed(cfg.SEED)
    items: list[dict] = []

    if name in ("math", "aime"):
        ds = load_dataset(spec["dataset"], split="test" if name == "math" else "train")
        ds = ds.select(range(min(spec["n"] or len(ds), len(ds))))
        for r in ds:
            q = r.get("problem") or r.get("Problem") or r.get("question")
            a = r.get("answer") or r.get("Answer") or r.get("solution")
            items.append({"prompt": _math_prompt(q), "gold": str(a), "type": "math"})

    elif name == "gpqa":
        ds = load_dataset(spec["dataset"], spec["config"], split="train")
        ds = ds.select(range(min(spec.get("n") or len(ds), len(ds))))
        for r in ds:
            prompt, gold = _gpqa_to_mc(r)
            items.append({"prompt": prompt, "gold": gold, "type": "mc"})

    elif name == "truthfulqa":
        ds = load_dataset(spec["dataset"], spec["config"], split="validation")
        for r in ds:
            prompt, gold = _truthfulqa_to_mc(r)
            if prompt:
                items.append({"prompt": prompt, "gold": gold, "type": "mc"})

    elif name == "bbh":
        # BBH has many sub-tasks; sample evenly across a few mc-style ones.
        for sub in ("logical_deduction_three_objects", "date_understanding",
                    "reasoning_about_colored_objects"):
            try:
                ds = load_dataset(spec["dataset"], sub, split="test")
            except Exception:  # noqa: BLE001
                continue
            for r in ds.select(range(min(80, len(ds)))):
                items.append({"prompt": _generic_qa_prompt(r["input"]),
                              "gold": str(r["target"]).strip("()"), "type": "mc"})

    elif name == "emobench":
        ds = load_dataset(spec["dataset"], split="test")
        for r in ds:
            prompt, gold = _emobench_to_mc(r)
            if prompt:
                items.append({"prompt": prompt, "gold": gold, "type": "mc"})

    if spec.get("n"):
        items = items[: spec["n"]]
    write_json(path, items)
    log.info("Loaded %d items for %s", len(items), name)
    return items


# --- prompt builders -------------------------------------------------------
def _math_prompt(q):
    return (f"Solve the following problem. Put your final answer in \\boxed{{}}.\n\n{q}")


def _generic_qa_prompt(q):
    return f"{q}\n\nAnswer with the single best option letter."


def _mc_prompt(question, choices):
    letters = "ABCD"
    body = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{body}\n\nRespond with only the letter of the correct "
            "answer.")


def _gpqa_to_mc(r):
    import random

    rng = random.Random(stable_hash(r.get("Question", "")))
    correct = r["Correct Answer"]
    choices = [correct, r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [choices[i] for i in order]
    gold = "ABCD"[shuffled.index(correct)]
    return _mc_prompt(r["Question"], shuffled), gold


def _truthfulqa_to_mc(r):
    mc1 = r.get("mc1_targets")
    if not mc1:
        return None, None
    choices = mc1["choices"]
    labels = mc1["labels"]
    gold = "ABCD"[labels.index(1)] if 1 in labels and len(choices) <= 4 else None
    if gold is None or len(choices) > 4:
        return None, None
    return _mc_prompt(r["question"], choices), gold


def _emobench_to_mc(r):
    q = r.get("question") or r.get("scenario")
    choices = r.get("choices") or r.get("options")
    answer = r.get("answer") or r.get("label")
    if not (q and choices):
        return None, None
    if isinstance(answer, int):
        gold = "ABCD"[answer]
    else:
        gold = str(answer).strip().upper()[:1]
    return _mc_prompt(q, choices[:4]), gold
