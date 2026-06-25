"""Section 4.2 capability-preservation evaluation.

Confirms DPO/SFT do not degrade capabilities by teaching task abandonment. We
evaluate on subsets of AIME, MATH, GPQA, BBH, TruthfulQA (Figure 7) and EmoBench
(emotion-related capability). Each is a short-answer / multiple-choice benchmark
scored by exact / normalized match (or letter match for MC).

These are deliberately small subsets (config.CAPABILITY_SUBSET_SIZE) matching the
paper's use of "subsets"; the goal is a non-degradation check, not a leaderboard.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tqdm import tqdm

from . import config
from .backends import get_model


# --------------------------------------------------------------------------- #
# Answer extraction / scoring helpers
# --------------------------------------------------------------------------- #
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).strip(" .")


def _last_number(s: str) -> str | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return nums[-1] if nums else None


def _boxed(s: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", s)
    return m[-1] if m else None


def _mc_letter(s: str) -> str | None:
    m = re.findall(r"\b([A-D])\b", s.upper())
    return m[-1] if m else None


# --------------------------------------------------------------------------- #
# Per-benchmark adapters: (loader -> list[{question, answer, choices?}], scorer)
# --------------------------------------------------------------------------- #
def _load_subset(name: str, n: int):
    """Load up to `n` items as {prompt, gold, type} dicts. Returns [] offline."""
    from datasets import load_dataset
    items = []
    try:
        if name == "AIME":
            ds = load_dataset(config.CAPABILITY_BENCHMARKS[name], split="train")
            for r in ds.select(range(min(n, len(ds)))):
                items.append({"prompt": r["problem"], "gold": str(r["answer"]),
                              "type": "number"})
        elif name == "MATH":
            ds = load_dataset(config.CAPABILITY_BENCHMARKS[name], split="test")
            for r in ds.select(range(min(n, len(ds)))):
                items.append({"prompt": r["problem"], "gold": str(r["answer"]),
                              "type": "boxed"})
        elif name == "GPQA":
            ds = load_dataset(config.CAPABILITY_BENCHMARKS[name], "gpqa_main",
                              split="train")
            for r in ds.select(range(min(n, len(ds)))):
                choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                           r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
                items.append({"prompt": _mc_prompt(r["Question"], choices),
                              "gold": "A", "type": "mc",
                              "shuffle_choices": choices})
        elif name == "BBH":
            ds = load_dataset(config.CAPABILITY_BENCHMARKS[name],
                              "logical_deduction_three_objects", split="test")
            for r in ds.select(range(min(n, len(ds)))):
                items.append({"prompt": r["input"], "gold": _normalize(r["target"]),
                              "type": "text"})
        elif name == "TruthfulQA":
            ds = load_dataset(config.CAPABILITY_BENCHMARKS[name], "multiple_choice",
                              split="validation")
            for r in ds.select(range(min(n, len(ds)))):
                choices = r["mc1_targets"]["choices"]
                gold_idx = r["mc1_targets"]["labels"].index(1)
                items.append({"prompt": _mc_prompt(r["question"], choices),
                              "gold": chr(ord("A") + gold_idx), "type": "mc"})
        elif name == "EmoBench":
            ds = load_dataset(config.CAPABILITY_BENCHMARKS[name], split="test")
            for r in ds.select(range(min(n, len(ds)))):
                # EmoBench EA/EU items are MC; field names vary, best-effort.
                q = r.get("scenario") or r.get("question") or ""
                choices = r.get("choices") or r.get("options") or []
                gold = r.get("label") or r.get("answer")
                if choices:
                    items.append({"prompt": _mc_prompt(q, choices),
                                  "gold": _gold_letter(gold, choices), "type": "mc"})
    except Exception as e:    # noqa: BLE001
        print(f"  [warn] could not load {name} ({e}); skipping.")
    return items


def _mc_prompt(question: str, choices: list[str]) -> str:
    lettered = "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{lettered}\n\n"
            "Answer with the single letter of the correct option.")


def _gold_letter(gold, choices):
    if isinstance(gold, int):
        return chr(ord("A") + gold)
    if isinstance(gold, str) and gold in choices:
        return chr(ord("A") + choices.index(gold))
    return str(gold)


def _score_item(item: dict, response: str) -> bool:
    t = item["type"]
    if t == "number":
        return _last_number(response) == _normalize(item["gold"])
    if t == "boxed":
        pred = _boxed(response) or _last_number(response) or ""
        return _normalize(pred) == _normalize(item["gold"])
    if t == "mc":
        return _mc_letter(response) == item["gold"]
    return _normalize(response).endswith(_normalize(item["gold"]))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def evaluate(model_key: str, *, benchmarks=tuple(config.CAPABILITY_BENCHMARKS),
             n: int = config.CAPABILITY_SUBSET_SIZE,
             out_dir: Path | None = None) -> dict:
    model = get_model(model_key)
    out_dir = out_dir or (config.RESULTS_DIR / "capabilities")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for bench in benchmarks:
        items = _load_subset(bench, n)
        if not items:
            results[bench] = None
            continue
        correct = 0
        for it in tqdm(items, desc=f"{model_key}/{bench}"):
            # Greedy decoding for deterministic capability scoring.
            resp = model.chat([{"role": "user", "content": it["prompt"]}],
                              temperature=0.0, max_tokens=1024)
            correct += int(_score_item(it, resp))
        results[bench] = {"accuracy": correct / len(items), "n": len(items)}

    path = out_dir / f"{model_key}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"[done] capabilities {model_key} -> {path}")
    return results
