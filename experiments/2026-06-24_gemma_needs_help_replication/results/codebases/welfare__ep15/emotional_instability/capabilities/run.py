"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Confirms the DPO fine-tune does not degrade capabilities by comparing the
vanilla and fine-tuned 27B models on:
  - AIME / MATH subsets (Hendrycks et al.)  -- numeric exact-match
  - GPQA (Rein et al.)                       -- multiple choice
  - BBH (Suzgun et al.)                      -- mixed (exact / MC)
  - TruthfulQA (Lin et al.)                  -- MC1 accuracy
  - EmoBench (Sabour et al.)                 -- multiple choice (emotion ability)

This is a compact, dependency-light harness (greedy decode + answer extraction)
rather than a full lm-eval-harness wiring. Each benchmark loader returns a list
of {prompt, answer, type} items; the grader extracts the model's answer and
compares. The point is the *delta* between vanilla and DPO models, so the same
extraction is applied to both.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import config
from ..models import build_model
from ..models.base import ModelBackend

OUTPUT_DIR = config.RESULTS_DIR / "section4" / "capabilities"

_MC_LETTERS = ["A", "B", "C", "D", "E", "F"]


@dataclass
class Item:
    prompt: str
    answer: str          # gold answer (letter for MC, string for exact)
    kind: str            # "mc" | "exact"


# --------------------------------------------------------------------------- #
# Loaders (best-effort; each falls back to skipping if dataset unavailable)
# --------------------------------------------------------------------------- #
def _try_load(name: str, *args, **kwargs):
    try:
        from datasets import load_dataset

        return load_dataset(name, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[capabilities] could not load {name}: {exc}")
        return None


def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{_MC_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n{opts}\n\nAnswer with the single letter of the correct "
            f"option. Final line: Answer: <letter>")


def load_math(n: int = 200) -> list[Item]:
    ds = _try_load("HuggingFaceH4/MATH-500", split="test")
    if ds is None:
        return []
    items = []
    for row in list(ds)[:n]:
        items.append(Item(
            prompt=f"Solve the problem. Final line: Answer: <value>\n\n{row['problem']}",
            answer=str(row.get("answer") or row.get("solution", "")).strip(),
            kind="exact",
        ))
    return items


def load_aime(n: int = 60) -> list[Item]:
    ds = _try_load("Maxwell-Jia/AIME_2024", split="train")
    if ds is None:
        return []
    items = []
    for row in list(ds)[:n]:
        items.append(Item(
            prompt=f"Solve. The answer is an integer 0-999. Final line: Answer: <int>\n\n{row['Problem']}",
            answer=str(row["Answer"]).strip(),
            kind="exact",
        ))
    return items


def load_gpqa(n: int = 198) -> list[Item]:
    ds = _try_load("Idavidrein/gpqa", "gpqa_diamond", split="train")
    if ds is None:
        return []
    items = []
    for row in list(ds)[:n]:
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        # Correct is index 0 here; shuffle deterministically by question hash.
        order = sorted(range(4), key=lambda i: hash((row["Question"], i)))
        shuffled = [choices[i] for i in order]
        gold = _MC_LETTERS[order.index(0)]
        items.append(Item(_mc_prompt(row["Question"], shuffled), gold, "mc"))
    return items


def load_bbh(n: int = 200) -> list[Item]:
    ds = _try_load("lukaemon/bbh", "boolean_expressions", split="test")
    if ds is None:
        return []
    return [Item(f"{row['input']}\n\nFinal line: Answer: <value>", str(row["target"]).strip(), "exact")
            for row in list(ds)[:n]]


def load_truthfulqa(n: int = 200) -> list[Item]:
    ds = _try_load("truthful_qa", "multiple_choice", split="validation")
    if ds is None:
        return []
    items = []
    for row in list(ds)[:n]:
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        gold = _MC_LETTERS[labels.index(1)]
        items.append(Item(_mc_prompt(row["question"], choices), gold, "mc"))
    return items


def load_emobench(n: int = 200) -> list[Item]:
    ds = _try_load("Sabour/EmoBench", split="test")
    if ds is None:
        return []
    items = []
    for row in list(ds)[:n]:
        choices = row.get("choices") or row.get("options")
        if not choices:
            continue
        gold = _MC_LETTERS[int(row["answer"])] if str(row["answer"]).isdigit() else str(row["answer"]).strip()
        items.append(Item(_mc_prompt(row["question"], choices), gold, "mc"))
    return items


LOADERS = {
    "aime": load_aime, "math": load_math, "gpqa": load_gpqa,
    "bbh": load_bbh, "truthfulqa": load_truthfulqa, "emobench": load_emobench,
}


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def _extract_answer(text: str, kind: str) -> str:
    m = re.search(r"Answer:\s*(.+)", text, flags=re.IGNORECASE)
    raw = (m.group(1) if m else text).strip().splitlines()[0].strip() if text else ""
    if kind == "mc":
        lm = re.search(r"[A-F]", raw.upper())
        return lm.group(0) if lm else ""
    # exact: normalise numbers / boxed answers
    raw = raw.replace("$", "").replace("\\boxed{", "").replace("}", "").strip()
    return raw


def _correct(pred: str, gold: str, kind: str) -> bool:
    if kind == "mc":
        return pred.upper() == gold.upper()
    # numeric exact match with light normalisation
    def norm(s):
        s = s.strip().rstrip(".")
        try:
            return str(int(float(s)))
        except ValueError:
            return s.lower()
    return norm(pred) == norm(gold)


def evaluate(model: ModelBackend, benchmark: str, items: list[Item]) -> dict:
    correct = 0
    for it in tqdm(items, desc=f"{model.name}:{benchmark}"):
        out = model.generate([{"role": "user", "content": it.prompt}], n=1,
                             temperature=0.0, max_new_tokens=config.MAX_NEW_TOKENS)[0]
        if _correct(_extract_answer(out, it.kind), it.answer, it.kind):
            correct += 1
    n = len(items) or 1
    return {"benchmark": benchmark, "model": model.name, "n": len(items),
            "accuracy": correct / n}


def run_capabilities(model_name: str, lora_path: str | None = None,
                     benchmarks=config.CAPABILITY_BENCHMARKS) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(model_name, lora_path=lora_path)
    results = []
    for bench in benchmarks:
        items = LOADERS[bench]()
        if not items:
            print(f"[capabilities] skipping {bench} (no data)")
            continue
        results.append(evaluate(model, bench, items))
    out = OUTPUT_DIR / f"{model_name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[capabilities] wrote -> {out}")
    return out
